#!/usr/bin/env python3
"""Export Aqua-D4RT query-level 3D predictions as filtered point clouds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_aqua_transient_heads import (  # noqa: E402
    _binary_metrics,
    _grid_queries,
    _load_clip,
    _load_model,
    _sigmoid,
)
from infer_track_3d import _resolve_device  # noqa: E402
from src.core import load_yaml_config, seed_everything  # noqa: E402
from src.eval.tasks import _encode_model_memory, _run_model_for_queries  # noqa: E402


def _write_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    props: dict[str, np.ndarray] | None = None,
    comments: list[str] | None = None,
) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    if colors.shape != points.shape:
        raise ValueError(f"colors must have shape [N, 3], got {colors.shape}")
    props = props or {}
    prop_arrays = {name: np.asarray(value).reshape(-1) for name, value in props.items()}
    for name, value in prop_arrays.items():
        if value.shape[0] != points.shape[0]:
            raise ValueError(f"property {name!r} has {value.shape[0]} rows, expected {points.shape[0]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        for comment in comments or []:
            f.write(f"comment {comment}\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        for name in prop_arrays:
            f.write(f"property float {name}\n")
        f.write("end_header\n")
        for idx in range(points.shape[0]):
            row = [
                f"{float(points[idx, 0]):.7g}",
                f"{float(points[idx, 1]):.7g}",
                f"{float(points[idx, 2]):.7g}",
                str(int(colors[idx, 0])),
                str(int(colors[idx, 1])),
                str(int(colors[idx, 2])),
            ]
            for name in prop_arrays:
                row.append(f"{float(prop_arrays[name][idx]):.7g}")
            f.write(" ".join(row) + "\n")


def _point_props(
    *,
    dynamic_probs: np.ndarray,
    particle_probs: np.ndarray,
    confidence_probs: np.ndarray,
    static_probs: np.ndarray,
    labels_dynamic: np.ndarray,
    labels_particle: np.ndarray,
    pred_dynamic: np.ndarray,
    pred_particle: np.ndarray,
    keep_f1: np.ndarray,
    keep_clean: np.ndarray,
    frame_idx: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "dynamic_prob": dynamic_probs.astype(np.float32),
        "particle_prob": particle_probs.astype(np.float32),
        "confidence_prob": confidence_probs.astype(np.float32),
        "static_confidence": static_probs.astype(np.float32),
        "label_dynamic": labels_dynamic.astype(np.float32),
        "label_particle": labels_particle.astype(np.float32),
        "pred_dynamic": pred_dynamic.astype(np.float32),
        "pred_particle": pred_particle.astype(np.float32),
        "keep_static_f1": keep_f1.astype(np.float32),
        "keep_static_clean": keep_clean.astype(np.float32),
        "frame_idx": frame_idx.astype(np.float32),
        "x_px": x_px.astype(np.float32),
        "y_px": y_px.astype(np.float32),
    }


def _status_colors(
    base_rgb: np.ndarray,
    *,
    dynamic_mask: np.ndarray,
    particle_mask: np.ndarray,
    static_mask: np.ndarray | None = None,
    rejected_color: tuple[int, int, int] = (85, 85, 85),
) -> np.ndarray:
    out = np.asarray(base_rgb, dtype=np.uint8).copy()
    if static_mask is not None:
        out[~static_mask.astype(bool)] = np.asarray(rejected_color, dtype=np.uint8)
    out[dynamic_mask.astype(bool)] = np.asarray([255, 70, 45], dtype=np.uint8)
    out[particle_mask.astype(bool)] = np.asarray([40, 220, 255], dtype=np.uint8)
    both = dynamic_mask.astype(bool) & particle_mask.astype(bool)
    out[both] = np.asarray([255, 210, 45], dtype=np.uint8)
    return out


def _sample_indices(mask: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask.astype(bool))
    if idx.size <= int(max_points):
        return idx
    return np.sort(rng.choice(idx, size=int(max_points), replace=False))


def _robust_limits(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not (np.isfinite(lo) and np.isfinite(hi)) or abs(float(hi - lo)) < 1e-6:
        center = float(np.nanmean(finite))
        return center - 1.0, center + 1.0
    pad = 0.05 * float(hi - lo)
    return float(lo - pad), float(hi + pad)


def _plot_projection(
    *,
    output_path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    masks: dict[str, np.ndarray],
    max_points: int,
    seed: int,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional visualization dependency
        print(f"Skipping point cloud preview; matplotlib is unavailable: {exc}")
        return False

    rng = np.random.default_rng(int(seed))
    panels = [
        ("All D4RT query points", masks["all"], colors),
        ("GT transient labels", masks["gt_transient"], _status_colors(colors, dynamic_mask=masks["gt_dynamic"], particle_mask=masks["gt_particle"])),
        (
            "Pred transient labels",
            masks["pred_transient"],
            _status_colors(colors, dynamic_mask=masks["pred_dynamic"], particle_mask=masks["pred_particle"]),
        ),
        ("Static F1 mode", masks["static_f1"], colors),
        ("Static clean-map mode", masks["static_clean"], colors),
        (
            "Rejected by clean-map",
            masks["all"] & ~masks["static_clean"],
            _status_colors(colors, dynamic_mask=masks["pred_dynamic"], particle_mask=masks["pred_particle"]),
        ),
    ]

    xlim = _robust_limits(points[:, 0])
    zlim = _robust_limits(points[:, 2])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150)
    for ax, (title, mask, panel_colors) in zip(axes.reshape(-1), panels):
        sample = _sample_indices(mask, max_points=max(1, int(max_points)), rng=rng)
        if sample.size > 0:
            ax.scatter(
                points[sample, 0],
                points[sample, 2],
                c=np.asarray(panel_colors, dtype=np.float32)[sample] / 255.0,
                s=1.0,
                linewidths=0,
                alpha=0.88,
            )
        ax.set_title(f"{title}\nN={int(mask.sum())}", fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_xlim(*xlim)
        ax.set_ylim(*zlim)
        ax.grid(True, linewidth=0.25, alpha=0.35)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Aqua-D4RT query-level 3D filtering preview (x-z projection)", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path.exists() and output_path.stat().st_size > 0


def _filter_metrics(static_probs: np.ndarray, labels_transient: np.ndarray, threshold: float) -> dict[str, float]:
    keep = static_probs >= float(threshold)
    transient = labels_transient.astype(bool)
    static = ~transient
    kept = int(keep.sum())
    kept_static = int(np.logical_and(keep, static).sum())
    kept_transient = int(np.logical_and(keep, transient).sum())
    return {
        "threshold": float(threshold),
        "kept": kept,
        "kept_rate": kept / float(max(1, keep.size)),
        "kept_static": kept_static,
        "kept_transient": kept_transient,
        "contamination": kept_transient / float(max(1, kept)),
        "static_retention": kept_static / float(max(1, int(static.sum()))),
        "static_precision": 1.0 - kept_transient / float(max(1, kept)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--grid-stride", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--dynamic-threshold", type=float, default=0.79)
    parser.add_argument("--particle-threshold", type=float, default=0.83)
    parser.add_argument("--static-f1-threshold", type=float, default=0.11)
    parser.add_argument("--static-clean-threshold", type=float, default=0.55)
    parser.add_argument("--min-visibility-prob", type=float, default=0.0)
    parser.add_argument("--min-confidence-prob", type=float, default=0.0)
    parser.add_argument("--preview-max-points", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    device = _resolve_device(args.device)
    cfg = load_yaml_config(args.model_config)
    image_hw = tuple(int(v) for v in cfg["model"]["input"].get("image_size", [256, 256]))
    manifest_path = Path(args.manifest)
    video, dynamic_mask, particle_mask, manifest = _load_clip(
        manifest_path,
        image_hw=image_hw,
        max_frames=int(args.max_frames),
    )
    query_cpu, coord_txy = _grid_queries(video.shape[0], video.shape[1], video.shape[2], stride=int(args.grid_stride))
    labels_dynamic = dynamic_mask[coord_txy[:, 0], coord_txy[:, 2], coord_txy[:, 1]]
    labels_particle = particle_mask[coord_txy[:, 0], coord_txy[:, 2], coord_txy[:, 1]]
    labels_transient = labels_dynamic | labels_particle
    rgb = video[coord_txy[:, 0], coord_txy[:, 2], coord_txy[:, 1]]

    model = _load_model(Path(args.model_config), Path(args.ckpt_path), device=device)
    video_b = torch.from_numpy(video).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
    aspect = torch.tensor(
        [[float(manifest.get("width", video.shape[2])) / float(max(1, manifest.get("height", video.shape[1])))]],
        device=device,
    )
    query = {key: value.to(device=device) for key, value in query_cpu.items()}
    with torch.no_grad():
        memory = _encode_model_memory(model=model, video_b=video_b, aspect_b=aspect)
        pred = _run_model_for_queries(
            model=model,
            video_b=video_b,
            aspect_b=aspect,
            query=query,
            chunk_size=int(args.query_chunk_size),
            memory_b=memory,
        )

    xyz = pred["xyz_3d"].numpy().astype(np.float32)
    dynamic_probs = _sigmoid(pred["dynamic_object_logit"].numpy())
    particle_probs = _sigmoid(pred["particle_logit"].numpy())
    confidence_probs = _sigmoid(pred["confidence"].numpy())
    visibility_probs = _sigmoid(pred["visibility"].numpy())
    if "static_confidence" in pred:
        static_probs = pred["static_confidence"].numpy().astype(np.float32)
    else:
        static_probs = confidence_probs * (1.0 - dynamic_probs) * (1.0 - particle_probs)

    finite = np.isfinite(xyz).all(axis=1)
    finite &= np.isfinite(static_probs)
    finite &= visibility_probs >= float(args.min_visibility_prob)
    finite &= confidence_probs >= float(args.min_confidence_prob)
    pred_dynamic = dynamic_probs >= float(args.dynamic_threshold)
    pred_particle = particle_probs >= float(args.particle_threshold)
    pred_transient = pred_dynamic | pred_particle
    keep_f1 = static_probs >= float(args.static_f1_threshold)
    keep_clean = static_probs >= float(args.static_clean_threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comments = [
        "Aqua-D4RT query-level export; xyz are raw D4RT 3D query predictions.",
        f"manifest={manifest_path.resolve()}",
        f"checkpoint={Path(args.ckpt_path).resolve()}",
        f"grid_stride={int(args.grid_stride)}",
    ]

    all_props = _point_props(
        dynamic_probs=dynamic_probs,
        particle_probs=particle_probs,
        confidence_probs=confidence_probs,
        static_probs=static_probs,
        labels_dynamic=labels_dynamic,
        labels_particle=labels_particle,
        pred_dynamic=pred_dynamic,
        pred_particle=pred_particle,
        keep_f1=keep_f1,
        keep_clean=keep_clean,
        frame_idx=coord_txy[:, 0],
        x_px=coord_txy[:, 1],
        y_px=coord_txy[:, 2],
    )

    clouds = {
        "all_points": finite,
        "gt_transient_points": finite & labels_transient,
        "pred_transient_points": finite & pred_transient,
        "static_f1_points": finite & keep_f1,
        "static_clean_points": finite & keep_clean,
        "rejected_by_static_clean_points": finite & ~keep_clean,
    }
    cloud_files: dict[str, str] = {}
    for name, mask in clouds.items():
        if name == "gt_transient_points":
            colors = _status_colors(rgb, dynamic_mask=labels_dynamic, particle_mask=labels_particle)
        elif name in {"pred_transient_points", "rejected_by_static_clean_points"}:
            colors = _status_colors(rgb, dynamic_mask=pred_dynamic, particle_mask=pred_particle)
        elif name == "static_f1_points":
            colors = _status_colors(rgb, dynamic_mask=pred_dynamic, particle_mask=pred_particle, static_mask=keep_f1)
        elif name == "static_clean_points":
            colors = _status_colors(rgb, dynamic_mask=pred_dynamic, particle_mask=pred_particle, static_mask=keep_clean)
        else:
            colors = rgb
        selected_props = {key: value[mask] for key, value in all_props.items()}
        ply_path = output_dir / f"{name}.ply"
        _write_ply(ply_path, xyz[mask], colors[mask], props=selected_props, comments=comments)
        cloud_files[name] = str(ply_path.resolve())

    preview_path = output_dir / "pointcloud_preview.png"
    preview_ok = _plot_projection(
        output_path=preview_path,
        points=xyz[finite],
        colors=rgb[finite],
        masks={
            "all": np.ones(int(finite.sum()), dtype=bool),
            "gt_transient": labels_transient[finite],
            "gt_dynamic": labels_dynamic[finite],
            "gt_particle": labels_particle[finite],
            "pred_transient": pred_transient[finite],
            "pred_dynamic": pred_dynamic[finite],
            "pred_particle": pred_particle[finite],
            "static_f1": keep_f1[finite],
            "static_clean": keep_clean[finite],
        },
        max_points=int(args.preview_max_points),
        seed=int(args.seed),
    )

    summary: dict[str, Any] = {
        "inputs": {
            "manifest": str(manifest_path.resolve()),
            "model_config": str(Path(args.model_config).resolve()),
            "ckpt_path": str(Path(args.ckpt_path).resolve()),
            "num_frames": int(video.shape[0]),
            "image_hw": [int(video.shape[1]), int(video.shape[2])],
            "grid_stride": int(args.grid_stride),
            "num_queries": int(coord_txy.shape[0]),
            "num_finite_queries": int(finite.sum()),
        },
        "thresholds": {
            "dynamic_object": float(args.dynamic_threshold),
            "particle": float(args.particle_threshold),
            "static_f1": float(args.static_f1_threshold),
            "static_clean": float(args.static_clean_threshold),
            "min_visibility_prob": float(args.min_visibility_prob),
            "min_confidence_prob": float(args.min_confidence_prob),
        },
        "cloud_counts": {name: int(mask.sum()) for name, mask in clouds.items()},
        "mask_coverage_eval_grid": {
            "dynamic_object": float(labels_dynamic[finite].mean()) if np.any(finite) else 0.0,
            "particle": float(labels_particle[finite].mean()) if np.any(finite) else 0.0,
            "transient": float(labels_transient[finite].mean()) if np.any(finite) else 0.0,
        },
        "dynamic_object": {
            "calibrated_threshold": _binary_metrics(dynamic_probs[finite], labels_dynamic[finite], threshold=float(args.dynamic_threshold))
            if np.any(finite)
            else {},
        },
        "particle": {
            "calibrated_threshold": _binary_metrics(particle_probs[finite], labels_particle[finite], threshold=float(args.particle_threshold))
            if np.any(finite)
            else {},
        },
        "static_filtering": {
            "f1_mode": _filter_metrics(static_probs[finite], labels_transient[finite], threshold=float(args.static_f1_threshold))
            if np.any(finite)
            else {},
            "clean_map_mode": _filter_metrics(static_probs[finite], labels_transient[finite], threshold=float(args.static_clean_threshold))
            if np.any(finite)
            else {},
        },
        "score_stats": {
            "dynamic_prob_mean": float(np.mean(dynamic_probs[finite])) if np.any(finite) else 0.0,
            "particle_prob_mean": float(np.mean(particle_probs[finite])) if np.any(finite) else 0.0,
            "confidence_prob_mean": float(np.mean(confidence_probs[finite])) if np.any(finite) else 0.0,
            "visibility_prob_mean": float(np.mean(visibility_probs[finite])) if np.any(finite) else 0.0,
            "static_confidence_mean": float(np.mean(static_probs[finite])) if np.any(finite) else 0.0,
        },
        "outputs": {
            "pointcloud_preview": str(preview_path.resolve()) if preview_ok else None,
            "ply_files": cloud_files,
        },
        "interpretation_note": (
            "These are D4RT query-level 3D predictions filtered by Aqua transient scores. "
            "They are not yet a globally fused SLAM map unless a downstream pose/fusion stage is applied."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved point clouds to: {output_dir}")
    print(f"Saved preview: {preview_path if preview_ok else 'FAILED'}")
    print(
        "Static clean-map: "
        f"kept={summary['static_filtering']['clean_map_mode'].get('kept', 0)} "
        f"contamination={summary['static_filtering']['clean_map_mode'].get('contamination', 0.0):.4f} "
        f"retention={summary['static_filtering']['clean_map_mode'].get('static_retention', 0.0):.4f}"
    )
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
