#!/usr/bin/env python3
"""Evaluate Aqua-D4RT against OpenD4RT RGB-prefilter baselines."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_aqua_transient_heads import (  # noqa: E402
    _best_f1,
    _binary_metrics,
    _grid_queries,
    _load_clip,
    _load_model,
    _load_rgb,
    _sigmoid,
)
from infer_track_3d import _resolve_device  # noqa: E402
from src.core import load_yaml_config, seed_everything  # noqa: E402
from src.eval.tasks import _encode_model_memory, _run_model_for_queries  # noqa: E402
from aqua_prefilter_utils import (  # noqa: E402
    binary_mask_metrics,
    inpaint_video_with_mask,
    query_labels_from_mask,
    temporal_rgb_pseudo_mask,
)


BASE_SYSTEMS = (
    "opend4rt_raw_confidence",
    "opend4rt_oracle_mask_prefilter",
    "opend4rt_oracle_clean_confidence",
    "aqua_raw_static_confidence",
)
TEMPORAL_RGB_SYSTEMS = (
    "opend4rt_temporal_rgb_mask_prefilter",
    "opend4rt_temporal_rgb_clean_confidence",
)


def _read_manifest_list(path: str | Path) -> list[str]:
    items: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            items.append(value)
    return items


def _resolve_manifests(args: argparse.Namespace) -> list[Path]:
    items: list[str] = []
    if args.manifest:
        for value in args.manifest:
            items.extend(part.strip() for part in str(value).split(",") if part.strip())
    if args.manifest_list:
        for value in args.manifest_list:
            items.extend(_read_manifest_list(value))
    if not items:
        raise ValueError("Provide --manifest or --manifest-list")
    out: list[Path] = []
    seen: set[str] = set()
    for item in items:
        path = Path(item)
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _load_clean_video(manifest: dict[str, Any], image_hw: tuple[int, int], max_frames: int) -> np.ndarray | None:
    paths = manifest.get("clean_frames")
    if not isinstance(paths, list) or not paths:
        return None
    if max_frames > 0:
        paths = paths[: int(max_frames)]
    return np.stack([_load_rgb(path, image_hw=image_hw) for path in paths], axis=0)


def _oracle_clean_video(
    video: np.ndarray,
    clean_video: np.ndarray | None,
    transient_mask: np.ndarray,
) -> tuple[np.ndarray, str]:
    mask = transient_mask.astype(bool)
    if clean_video is not None and clean_video.shape == video.shape:
        out = video.copy()
        out[mask] = clean_video[mask]
        return out, "clean_frame_replacement"

    return inpaint_video_with_mask(video, mask), "opencv_telea_inpaint"


def _predict(
    *,
    model: torch.nn.Module,
    video: np.ndarray,
    manifest: dict[str, Any],
    query_cpu: dict[str, torch.Tensor],
    device: torch.device,
    query_chunk_size: int,
) -> dict[str, np.ndarray]:
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
            chunk_size=int(query_chunk_size),
            memory_b=memory,
        )
    return {
        key: value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        for key, value in pred.items()
    }


def _static_curve(static_probs: np.ndarray, transient_labels: np.ndarray, thresholds: np.ndarray) -> list[dict[str, Any]]:
    transient = transient_labels.astype(bool)
    static_label = ~transient
    n_total = int(static_probs.shape[0])
    n_static = int(static_label.sum())
    out: list[dict[str, Any]] = []
    for threshold in thresholds:
        keep = static_probs >= float(threshold)
        kept = int(keep.sum())
        kept_static = int(np.logical_and(keep, static_label).sum())
        kept_transient = int(np.logical_and(keep, transient).sum())
        contamination = kept_transient / float(max(1, kept))
        static_retention = kept_static / float(max(1, n_static))
        out.append(
            {
                "threshold": float(threshold),
                "kept": kept,
                "kept_rate": kept / float(max(1, n_total)),
                "kept_static": kept_static,
                "kept_transient": kept_transient,
                "contamination": float(contamination),
                "static_retention": float(static_retention),
                "static_precision": float(1.0 - contamination),
            }
        )
    return out


def _clean_map_point(curve: list[dict[str, Any]], max_contamination: float) -> dict[str, Any]:
    candidates = [item for item in curve if float(item["contamination"]) <= float(max_contamination)]
    if not candidates:
        return {
            "target_max_contamination": float(max_contamination),
            "found": False,
            "reason": "no threshold satisfies contamination target",
        }
    best = max(
        candidates,
        key=lambda item: (
            float(item["static_retention"]),
            float(item["kept_rate"]),
            float(item["threshold"]),
        ),
    )
    out = dict(best)
    out["target_max_contamination"] = float(max_contamination)
    out["found"] = True
    return out


def _summarize_scores(
    scores: np.ndarray,
    labels_transient: np.ndarray,
    thresholds: np.ndarray,
    clean_map_max_contamination: float,
) -> dict[str, Any]:
    static_labels = ~labels_transient.astype(bool)
    curve = _static_curve(scores, labels_transient, thresholds)
    return {
        "threshold_0_5": _binary_metrics(scores, static_labels, threshold=0.5),
        "best_static_f1": _best_f1(scores, static_labels),
        "clean_map_operating_point": _clean_map_point(curve, clean_map_max_contamination),
        "static_contamination_curve": curve,
        "score_stats": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        },
    }


def _manifest_name(path: Path) -> str:
    parent = path.parent.name
    return parent if parent else path.stem


def _evaluate_base_model(
    *,
    manifests: list[Path],
    model_config: Path,
    base_ckpt_path: Path,
    image_hw: tuple[int, int],
    device: torch.device,
    max_frames: int,
    grid_stride: int,
    query_chunk_size: int,
    systems: tuple[str, ...],
    enable_temporal_rgb_prefilter: bool,
    temporal_prefilter_kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[np.ndarray]], dict[str, Any]]:
    model = _load_model(model_config, base_ckpt_path, device=device)
    per_clip: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in systems}
    arrays["labels_dynamic"] = []
    arrays["labels_particle"] = []
    arrays["labels_transient"] = []
    clean_method_counts: dict[str, int] = {}
    temporal_method_counts: dict[str, int] = {}

    for idx, manifest_path in enumerate(manifests):
        print(f"[base {idx + 1}/{len(manifests)}] {manifest_path}")
        video, dynamic_mask, particle_mask, manifest = _load_clip(
            manifest_path,
            image_hw=image_hw,
            max_frames=int(max_frames),
        )
        query_cpu, coord_txy = _grid_queries(video.shape[0], video.shape[1], video.shape[2], stride=int(grid_stride))
        labels_dynamic = dynamic_mask[coord_txy[:, 0], coord_txy[:, 2], coord_txy[:, 1]].astype(bool)
        labels_particle = particle_mask[coord_txy[:, 0], coord_txy[:, 2], coord_txy[:, 1]].astype(bool)
        labels_transient = labels_dynamic | labels_particle

        raw_pred = _predict(
            model=model,
            video=video,
            manifest=manifest,
            query_cpu=query_cpu,
            device=device,
            query_chunk_size=int(query_chunk_size),
        )
        raw_conf = _sigmoid(raw_pred["confidence"]).astype(np.float32)

        clean_video = _load_clean_video(manifest, image_hw=image_hw, max_frames=int(max_frames))
        oracle_clean, clean_method = _oracle_clean_video(video, clean_video, dynamic_mask | particle_mask)
        clean_method_counts[clean_method] = clean_method_counts.get(clean_method, 0) + 1
        clean_pred = _predict(
            model=model,
            video=oracle_clean,
            manifest=manifest,
            query_cpu=query_cpu,
            device=device,
            query_chunk_size=int(query_chunk_size),
        )
        clean_conf = _sigmoid(clean_pred["confidence"]).astype(np.float32)

        arrays["opend4rt_raw_confidence"].append(raw_conf)
        arrays["opend4rt_oracle_mask_prefilter"].append((raw_conf * (~labels_transient).astype(np.float32)).astype(np.float32))
        arrays["opend4rt_oracle_clean_confidence"].append(clean_conf)
        arrays["labels_dynamic"].append(labels_dynamic)
        arrays["labels_particle"].append(labels_particle)
        arrays["labels_transient"].append(labels_transient)

        item = {
            "manifest": str(manifest_path.resolve()),
            "clip": _manifest_name(manifest_path),
            "num_queries": int(labels_transient.shape[0]),
            "mask_coverage_eval_grid": {
                "dynamic_object": float(labels_dynamic.mean()),
                "particle": float(labels_particle.mean()),
                "transient": float(labels_transient.mean()),
            },
            "oracle_clean_method": clean_method,
            "base_score_stats": {
                "raw_conf_mean": float(raw_conf.mean()),
                "raw_conf_std": float(raw_conf.std()),
                "oracle_clean_conf_mean": float(clean_conf.mean()),
                "oracle_clean_conf_std": float(clean_conf.std()),
            },
        }
        if enable_temporal_rgb_prefilter:
            pseudo_mask = temporal_rgb_pseudo_mask(video, **temporal_prefilter_kwargs)
            pseudo_labels = query_labels_from_mask(pseudo_mask, coord_txy)
            arrays["opend4rt_temporal_rgb_mask_prefilter"].append(
                (raw_conf * (~pseudo_labels).astype(np.float32)).astype(np.float32)
            )
            pseudo_clean = inpaint_video_with_mask(video, pseudo_mask)
            pseudo_pred = _predict(
                model=model,
                video=pseudo_clean,
                manifest=manifest,
                query_cpu=query_cpu,
                device=device,
                query_chunk_size=int(query_chunk_size),
            )
            pseudo_conf = _sigmoid(pseudo_pred["confidence"]).astype(np.float32)
            arrays["opend4rt_temporal_rgb_clean_confidence"].append(pseudo_conf)
            temporal_method_counts["temporal_rgb_median_residual"] = temporal_method_counts.get("temporal_rgb_median_residual", 0) + 1
            item["temporal_rgb_prefilter"] = {
                "method": "temporal_rgb_median_residual",
                "dense_mask_metrics_vs_gt_transient": binary_mask_metrics(pseudo_mask, dynamic_mask | particle_mask),
                "query_mask_metrics_vs_gt_transient": binary_mask_metrics(pseudo_labels, labels_transient),
                "pseudo_mask_coverage_dense": float(pseudo_mask.mean()),
                "pseudo_mask_coverage_eval_grid": float(pseudo_labels.mean()),
                "pseudo_clean_conf_mean": float(pseudo_conf.mean()),
                "pseudo_clean_conf_std": float(pseudo_conf.std()),
            }
        per_clip.append(item)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return per_clip, arrays, {
        "oracle_clean_method_counts": clean_method_counts,
        "temporal_rgb_prefilter_method_counts": temporal_method_counts,
    }


def _evaluate_aqua_model(
    *,
    manifests: list[Path],
    model_config: Path,
    aqua_ckpt_path: Path,
    image_hw: tuple[int, int],
    device: torch.device,
    max_frames: int,
    grid_stride: int,
    query_chunk_size: int,
    arrays: dict[str, list[np.ndarray]],
    per_clip: list[dict[str, Any]],
) -> None:
    model = _load_model(model_config, aqua_ckpt_path, device=device)
    for idx, manifest_path in enumerate(manifests):
        print(f"[aqua {idx + 1}/{len(manifests)}] {manifest_path}")
        video, _, _, manifest = _load_clip(
            manifest_path,
            image_hw=image_hw,
            max_frames=int(max_frames),
        )
        query_cpu, _ = _grid_queries(video.shape[0], video.shape[1], video.shape[2], stride=int(grid_stride))
        pred = _predict(
            model=model,
            video=video,
            manifest=manifest,
            query_cpu=query_cpu,
            device=device,
            query_chunk_size=int(query_chunk_size),
        )
        dynamic_probs = _sigmoid(pred["dynamic_object_logit"]).astype(np.float32)
        particle_probs = _sigmoid(pred["particle_logit"]).astype(np.float32)
        if "static_confidence" in pred:
            static_probs = pred["static_confidence"].astype(np.float32)
        else:
            static_probs = (
                _sigmoid(pred["confidence"]) * (1.0 - dynamic_probs) * (1.0 - particle_probs)
            ).astype(np.float32)
        arrays["aqua_raw_static_confidence"].append(static_probs)
        per_clip[idx]["aqua_score_stats"] = {
            "dynamic_prob_mean": float(dynamic_probs.mean()),
            "particle_prob_mean": float(particle_probs.mean()),
            "static_confidence_mean": float(static_probs.mean()),
        }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def _aggregate(
    arrays: dict[str, list[np.ndarray]],
    thresholds: np.ndarray,
    clean_map_max_contamination: float,
    systems: tuple[str, ...],
) -> dict[str, Any]:
    labels_dynamic = np.concatenate(arrays["labels_dynamic"], axis=0).astype(bool)
    labels_particle = np.concatenate(arrays["labels_particle"], axis=0).astype(bool)
    labels_transient = np.concatenate(arrays["labels_transient"], axis=0).astype(bool)
    out: dict[str, Any] = {
        "num_queries": int(labels_transient.shape[0]),
        "mask_coverage_eval_grid": {
            "dynamic_object": float(labels_dynamic.mean()),
            "particle": float(labels_particle.mean()),
            "transient": float(labels_transient.mean()),
        },
        "systems": {},
    }
    for system in systems:
        scores = np.concatenate(arrays[system], axis=0).astype(np.float32)
        summary = _summarize_scores(
            scores=scores,
            labels_transient=labels_transient,
            thresholds=thresholds,
            clean_map_max_contamination=float(clean_map_max_contamination),
        )
        if system == "opend4rt_oracle_clean_confidence":
            # This diagnostic keeps the original transient-coordinate labels so
            # we can see whether a cleaned RGB input still activates those
            # coordinates. It is not the main contamination metric because the
            # oracle-clean video has already replaced dynamic pixels.
            summary["interpretation_note"] = (
                "Diagnostic only: metrics use original transient coordinates even though "
                "the RGB input was oracle-cleaned."
            )
        if system == "opend4rt_oracle_mask_prefilter":
            summary["interpretation_note"] = (
                "Oracle upper bound: GT transient query coordinates are discarded before "
                "static aggregation, so contamination can be zero by construction."
            )
        if system == "opend4rt_temporal_rgb_mask_prefilter":
            summary["interpretation_note"] = (
                "Non-oracle RGB baseline: temporal RGB pseudo-mask coordinates are discarded "
                "before static aggregation. The mask is generated without GT labels."
            )
        if system == "opend4rt_temporal_rgb_clean_confidence":
            summary["interpretation_note"] = (
                "Non-oracle RGB baseline: temporal RGB pseudo-mask regions are inpainted "
                "before running OpenD4RT. Metrics still use original transient coordinates."
            )
        out["systems"][system] = summary
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=None)
    parser.add_argument("--manifest-list", action="append", default=None)
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--base-ckpt-path", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt")
    parser.add_argument("--aqua-ckpt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--grid-stride", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--clean-map-max-contamination", type=float, default=0.005)
    parser.add_argument("--enable-temporal-rgb-prefilter", action="store_true")
    parser.add_argument("--temporal-prefilter-percentile", type=float, default=92.0)
    parser.add_argument("--temporal-prefilter-min-threshold", type=float, default=18.0)
    parser.add_argument("--temporal-prefilter-blur-kernel", type=int, default=5)
    parser.add_argument("--temporal-prefilter-morph-kernel", type=int, default=5)
    parser.add_argument("--temporal-prefilter-dilate-iterations", type=int, default=1)
    parser.add_argument("--temporal-prefilter-min-component-area", type=int, default=20)
    parser.add_argument("--temporal-prefilter-max-mask-fraction", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    device = _resolve_device(args.device)
    model_config = Path(args.model_config)
    base_ckpt_path = Path(args.base_ckpt_path)
    aqua_ckpt_path = Path(args.aqua_ckpt_path)
    cfg = load_yaml_config(model_config)
    image_hw = tuple(int(v) for v in cfg["model"]["input"].get("image_size", [256, 256]))
    manifests = _resolve_manifests(args)
    systems = tuple(list(BASE_SYSTEMS) + (list(TEMPORAL_RGB_SYSTEMS) if bool(args.enable_temporal_rgb_prefilter) else []))
    temporal_prefilter_kwargs = {
        "percentile": float(args.temporal_prefilter_percentile),
        "min_threshold": float(args.temporal_prefilter_min_threshold),
        "blur_kernel": int(args.temporal_prefilter_blur_kernel),
        "morph_kernel": int(args.temporal_prefilter_morph_kernel),
        "dilate_iterations": int(args.temporal_prefilter_dilate_iterations),
        "min_component_area": int(args.temporal_prefilter_min_component_area),
        "max_mask_fraction": float(args.temporal_prefilter_max_mask_fraction),
    }
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, num=99, dtype=np.float32),
                np.asarray([0.11, 0.5, 0.55], dtype=np.float32),
            ]
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_clip, arrays, extra = _evaluate_base_model(
        manifests=manifests,
        model_config=model_config,
        base_ckpt_path=base_ckpt_path,
        image_hw=image_hw,
        device=device,
        max_frames=int(args.max_frames),
        grid_stride=int(args.grid_stride),
        query_chunk_size=int(args.query_chunk_size),
        systems=systems,
        enable_temporal_rgb_prefilter=bool(args.enable_temporal_rgb_prefilter),
        temporal_prefilter_kwargs=temporal_prefilter_kwargs,
    )
    _evaluate_aqua_model(
        manifests=manifests,
        model_config=model_config,
        aqua_ckpt_path=aqua_ckpt_path,
        image_hw=image_hw,
        device=device,
        max_frames=int(args.max_frames),
        grid_stride=int(args.grid_stride),
        query_chunk_size=int(args.query_chunk_size),
        arrays=arrays,
        per_clip=per_clip,
    )
    aggregate = _aggregate(
        arrays=arrays,
        thresholds=thresholds,
        clean_map_max_contamination=float(args.clean_map_max_contamination),
        systems=systems,
    )
    aggregate.update(
        {
            "inputs": {
                "model_config": str(model_config.resolve()),
                "base_ckpt_path": str(base_ckpt_path.resolve()),
                "aqua_ckpt_path": str(aqua_ckpt_path.resolve()),
                "num_clips": len(manifests),
                "manifests": [str(path.resolve()) for path in manifests],
                "max_frames": int(args.max_frames),
                "grid_stride": int(args.grid_stride),
                "image_hw": [int(image_hw[0]), int(image_hw[1])],
                "clean_map_max_contamination": float(args.clean_map_max_contamination),
                "systems": list(systems),
                "enable_temporal_rgb_prefilter": bool(args.enable_temporal_rgb_prefilter),
                "temporal_rgb_prefilter": temporal_prefilter_kwargs if bool(args.enable_temporal_rgb_prefilter) else None,
            },
            **extra,
        }
    )
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    with (output_dir / "per_clip_metrics.jsonl").open("w", encoding="utf-8") as f:
        for item in per_clip:
            f.write(json.dumps(item) + "\n")

    table = []
    for system, metrics in aggregate["systems"].items():
        clean = metrics["clean_map_operating_point"]
        best = metrics["best_static_f1"]
        table.append(
            {
                "system": system,
                "best_static_f1": best["f1"],
                "best_static_threshold": best["threshold"],
                "clean_found": clean.get("found", False),
                "clean_threshold": clean.get("threshold"),
                "clean_contamination": clean.get("contamination"),
                "clean_static_retention": clean.get("static_retention"),
                "clean_kept_rate": clean.get("kept_rate"),
            }
        )
    (output_dir / "summary_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"Saved aggregate: {output_dir / 'aggregate_metrics.json'}")
    print(f"Saved per-clip metrics: {output_dir / 'per_clip_metrics.jsonl'}")
    print(f"Saved table: {output_dir / 'summary_table.json'}")
    for row in table:
        print(
            f"{row['system']}: best_static_f1={row['best_static_f1']:.4f}@{row['best_static_threshold']:.2f} "
            f"clean_retention={row['clean_static_retention']} clean_contam={row['clean_contamination']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
