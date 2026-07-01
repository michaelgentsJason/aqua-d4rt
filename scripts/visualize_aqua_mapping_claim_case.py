#!/usr/bin/env python3
"""Visualize the main Aqua-D4RT mapping claim for one clip.

This script intentionally avoids presenting Aqua as a pixel-perfect fish
segmenter.  It shows the query-level static map candidates and the feature
front-end cleanliness that are closer to the paper claim.
"""

from __future__ import annotations

import argparse
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

from eval_aqua_downstream_slam_proxy import _detect_features, _make_detector, _sample_mask  # noqa: E402
from eval_aqua_transient_heads import _grid_queries, _load_clip, _load_model, _load_rgb, _sigmoid  # noqa: E402
from infer_track_3d import _resolve_device  # noqa: E402
from src.core import load_yaml_config, seed_everything  # noqa: E402
from src.eval.tasks import _encode_model_memory, _run_model_for_queries  # noqa: E402


def _safe_rate(num: int | float, den: int | float) -> float:
    return float(num) / float(max(float(den), 1.0))


def _parse_frame_ids(value: str, num_frames: int) -> list[int]:
    if value.strip():
        items = [int(item) for item in value.split(",") if item.strip()]
    else:
        items = np.linspace(0, max(0, num_frames - 1), num=min(4, num_frames)).round().astype(int).tolist()
    return sorted({int(np.clip(item, 0, max(0, num_frames - 1))) for item in items})


def _load_clean_video(manifest: dict[str, Any], image_hw: tuple[int, int], max_frames: int) -> np.ndarray | None:
    paths = manifest.get("clean_frames")
    if not isinstance(paths, list) or not paths:
        return None
    if max_frames > 0:
        paths = paths[: int(max_frames)]
    return np.stack([_load_rgb(path, image_hw=image_hw) for path in paths], axis=0)


def _grid_map(
    coord_txy: np.ndarray,
    values: np.ndarray,
    frame_idx: int,
    height: int,
    width: int,
    stride: int,
    *,
    smooth: bool = False,
) -> np.ndarray:
    mask = coord_txy[:, 0] == int(frame_idx)
    out = np.zeros((height, width), dtype=np.float32)
    if not np.any(mask):
        return out
    xy = coord_txy[mask][:, 1:3]
    values_f = values[mask].astype(np.float32)
    if smooth:
        xs = np.unique(xy[:, 0])
        ys = np.unique(xy[:, 1])
        coarse = np.zeros((len(ys), len(xs)), dtype=np.float32)
        x_to_i = {int(x): i for i, x in enumerate(xs)}
        y_to_i = {int(y): i for i, y in enumerate(ys)}
        for (x, y), value in zip(xy, values_f):
            coarse[y_to_i[int(y)], x_to_i[int(x)]] = float(value)
        dense = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
        dense = cv2.GaussianBlur(dense, (7, 7), 0)
        return np.clip(dense, 0.0, 1.0).astype(np.float32)
    out[xy[:, 1], xy[:, 0]] = values_f
    kernel = np.ones((max(3, int(stride)), max(3, int(stride))), np.uint8)
    return cv2.dilate(out, kernel)


def _draw_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    del h
    bar = np.full((30, w, 3), 24, dtype=np.uint8)
    cv2.putText(bar, title, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (238, 238, 238), 1, cv2.LINE_AA)
    return np.concatenate([bar, image_bgr], axis=0)


def _overlay_gt(frame_rgb: np.ndarray, dynamic_mask: np.ndarray, particle_mask: np.ndarray, title: str) -> np.ndarray:
    base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    out = base.copy()
    dyn = dynamic_mask.astype(bool)
    par = particle_mask.astype(bool)
    out[dyn] = (45, 70, 245)
    out[par] = (255, 220, 45)
    both = dyn & par
    out[both] = (220, 60, 220)
    out = cv2.addWeighted(out, 0.44, base, 0.56, 0.0)
    edge = cv2.morphologyEx((dyn | par).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    out[edge] = (35, 35, 255)
    return _draw_title(out, title)


def _heatmap_overlay(frame_rgb: np.ndarray, score_map: np.ndarray, title: str, *, invert: bool = False) -> np.ndarray:
    base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    scores = np.clip(score_map, 0.0, 1.0)
    if invert:
        scores = 1.0 - scores
    heat = cv2.applyColorMap((scores * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    out = cv2.addWeighted(base, 0.68, heat, 0.32, 0.0)
    return _draw_title(out, title)


def _query_panel(
    frame_rgb: np.ndarray,
    coord_txy: np.ndarray,
    labels_transient: np.ndarray,
    keep: np.ndarray,
    frame_idx: int,
    title: str,
    *,
    draw_rejected: bool,
) -> np.ndarray:
    out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    mask = coord_txy[:, 0] == int(frame_idx)
    xy = coord_txy[mask][:, 1:3]
    labels = labels_transient[mask].astype(bool)
    keep_f = keep[mask].astype(bool)
    if draw_rejected:
        rejected_xy = xy[~keep_f]
        for x, y in rejected_xy:
            cv2.circle(out, (int(x), int(y)), 1, (92, 92, 92), -1, cv2.LINE_AA)
    kept_xy = xy[keep_f]
    kept_lab = labels[keep_f]
    for (x, y), is_transient in zip(kept_xy, kept_lab):
        color = (35, 35, 245) if bool(is_transient) else (55, 225, 120)
        cv2.circle(out, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
    return _draw_title(out, title)


def _feature_panel(
    frame_rgb: np.ndarray,
    static_mask: np.ndarray,
    transient_gt: np.ndarray,
    frame_idx: int,
    title: str,
    detector: Any,
    max_features: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    keypoints, _ = _detect_features(
        frame_rgb=frame_rgb,
        static_mask=static_mask.astype(bool),
        detector=detector,
        max_features=int(max_features),
    )
    pts = np.asarray([kp.pt for kp in keypoints], dtype=np.float32) if keypoints else np.zeros((0, 2), dtype=np.float32)
    contam = _sample_mask(transient_gt, pts, frame_idx)
    out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if not bool(static_mask.all()):
        dim = np.clip(out.astype(np.float32) * 0.28 + np.array([28, 28, 28], dtype=np.float32), 0, 255).astype(np.uint8)
        out = np.where(static_mask[..., None].astype(bool), out, dim)
    for pt, is_bad in zip(pts, contam):
        color = (35, 35, 245) if bool(is_bad) else (255, 210, 70)
        cv2.circle(out, (int(round(pt[0])), int(round(pt[1]))), 2, color, 1, cv2.LINE_AA)
    contam_rate = float(contam.mean()) if contam.size else 0.0
    metric = {"num_features": int(len(keypoints)), "feature_contamination": contam_rate}
    panel = _draw_title(out, f"{title}: {len(keypoints)} feat, {100.0 * contam_rate:.1f}% red")
    return panel, metric


def _metrics_box(
    *,
    width: int,
    height: int,
    clip_metrics: dict[str, Any],
    frame_metrics: dict[str, Any],
    note: str,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    y = 24
    lines = [
        "Clip metrics",
        f"Raw query contam: {100.0 * clip_metrics['raw_query_contamination']:.1f}%",
        f"Aqua query contam: {100.0 * clip_metrics['aqua_query_contamination']:.1f}%",
        f"Static retention: {100.0 * clip_metrics['aqua_static_retention']:.1f}%",
        f"Kept queries: {100.0 * clip_metrics['aqua_kept_rate']:.1f}%",
        "",
        "This frame",
        f"Raw ORB contam: {100.0 * frame_metrics['raw_features']['feature_contamination']:.1f}%",
        f"Aqua ORB contam: {100.0 * frame_metrics['aqua_features']['feature_contamination']:.1f}%",
        f"Raw/Aqua feat: {frame_metrics['raw_features']['num_features']}/{frame_metrics['aqua_features']['num_features']}",
        "",
    ]
    lines.extend(note.splitlines())
    for i, line in enumerate(lines):
        if not line:
            y += 10
            continue
        color = (20, 20, 20) if i == 0 else (65, 65, 65)
        scale = 0.50 if len(line) < 34 else 0.43
        cv2.putText(canvas, line[:54], (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += 20
        if y > height - 10:
            break
    return _draw_title(canvas, "Numbers")


def _query_metrics(keep: np.ndarray, labels_transient: np.ndarray) -> dict[str, Any]:
    keep_b = keep.astype(bool)
    transient = labels_transient.astype(bool)
    static = ~transient
    kept = int(keep_b.sum())
    kept_transient = int(np.logical_and(keep_b, transient).sum())
    kept_static = int(np.logical_and(keep_b, static).sum())
    total_static = int(static.sum())
    total_transient = int(transient.sum())
    return {
        "kept": kept,
        "total": int(keep_b.size),
        "kept_rate": _safe_rate(kept, keep_b.size),
        "contamination": _safe_rate(kept_transient, kept),
        "static_retention": _safe_rate(kept_static, total_static),
        "transient_rejection": 1.0 - _safe_rate(kept_transient, total_transient),
    }


def _run_aqua(
    *,
    video: np.ndarray,
    manifest: dict[str, Any],
    model_config: Path,
    ckpt_path: Path,
    device: torch.device,
    grid_stride: int,
    query_chunk_size: int,
) -> dict[str, Any]:
    query_cpu, coord_txy = _grid_queries(video.shape[0], video.shape[1], video.shape[2], stride=int(grid_stride))
    model = _load_model(model_config, ckpt_path, device=device)
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
    dynamic_probs = _sigmoid(pred["dynamic_object_logit"].numpy()).astype(np.float32)
    particle_probs = _sigmoid(pred["particle_logit"].numpy()).astype(np.float32)
    confidence_probs = _sigmoid(pred["confidence"].numpy()).astype(np.float32)
    if "static_confidence" in pred:
        static_probs = pred["static_confidence"].numpy().astype(np.float32)
    else:
        static_probs = confidence_probs * (1.0 - dynamic_probs) * (1.0 - particle_probs)
    return {
        "coord_txy": coord_txy,
        "dynamic_probs": dynamic_probs,
        "particle_probs": particle_probs,
        "confidence_probs": confidence_probs,
        "static_probs": static_probs.astype(np.float32),
    }


def _make_frame_row(
    *,
    video: np.ndarray,
    clean_video: np.ndarray | None,
    dynamic_mask: np.ndarray,
    particle_mask: np.ndarray,
    transient_gt: np.ndarray,
    aqua: dict[str, Any],
    labels_transient: np.ndarray,
    raw_keep: np.ndarray,
    aqua_keep: np.ndarray,
    frame_idx: int,
    static_threshold: float,
    grid_stride: int,
    detector: Any,
    max_features: int,
    clip_metrics: dict[str, Any],
    note: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = video.shape[1:3]
    static_map = _grid_map(aqua["coord_txy"], aqua["static_probs"], frame_idx, h, w, grid_stride, smooth=True)
    transient_map = _grid_map(
        aqua["coord_txy"],
        np.maximum(aqua["dynamic_probs"], aqua["particle_probs"]),
        frame_idx,
        h,
        w,
        grid_stride,
        smooth=True,
    )
    aqua_static_dense = static_map >= float(static_threshold)
    raw_panel, raw_feature_metrics = _feature_panel(
        video[frame_idx],
        np.ones((h, w), dtype=bool),
        transient_gt,
        frame_idx,
        "Raw ORB",
        detector,
        max_features,
    )
    aqua_panel, aqua_feature_metrics = _feature_panel(
        video[frame_idx],
        aqua_static_dense,
        transient_gt,
        frame_idx,
        "Aqua static ORB",
        detector,
        max_features,
    )
    gt_title = "GT transient"
    if particle_mask[frame_idx].any():
        gt_title = "GT fish(red)+snow(yellow)"
    frame_metrics = {
        "raw_features": raw_feature_metrics,
        "aqua_features": aqua_feature_metrics,
    }
    input_rgb = clean_video[frame_idx] if clean_video is not None else video[frame_idx]
    input_title = "Clean background" if clean_video is not None else "Input"
    columns = [
        _draw_title(cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR), input_title),
        _draw_title(cv2.cvtColor(video[frame_idx], cv2.COLOR_RGB2BGR), f"Corrupted/Input f{frame_idx:02d}"),
        _overlay_gt(video[frame_idx], dynamic_mask[frame_idx], particle_mask[frame_idx], gt_title),
        _query_panel(
            video[frame_idx],
            aqua["coord_txy"],
            labels_transient,
            raw_keep,
            frame_idx,
            "Raw query map",
            draw_rejected=False,
        ),
        _query_panel(
            video[frame_idx],
            aqua["coord_txy"],
            labels_transient,
            aqua_keep,
            frame_idx,
            f"Aqua kept q >= {static_threshold:.2f}",
            draw_rejected=True,
        ),
        _heatmap_overlay(video[frame_idx], static_map, "Static confidence"),
        _heatmap_overlay(video[frame_idx], transient_map, "Transient prob"),
        raw_panel,
        aqua_panel,
        _metrics_box(
            width=w,
            height=h,
            clip_metrics=clip_metrics,
            frame_metrics=frame_metrics,
            note=note,
        ),
    ]
    return np.concatenate(columns, axis=1), frame_metrics


def _write_video(frames_bgr: list[np.ndarray], output_path: Path, fps: float) -> bool:
    if not frames_bgr:
        return False
    height, width = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        return False
    try:
        for frame in frames_bgr:
            writer.write(frame)
    finally:
        writer.release()
    return output_path.exists() and output_path.stat().st_size > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", default="output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--grid-stride", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--static-threshold", type=float, default=0.55)
    parser.add_argument("--detector", default="orb", choices=("orb", "sift"))
    parser.add_argument("--max-features", type=int, default=900)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    device = _resolve_device(str(args.device))
    cfg = load_yaml_config(args.model_config)
    image_hw = tuple(int(v) for v in cfg["model"]["input"].get("image_size", [256, 256]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    video, dynamic_mask, particle_mask, manifest = _load_clip(
        manifest_path,
        image_hw=image_hw,
        max_frames=int(args.max_frames),
    )
    clean_video = _load_clean_video(manifest, image_hw=image_hw, max_frames=int(args.max_frames))
    transient_gt = (dynamic_mask | particle_mask).astype(bool)
    aqua = _run_aqua(
        video=video,
        manifest=manifest,
        model_config=Path(args.model_config),
        ckpt_path=Path(args.ckpt_path),
        device=device,
        grid_stride=int(args.grid_stride),
        query_chunk_size=int(args.query_chunk_size),
    )

    coord_txy = aqua["coord_txy"]
    labels_transient = transient_gt[coord_txy[:, 0], coord_txy[:, 2], coord_txy[:, 1]]
    raw_keep = np.ones_like(labels_transient, dtype=bool)
    aqua_keep = aqua["static_probs"] >= float(args.static_threshold)
    raw_metrics = _query_metrics(raw_keep, labels_transient)
    aqua_metrics = _query_metrics(aqua_keep, labels_transient)
    clip_metrics = {
        "raw_query_contamination": raw_metrics["contamination"],
        "aqua_query_contamination": aqua_metrics["contamination"],
        "aqua_static_retention": aqua_metrics["static_retention"],
        "aqua_kept_rate": aqua_metrics["kept_rate"],
        "aqua_transient_rejection": aqua_metrics["transient_rejection"],
    }

    note = "Query colors:\ngreen=static, red=transient,\ngray=rejected."
    if "WebUOT" in str(manifest.get("dataset", "")) or "webuot" in str(manifest_path).lower():
        note += "\nWebUOT GT is tracked\nbbox mask, not full fish."
    if particle_mask.any():
        note += "\nYellow GT marks particles."

    detector, _ = _make_detector(str(args.detector), int(args.max_features))
    frame_ids = _parse_frame_ids(str(args.frame_ids), video.shape[0])
    rows: list[np.ndarray] = []
    frame_metrics: dict[str, Any] = {}
    for frame_idx in frame_ids:
        row, metrics = _make_frame_row(
            video=video,
            clean_video=clean_video,
            dynamic_mask=dynamic_mask,
            particle_mask=particle_mask,
            transient_gt=transient_gt,
            aqua=aqua,
            labels_transient=labels_transient,
            raw_keep=raw_keep,
            aqua_keep=aqua_keep,
            frame_idx=frame_idx,
            static_threshold=float(args.static_threshold),
            grid_stride=int(args.grid_stride),
            detector=detector,
            max_features=int(args.max_features),
            clip_metrics=clip_metrics,
            note=note,
        )
        rows.append(row)
        frame_metrics[str(frame_idx)] = metrics
        cv2.imwrite(str(output_dir / f"frame_{frame_idx:03d}_mapping_claim.png"), row)

    sheet_path = output_dir / "mapping_claim_sheet.png"
    if rows:
        cv2.imwrite(str(sheet_path), np.concatenate(rows, axis=0))

    video_path: str | None = None
    if bool(args.save_video):
        movie_rows: list[np.ndarray] = []
        for frame_idx in range(video.shape[0]):
            row, _ = _make_frame_row(
                video=video,
                clean_video=clean_video,
                dynamic_mask=dynamic_mask,
                particle_mask=particle_mask,
                transient_gt=transient_gt,
                aqua=aqua,
                labels_transient=labels_transient,
                raw_keep=raw_keep,
                aqua_keep=aqua_keep,
                frame_idx=frame_idx,
                static_threshold=float(args.static_threshold),
                grid_stride=int(args.grid_stride),
                detector=detector,
                max_features=int(args.max_features),
                clip_metrics=clip_metrics,
                note=note,
            )
            scale = 160.0 / float(max(1, video.shape[2]))
            movie_rows.append(cv2.resize(row, (int(round(row.shape[1] * scale)), int(round(row.shape[0] * scale))), interpolation=cv2.INTER_AREA))
        movie_path = output_dir / "mapping_claim_video.mp4"
        if _write_video(movie_rows, movie_path, fps=float(args.fps)):
            video_path = str(movie_path.resolve())

    summary = {
        "manifest": str(manifest_path.resolve()),
        "clip_name": str(manifest.get("name", manifest_path.parent.name)),
        "dataset": str(manifest.get("dataset", "unknown")),
        "image_hw": [int(video.shape[1]), int(video.shape[2])],
        "num_frames": int(video.shape[0]),
        "frame_ids": frame_ids,
        "static_threshold": float(args.static_threshold),
        "grid_stride": int(args.grid_stride),
        "query_metrics": {"raw": raw_metrics, "aqua_static": aqua_metrics},
        "frame_metrics": frame_metrics,
        "outputs": {
            "mapping_claim_sheet": str(sheet_path.resolve()),
            "mapping_claim_video": video_path,
        },
        "notes": [
            "This visualization targets query-map/static-feature cleanliness, not contour segmentation quality.",
            "Raw query map keeps all grid queries as candidates; Aqua keeps static_confidence >= threshold.",
            "Feature contamination is measured by sampling ORB keypoints against the available transient mask.",
        ],
    }
    if "WebUOT" in str(manifest.get("dataset", "")) or "webuot" in str(manifest_path).lower():
        summary["notes"].append("WebUOT masks are tracked-target bounding boxes only; unlabeled fish are not counted as GT transient.")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved sheet: {sheet_path}")
    if video_path:
        print(f"Saved video: {video_path}")
    print(f"Saved summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
