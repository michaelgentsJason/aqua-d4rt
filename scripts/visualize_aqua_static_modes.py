#!/usr/bin/env python3
"""Visualize Aqua-D4RT static filtering operating points for one clip."""

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

from eval_aqua_transient_heads import _grid_queries, _load_clip, _load_model, _load_rgb, _sigmoid  # noqa: E402
from infer_track_3d import _resolve_device  # noqa: E402
from src.core import load_yaml_config, seed_everything  # noqa: E402
from src.eval.tasks import _encode_model_memory, _run_model_for_queries  # noqa: E402


def _load_clean_video(manifest: dict[str, Any], image_hw: tuple[int, int], max_frames: int) -> np.ndarray | None:
    paths = manifest.get("clean_frames")
    if not isinstance(paths, list) or not paths:
        return None
    if max_frames > 0:
        paths = paths[: int(max_frames)]
    return np.stack([_load_rgb(path, image_hw=image_hw) for path in paths], axis=0)


def _parse_frame_ids(value: str, num_frames: int) -> list[int]:
    if value.strip():
        ids = [int(item) for item in value.split(",") if item.strip()]
    else:
        ids = [0, num_frames // 2, num_frames - 1]
    return sorted({int(np.clip(item, 0, max(0, num_frames - 1))) for item in ids})


def _grid_map(
    coord_txy: np.ndarray,
    values: np.ndarray,
    frame_idx: int,
    height: int,
    width: int,
    stride: int,
) -> np.ndarray:
    mask = coord_txy[:, 0] == int(frame_idx)
    out = np.zeros((height, width), dtype=np.float32)
    if not np.any(mask):
        return out
    xy = coord_txy[mask][:, 1:3]
    out[xy[:, 1], xy[:, 0]] = values[mask].astype(np.float32)
    kernel_size = max(3, int(stride))
    return cv2.dilate(out, np.ones((kernel_size, kernel_size), np.uint8))


def _overlay_transient(frame_rgb: np.ndarray, dynamic_mask: np.ndarray, particle_mask: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    tint = base.copy()
    tint[dynamic_mask.astype(bool)] = (40, 80, 255)
    tint[particle_mask.astype(bool)] = (255, 230, 80)
    return cv2.addWeighted(tint, 0.42, base, 0.58, 0.0)


def _static_preview(frame_rgb: np.ndarray, static_map: np.ndarray, threshold: float) -> np.ndarray:
    base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    keep = static_map >= float(threshold)
    reject = ~keep
    preview = base.copy()
    preview[reject] = np.clip(preview[reject].astype(np.float32) * 0.22 + np.array([28, 28, 28]), 0, 255).astype(np.uint8)
    edge = cv2.morphologyEx(reject.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    preview[edge] = (60, 220, 120)
    heat = np.clip(static_map, 0.0, 1.0)
    heat_bgr = cv2.applyColorMap((heat * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    preview = cv2.addWeighted(preview, 0.82, heat_bgr, 0.18, 0.0)
    return preview


def _draw_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    del h
    bar = np.full((28, w, 3), 24, dtype=np.uint8)
    cv2.putText(bar, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([bar, image_bgr], axis=0)


def _metrics_for_frame(
    coord_txy: np.ndarray,
    labels_transient: np.ndarray,
    static_probs: np.ndarray,
    frame_idx: int,
    threshold: float,
) -> dict[str, float]:
    mask = coord_txy[:, 0] == int(frame_idx)
    if not np.any(mask):
        return {"kept_rate": 0.0, "contamination": 0.0, "static_retention": 0.0}
    transient = labels_transient[mask].astype(bool)
    static = ~transient
    keep = static_probs[mask] >= float(threshold)
    kept = int(keep.sum())
    kept_transient = int(np.logical_and(keep, transient).sum())
    kept_static = int(np.logical_and(keep, static).sum())
    return {
        "kept_rate": kept / float(max(1, keep.size)),
        "contamination": kept_transient / float(max(1, kept)),
        "static_retention": kept_static / float(max(1, int(static.sum()))),
    }


def _make_sheet(
    *,
    clean_rgb: np.ndarray | None,
    corrupt_rgb: np.ndarray,
    gt_dynamic: np.ndarray,
    gt_particle: np.ndarray,
    pred_dynamic: np.ndarray,
    pred_particle: np.ndarray,
    static_map: np.ndarray,
    f1_threshold: float,
    clean_threshold: float,
) -> np.ndarray:
    clean_bgr = cv2.cvtColor(clean_rgb if clean_rgb is not None else corrupt_rgb, cv2.COLOR_RGB2BGR)
    corrupt_bgr = cv2.cvtColor(corrupt_rgb, cv2.COLOR_RGB2BGR)
    cols = [
        _draw_title(clean_bgr, "Clean background"),
        _draw_title(corrupt_bgr, "Corrupted input"),
        _draw_title(_overlay_transient(corrupt_rgb, gt_dynamic, gt_particle), "GT transient"),
        _draw_title(_overlay_transient(corrupt_rgb, pred_dynamic, pred_particle), "Pred transient"),
        _draw_title(_static_preview(corrupt_rgb, static_map, f1_threshold), f"Static F1 > {f1_threshold:.2f}"),
        _draw_title(_static_preview(corrupt_rgb, static_map, clean_threshold), f"Static clean > {clean_threshold:.2f}"),
    ]
    return np.concatenate(cols, axis=1)


def _write_mp4(frames_bgr: list[np.ndarray], output_path: Path, fps: float) -> bool:
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
    parser.add_argument("--frame-ids", default="0,16,31")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    device = _resolve_device(args.device)
    cfg = load_yaml_config(args.model_config)
    image_hw = tuple(int(v) for v in cfg["model"]["input"].get("image_size", [256, 256]))
    manifest_path = Path(args.manifest)
    video, dynamic_mask, particle_mask, manifest = _load_clip(manifest_path, image_hw=image_hw, max_frames=int(args.max_frames))
    clean_video = _load_clean_video(manifest, image_hw=image_hw, max_frames=int(args.max_frames))
    query_cpu, coord_txy = _grid_queries(video.shape[0], video.shape[1], video.shape[2], stride=int(args.grid_stride))
    labels_dynamic = dynamic_mask[coord_txy[:, 0], coord_txy[:, 2], coord_txy[:, 1]]
    labels_particle = particle_mask[coord_txy[:, 0], coord_txy[:, 2], coord_txy[:, 1]]
    labels_transient = labels_dynamic | labels_particle

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

    dynamic_probs = _sigmoid(pred["dynamic_object_logit"].numpy())
    particle_probs = _sigmoid(pred["particle_logit"].numpy())
    if "static_confidence" in pred:
        static_probs = pred["static_confidence"].numpy()
    else:
        confidence_probs = _sigmoid(pred["confidence"].numpy())
        static_probs = confidence_probs * (1.0 - dynamic_probs) * (1.0 - particle_probs)

    output_dir = Path(args.output_dir)
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = _parse_frame_ids(str(args.frame_ids), num_frames=video.shape[0])
    sheets: list[np.ndarray] = []
    frame_metrics: dict[str, Any] = {}
    for frame_idx in frame_ids:
        dyn_map = _grid_map(coord_txy, dynamic_probs, frame_idx, video.shape[1], video.shape[2], int(args.grid_stride))
        par_map = _grid_map(coord_txy, particle_probs, frame_idx, video.shape[1], video.shape[2], int(args.grid_stride))
        static_map = _grid_map(coord_txy, static_probs, frame_idx, video.shape[1], video.shape[2], int(args.grid_stride))
        sheet = _make_sheet(
            clean_rgb=None if clean_video is None else clean_video[frame_idx],
            corrupt_rgb=video[frame_idx],
            gt_dynamic=dynamic_mask[frame_idx],
            gt_particle=particle_mask[frame_idx],
            pred_dynamic=dyn_map >= float(args.dynamic_threshold),
            pred_particle=par_map >= float(args.particle_threshold),
            static_map=static_map,
            f1_threshold=float(args.static_f1_threshold),
            clean_threshold=float(args.static_clean_threshold),
        )
        sheets.append(sheet)
        cv2.imwrite(str(visuals_dir / f"frame_{frame_idx:03d}_static_modes.png"), sheet)
        frame_metrics[str(frame_idx)] = {
            "static_f1_mode": _metrics_for_frame(
                coord_txy, labels_transient, static_probs, frame_idx, float(args.static_f1_threshold)
            ),
            "static_clean_mode": _metrics_for_frame(
                coord_txy, labels_transient, static_probs, frame_idx, float(args.static_clean_threshold)
            ),
        }
    contact_path = output_dir / "static_modes_contact_sheet.png"
    if sheets:
        cv2.imwrite(str(contact_path), np.concatenate(sheets, axis=0))

    movie_frames: list[np.ndarray] = []
    for frame_idx in range(video.shape[0]):
        dyn_map = _grid_map(coord_txy, dynamic_probs, frame_idx, video.shape[1], video.shape[2], int(args.grid_stride))
        par_map = _grid_map(coord_txy, particle_probs, frame_idx, video.shape[1], video.shape[2], int(args.grid_stride))
        static_map = _grid_map(coord_txy, static_probs, frame_idx, video.shape[1], video.shape[2], int(args.grid_stride))
        movie_frames.append(
            _make_sheet(
                clean_rgb=None if clean_video is None else clean_video[frame_idx],
                corrupt_rgb=video[frame_idx],
                gt_dynamic=dynamic_mask[frame_idx],
                gt_particle=particle_mask[frame_idx],
                pred_dynamic=dyn_map >= float(args.dynamic_threshold),
                pred_particle=par_map >= float(args.particle_threshold),
                static_map=static_map,
                f1_threshold=float(args.static_f1_threshold),
                clean_threshold=float(args.static_clean_threshold),
            )
        )
    movie_path = output_dir / "static_modes.mp4"
    movie_ok = _write_mp4(movie_frames, movie_path, fps=float(args.fps))
    summary = {
        "manifest": str(manifest_path.resolve()),
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "thresholds": {
            "dynamic_object": float(args.dynamic_threshold),
            "particle": float(args.particle_threshold),
            "static_f1": float(args.static_f1_threshold),
            "static_clean": float(args.static_clean_threshold),
        },
        "contact_sheet": str(contact_path.resolve()),
        "movie": str(movie_path.resolve()) if movie_ok else None,
        "frame_metrics": frame_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved contact sheet: {contact_path}")
    print(f"Saved movie: {movie_path if movie_ok else 'FAILED'}")
    print(f"Saved summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
