#!/usr/bin/env python3
"""Run one Aqua-D4RT synthetic case and write side-by-side visualizations."""

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


def _parse_frame_ids(value: str, num_frames: int) -> list[int]:
    if value.strip():
        ids = [int(item) for item in value.split(",") if item.strip()]
    else:
        ids = [0, num_frames // 4, num_frames // 2, (3 * num_frames) // 4, num_frames - 1]
    return sorted({int(np.clip(item, 0, max(0, num_frames - 1))) for item in ids})


def _load_clean_video(manifest: dict[str, Any], image_hw: tuple[int, int], max_frames: int) -> np.ndarray | None:
    paths = manifest.get("clean_frames")
    if not isinstance(paths, list) or not paths:
        return None
    if max_frames > 0:
        paths = paths[: int(max_frames)]
    frames = [_load_rgb(path, image_hw=image_hw) for path in paths]
    return np.stack(frames, axis=0)


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
    kernel = np.ones((max(3, int(stride)), max(3, int(stride))), dtype=np.uint8)
    return cv2.dilate(out, kernel)


def _overlay_transient(frame_rgb: np.ndarray, dynamic_mask: np.ndarray, particle_mask: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    tint = base.copy()
    tint[dynamic_mask.astype(bool)] = (40, 80, 255)
    tint[particle_mask.astype(bool)] = (255, 230, 80)
    return cv2.addWeighted(tint, float(alpha), base, 1.0 - float(alpha), 0.0)


def _static_keep_preview(frame_rgb: np.ndarray, reject_mask: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    preview = base.copy()
    reject = reject_mask.astype(bool)
    preview[reject] = np.clip(preview[reject].astype(np.float32) * 0.20 + np.array([35, 35, 35]), 0, 255).astype(np.uint8)
    edge = cv2.morphologyEx(reject.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    preview[edge] = (60, 220, 120)
    return preview


def _draw_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    bar = np.full((28, w, 3), 24, dtype=np.uint8)
    cv2.putText(bar, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([bar, image_bgr], axis=0)


def _make_case_sheet(
    *,
    clean_rgb: np.ndarray | None,
    corrupt_rgb: np.ndarray,
    gt_dynamic: np.ndarray,
    gt_particle: np.ndarray,
    pred_dynamic_map: np.ndarray,
    pred_particle_map: np.ndarray,
) -> np.ndarray:
    gt_overlay = _overlay_transient(corrupt_rgb, gt_dynamic, gt_particle)
    pred_dynamic = pred_dynamic_map >= 0.5
    pred_particle = pred_particle_map >= 0.5
    pred_overlay = _overlay_transient(corrupt_rgb, pred_dynamic, pred_particle)
    static_preview = _static_keep_preview(corrupt_rgb, pred_dynamic | pred_particle)
    clean_bgr = cv2.cvtColor(clean_rgb if clean_rgb is not None else corrupt_rgb, cv2.COLOR_RGB2BGR)
    corrupt_bgr = cv2.cvtColor(corrupt_rgb, cv2.COLOR_RGB2BGR)
    columns = [
        _draw_title(clean_bgr, "Clean background"),
        _draw_title(corrupt_bgr, "Corrupted input"),
        _draw_title(gt_overlay, "GT transient"),
        _draw_title(pred_overlay, "Aqua-D4RT pred"),
        _draw_title(static_preview, "Static kept preview"),
    ]
    return np.concatenate(columns, axis=1)


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
    parser.add_argument("--manifest", default="data/aqua_synth/watermask_caves_32/manifest.json")
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", default="output/exp_aqua_d4rt/aqua_synth_phase_a_v1/checkpoints/best.ckpt")
    parser.add_argument("--output-dir", default="tmp/aqua_case_v1/watermask_caves_32")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--grid-stride", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--frame-ids", default="")
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
    video, dynamic_mask, particle_mask, manifest = _load_clip(
        manifest_path,
        image_hw=image_hw,
        max_frames=int(args.max_frames),
    )
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

    metrics = {
        "inputs": {
            "manifest": str(manifest_path.resolve()),
            "model_config": str(Path(args.model_config).resolve()),
            "ckpt_path": str(Path(args.ckpt_path).resolve()),
            "grid_stride": int(args.grid_stride),
            "num_queries": int(coord_txy.shape[0]),
            "num_frames": int(video.shape[0]),
            "image_hw": [int(video.shape[1]), int(video.shape[2])],
        },
        "dynamic_object": {
            "threshold_0_5": _binary_metrics(dynamic_probs, labels_dynamic, threshold=0.5),
            "best_f1": _best_f1(dynamic_probs, labels_dynamic),
        },
        "particle": {
            "threshold_0_5": _binary_metrics(particle_probs, labels_particle, threshold=0.5),
            "best_f1": _best_f1(particle_probs, labels_particle),
        },
        "static": {
            "threshold_0_5": _binary_metrics(static_probs, ~labels_transient, threshold=0.5),
            "best_f1": _best_f1(static_probs, ~labels_transient),
        },
    }

    output_dir = Path(args.output_dir)
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_dir / "predictions_grid.npz",
        coord_txy=coord_txy,
        dynamic_probs=dynamic_probs,
        particle_probs=particle_probs,
        static_probs=static_probs,
        labels_dynamic=labels_dynamic.astype(np.uint8),
        labels_particle=labels_particle.astype(np.uint8),
        labels_static=(~labels_transient).astype(np.uint8),
    )

    frame_ids = _parse_frame_ids(str(args.frame_ids), num_frames=int(video.shape[0]))
    sheets: list[np.ndarray] = []
    for frame_idx in frame_ids:
        pred_dynamic_map = _grid_map(
            coord_txy,
            dynamic_probs,
            frame_idx=frame_idx,
            height=video.shape[1],
            width=video.shape[2],
            stride=int(args.grid_stride),
        )
        pred_particle_map = _grid_map(
            coord_txy,
            particle_probs,
            frame_idx=frame_idx,
            height=video.shape[1],
            width=video.shape[2],
            stride=int(args.grid_stride),
        )
        sheet = _make_case_sheet(
            clean_rgb=None if clean_video is None else clean_video[frame_idx],
            corrupt_rgb=video[frame_idx],
            gt_dynamic=dynamic_mask[frame_idx],
            gt_particle=particle_mask[frame_idx],
            pred_dynamic_map=pred_dynamic_map,
            pred_particle_map=pred_particle_map,
        )
        sheets.append(sheet)
        cv2.imwrite(str(visuals_dir / f"frame_{frame_idx:03d}_case_sheet.png"), sheet)
    if sheets:
        contact = np.concatenate(sheets, axis=0)
        cv2.imwrite(str(output_dir / "case_contact_sheet.png"), contact)

    movie_frames: list[np.ndarray] = []
    for frame_idx in range(int(video.shape[0])):
        pred_dynamic_map = _grid_map(
            coord_txy,
            dynamic_probs,
            frame_idx=frame_idx,
            height=video.shape[1],
            width=video.shape[2],
            stride=int(args.grid_stride),
        )
        pred_particle_map = _grid_map(
            coord_txy,
            particle_probs,
            frame_idx=frame_idx,
            height=video.shape[1],
            width=video.shape[2],
            stride=int(args.grid_stride),
        )
        movie_frames.append(
            _make_case_sheet(
                clean_rgb=None if clean_video is None else clean_video[frame_idx],
                corrupt_rgb=video[frame_idx],
                gt_dynamic=dynamic_mask[frame_idx],
                gt_particle=particle_mask[frame_idx],
                pred_dynamic_map=pred_dynamic_map,
                pred_particle_map=pred_particle_map,
            )
        )
    movie_path = output_dir / "case_visualization.mp4"
    movie_ok = _write_mp4(movie_frames, movie_path, fps=float(args.fps))
    summary = {
        "metrics_json": str((output_dir / "metrics.json").resolve()),
        "predictions_npz": str((output_dir / "predictions_grid.npz").resolve()),
        "contact_sheet": str((output_dir / "case_contact_sheet.png").resolve()),
        "case_visualization_mp4": str(movie_path.resolve()) if movie_ok else None,
        "frame_sheets": [str((visuals_dir / f"frame_{frame_idx:03d}_case_sheet.png").resolve()) for frame_idx in frame_ids],
        "metrics_brief": {
            name: {
                "f1_at_0_5": metrics[name]["threshold_0_5"]["f1"],
                "precision_at_0_5": metrics[name]["threshold_0_5"]["precision"],
                "recall_at_0_5": metrics[name]["threshold_0_5"]["recall"],
                "best_f1": metrics[name]["best_f1"]["f1"],
            }
            for name in ("dynamic_object", "particle", "static")
        },
    }
    (output_dir / "case_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, values in summary["metrics_brief"].items():
        print(
            f"{name}: f1@0.5={values['f1_at_0_5']:.4f} "
            f"p={values['precision_at_0_5']:.4f} r={values['recall_at_0_5']:.4f} "
            f"best_f1={values['best_f1']:.4f}"
        )
    print(f"Saved case summary: {output_dir / 'case_summary.json'}")
    print(f"Saved contact sheet: {output_dir / 'case_contact_sheet.png'}")
    print(f"Saved case video: {movie_path if movie_ok else 'FAILED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
