#!/usr/bin/env python3
"""Visualize Aqua-D4RT predictions on a real underwater clip without GT masks."""

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

from infer_track_3d import _resolve_device  # noqa: E402
from run_d4rt_smoke import _load_manifest_or_frames  # noqa: E402
from src.core import load_yaml_config, seed_everything  # noqa: E402
from src.eval.tasks import _encode_model_memory, _run_model_for_queries  # noqa: E402
from eval_aqua_transient_heads import _grid_queries, _load_model, _sigmoid  # noqa: E402


def _parse_frame_ids(value: str, num_frames: int) -> list[int]:
    if value.strip():
        ids = [int(item) for item in value.split(",") if item.strip()]
    else:
        ids = [0, num_frames // 4, num_frames // 2, (3 * num_frames) // 4, num_frames - 1]
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
    kernel = np.ones((max(3, int(stride)), max(3, int(stride))), dtype=np.uint8)
    return cv2.dilate(out, kernel)


def _draw_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    del h
    bar = np.full((28, w, 3), 24, dtype=np.uint8)
    cv2.putText(bar, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([bar, image_bgr], axis=0)


def _heatmap_overlay(frame_rgb: np.ndarray, score_map: np.ndarray, title: str) -> np.ndarray:
    base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    heat = np.clip(score_map, 0.0, 1.0)
    heat_bgr = cv2.applyColorMap((heat * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    blend = cv2.addWeighted(base, 0.72, heat_bgr, 0.28, 0.0)
    return _draw_title(blend, title)


def _static_preview(frame_rgb: np.ndarray, static_map: np.ndarray, threshold: float) -> np.ndarray:
    base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    keep = static_map >= float(threshold)
    reject = ~keep
    preview = base.copy()
    preview[reject] = np.clip(preview[reject].astype(np.float32) * 0.20 + np.array([35, 35, 35]), 0, 255).astype(np.uint8)
    edge = cv2.morphologyEx(reject.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    preview[edge] = (60, 220, 120)
    heat = np.clip(static_map, 0.0, 1.0)
    heat_bgr = cv2.applyColorMap((heat * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    preview = cv2.addWeighted(preview, 0.82, heat_bgr, 0.18, 0.0)
    return _draw_title(preview, f"Static keep > {threshold:.2f}")


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


def _run_model(
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
    return {key: value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value) for key, value in pred.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Manifest.json or directory with frames.")
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--grid-stride", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--static-clean-threshold", type=float, default=0.55)
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    device = _resolve_device(args.device)
    cfg = load_yaml_config(Path(args.model_config))
    image_hw = tuple(int(v) for v in cfg["model"]["input"].get("image_size", [256, 256]))
    video, manifest = _load_manifest_or_frames(Path(args.input), max_frames=int(args.max_frames))
    if video.shape[0] == 0:
        raise RuntimeError("No frames loaded.")
    if video.shape[1:3] != image_hw:
        resized = [cv2.resize(frame, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_AREA) for frame in video]
        video = np.stack(resized, axis=0)

    query_cpu, coord_txy = _grid_queries(video.shape[0], video.shape[1], video.shape[2], stride=int(args.grid_stride))
    model = _load_model(Path(args.model_config), Path(args.ckpt_path), device=device)
    pred = _run_model(
        model=model,
        video=video,
        manifest=manifest,
        query_cpu=query_cpu,
        device=device,
        query_chunk_size=int(args.query_chunk_size),
    )

    dynamic_probs = _sigmoid(pred["dynamic_object_logit"]).astype(np.float32)
    particle_probs = _sigmoid(pred["particle_logit"]).astype(np.float32)
    confidence_probs = _sigmoid(pred["confidence"]).astype(np.float32)
    if "static_confidence" in pred:
        static_probs = pred["static_confidence"].astype(np.float32)
    else:
        static_probs = (confidence_probs * (1.0 - dynamic_probs) * (1.0 - particle_probs)).astype(np.float32)

    output_dir = Path(args.output_dir)
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "predictions_grid.npz",
        coord_txy=coord_txy,
        dynamic_probs=dynamic_probs,
        particle_probs=particle_probs,
        confidence_probs=confidence_probs,
        static_probs=static_probs,
    )

    summary = {
        "input": str(Path(args.input).resolve()),
        "model_config": str(Path(args.model_config).resolve()),
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "device": str(device),
        "num_frames": int(video.shape[0]),
        "image_hw": [int(video.shape[1]), int(video.shape[2])],
        "num_queries": int(coord_txy.shape[0]),
        "grid_stride": int(args.grid_stride),
        "score_stats": {
            "dynamic_prob_mean": float(np.mean(dynamic_probs)),
            "particle_prob_mean": float(np.mean(particle_probs)),
            "confidence_prob_mean": float(np.mean(confidence_probs)),
            "static_confidence_mean": float(np.mean(static_probs)),
        },
        "predicted_positive_rates": {
            "dynamic_0_79": float((dynamic_probs >= 0.79).mean()),
            "particle_0_83": float((particle_probs >= 0.83).mean()),
            "static_0_55": float((static_probs >= float(args.static_clean_threshold)).mean()),
        },
        "outputs": {
            "predictions_npz": str((output_dir / "predictions_grid.npz").resolve()),
        },
    }

    frame_ids = _parse_frame_ids(str(args.frame_ids), num_frames=int(video.shape[0]))
    contact_rows: list[np.ndarray] = []
    for frame_idx in frame_ids:
        dyn_map = _grid_map(coord_txy, dynamic_probs, frame_idx, video.shape[1], video.shape[2], int(args.grid_stride))
        par_map = _grid_map(coord_txy, particle_probs, frame_idx, video.shape[1], video.shape[2], int(args.grid_stride))
        stat_map = _grid_map(coord_txy, static_probs, frame_idx, video.shape[1], video.shape[2], int(args.grid_stride))
        columns = [
            _draw_title(cv2.cvtColor(video[frame_idx], cv2.COLOR_RGB2BGR), "Input"),
            _heatmap_overlay(video[frame_idx], dyn_map, "Dynamic prob"),
            _heatmap_overlay(video[frame_idx], par_map, "Particle prob"),
            _heatmap_overlay(video[frame_idx], stat_map, "Static confidence"),
            _static_preview(video[frame_idx], stat_map, float(args.static_clean_threshold)),
        ]
        sheet = np.concatenate(columns, axis=1)
        contact_rows.append(sheet)
        cv2.imwrite(str(visuals_dir / f"frame_{frame_idx:03d}_real_case.png"), sheet)

    contact_sheet = np.concatenate(contact_rows, axis=0) if contact_rows else None
    contact_path = output_dir / "real_case_contact_sheet.png"
    if contact_sheet is not None:
        cv2.imwrite(str(contact_path), contact_sheet)
    movie_frames: list[np.ndarray] = []
    for frame_idx in range(video.shape[0]):
        dyn_map = _grid_map(coord_txy, dynamic_probs, frame_idx, video.shape[1], video.shape[2], int(args.grid_stride))
        par_map = _grid_map(coord_txy, particle_probs, frame_idx, video.shape[1], video.shape[2], int(args.grid_stride))
        stat_map = _grid_map(coord_txy, static_probs, frame_idx, video.shape[1], video.shape[2], int(args.grid_stride))
        movie_frames.append(
            np.concatenate(
                [
                    _draw_title(cv2.cvtColor(video[frame_idx], cv2.COLOR_RGB2BGR), "Input"),
                    _heatmap_overlay(video[frame_idx], dyn_map, "Dynamic prob"),
                    _heatmap_overlay(video[frame_idx], par_map, "Particle prob"),
                    _heatmap_overlay(video[frame_idx], stat_map, "Static confidence"),
                    _static_preview(video[frame_idx], stat_map, float(args.static_clean_threshold)),
                ],
                axis=1,
            )
        )
    movie_path = output_dir / "real_case_visualization.mp4"
    movie_ok = _write_mp4(movie_frames, movie_path, fps=float(args.fps))
    summary["outputs"].update(
        {
            "contact_sheet": str(contact_path.resolve()) if contact_path.exists() else None,
            "movie": str(movie_path.resolve()) if movie_ok else None,
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {output_dir / 'summary.json'}")
    print(f"Saved contact sheet: {contact_path}")
    print(f"Saved movie: {movie_path if movie_ok else 'FAILED'}")
    print(
        "Predicted positive rates: "
        f"dynamic@0.79={summary['predicted_positive_rates']['dynamic_0_79']:.4f} "
        f"particle@0.83={summary['predicted_positive_rates']['particle_0_83']:.4f} "
        f"static@0.55={summary['predicted_positive_rates']['static_0_55']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
