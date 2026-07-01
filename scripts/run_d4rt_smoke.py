#!/usr/bin/env python3
"""Run a small OpenD4RT inference smoke test on a frame folder."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_track_3d import _grid_query_points, _infer_tracks, _resolve_device, _resize_video, _unwrap_state_dict
from src.core import load_checkpoint, load_yaml_config, seed_everything
from src.model import build_model


def _frame_sort_key(path: Path) -> tuple[str, int, str]:
    match = re.search(r"(\d+)(?=\D*$)", path.stem)
    frame_id = int(match.group(1)) if match else -1
    prefix = path.stem[: match.start(1)] if match else path.stem
    return prefix, frame_id, path.name


def _load_manifest_or_frames(path: Path, max_frames: int) -> tuple[np.ndarray, dict[str, Any]]:
    path = path.expanduser()
    if path.is_file() and path.name == "manifest.json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        frame_paths = [Path(item) for item in manifest["frames"]]
        root = path.parent
    else:
        root = path
        frames_dir = path / "frames" if (path / "frames").exists() else path
        exts = {".png", ".jpg", ".jpeg"}
        frame_paths = [item for item in frames_dir.iterdir() if item.is_file() and item.suffix.lower() in exts]
        frame_paths.sort(key=_frame_sort_key)
        manifest = {"name": root.name, "frames_dir": str(frames_dir.resolve())}

    if max_frames > 0:
        frame_paths = frame_paths[: int(max_frames)]
    if not frame_paths:
        raise RuntimeError(f"No frames found under {path}")

    images: list[np.ndarray] = []
    for frame_path in frame_paths:
        image_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read frame: {frame_path}")
        images.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    video_rgb = np.stack(images, axis=0)
    manifest = dict(manifest)
    manifest["resolved_root"] = str(root.resolve())
    manifest["used_frames"] = [str(item.resolve()) for item in frame_paths]
    return video_rgb, manifest


def _load_model(config_path: Path, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    cfg = load_yaml_config(config_path)
    model = build_model(cfg["model"])
    payload = load_checkpoint(ckpt_path, map_location="cpu")
    state = _unwrap_state_dict(payload)
    if not state:
        raise RuntimeError(f"No model state_dict found in checkpoint: {ckpt_path}")
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing:
        print(f"Missing checkpoint keys: {len(missing)}")
        for key in missing[:12]:
            print(f"  missing: {key}")
        if len(missing) > 12:
            print("  ...")
    if unexpected:
        print(f"Unexpected checkpoint keys: {len(unexpected)}")
        for key in unexpected[:12]:
            print(f"  unexpected: {key}")
        if len(unexpected) > 12:
            print("  ...")
    model.to(device)
    model.eval()
    return model


def _save_track_overlay(video_rgb: np.ndarray, tracks_uv_norm: np.ndarray, output_path: Path, fps: float = 10.0) -> bool:
    if video_rgb.ndim != 4 or tracks_uv_norm.ndim != 3:
        return False
    num_frames, height, width = video_rgb.shape[:3]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        return False
    try:
        colors = [
            (255, 80, 80),
            (80, 220, 120),
            (80, 160, 255),
            (255, 210, 80),
            (220, 80, 255),
            (80, 240, 240),
        ]
        for t in range(num_frames):
            frame = cv2.cvtColor(video_rgb[t], cv2.COLOR_RGB2BGR)
            for q_idx, uv in enumerate(tracks_uv_norm[:, t]):
                if not np.isfinite(uv).all():
                    continue
                x = int(np.clip(round(float(uv[0]) * (width - 1)), 0, width - 1))
                y = int(np.clip(round(float(uv[1]) * (height - 1)), 0, height - 1))
                color = colors[q_idx % len(colors)]
                cv2.circle(frame, (x, y), 3, color, thickness=-1)
            writer.write(frame)
    finally:
        writer.release()
    return output_path.exists() and output_path.stat().st_size > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/aqua_smoke/underwater_caves_sonar_32")
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt")
    parser.add_argument("--output-dir", default="tmp/aqua_smoke/d4rt_32clip")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--query-cols", type=int, default=4)
    parser.add_argument("--query-rows", type=int, default=4)
    parser.add_argument("--max-queries", type=int, default=16)
    parser.add_argument("--query-chunk-size", type=int, default=16)
    parser.add_argument("--margin-ratio", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-overlay", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    device = _resolve_device(args.device)

    config_path = Path(args.model_config)
    ckpt_path = Path(args.ckpt_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    video_raw_rgb, manifest = _load_manifest_or_frames(Path(args.input), max_frames=int(args.max_frames))
    cfg = load_yaml_config(config_path)
    image_hw = tuple(int(v) for v in cfg["model"]["input"].get("image_size", [256, 256]))
    video_model_rgb = _resize_video(video_raw_rgb, image_hw=image_hw)

    query_px = _grid_query_points(
        width=video_model_rgb.shape[2],
        height=video_model_rgb.shape[1],
        cols=int(args.query_cols),
        rows=int(args.query_rows),
        margin_ratio=float(args.margin_ratio),
        max_points=int(args.max_queries),
    )
    denom = np.asarray([max(video_model_rgb.shape[2] - 1, 1), max(video_model_rgb.shape[1] - 1, 1)], dtype=np.float32)
    query_uv_norm = np.clip(query_px / denom[None, :], 0.0, 1.0).astype(np.float32)

    print(f"Device: {device}")
    print(f"Frames: raw={video_raw_rgb.shape} model={video_model_rgb.shape}")
    print(f"Queries: {query_uv_norm.shape[0]} chunk={int(args.query_chunk_size)}")
    model = _load_model(config_path=config_path, ckpt_path=ckpt_path, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tracks = _infer_tracks(
        model=model,
        video_model_rgb=video_model_rgb,
        query_uv_norm=query_uv_norm,
        query_chunk_size=int(args.query_chunk_size),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        print(f"CUDA peak memory MB: {torch.cuda.max_memory_allocated(device) / (1024 ** 2):.1f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "tracks.npz"
    np.savez_compressed(
        npz_path,
        query_uv_norm=query_uv_norm,
        video_shape=np.asarray(video_model_rgb.shape, dtype=np.int32),
        **{key: value for key, value in tracks.items() if isinstance(value, np.ndarray)},
    )

    summary = {
        "input": str(Path(args.input).resolve()),
        "model_config": str(config_path.resolve()),
        "ckpt_path": str(ckpt_path.resolve()),
        "device": str(device),
        "num_frames": int(video_model_rgb.shape[0]),
        "image_hw": [int(video_model_rgb.shape[1]), int(video_model_rgb.shape[2])],
        "num_queries": int(query_uv_norm.shape[0]),
        "tracks_npz": str(npz_path.resolve()),
        "manifest": manifest,
        "finite_xyz_ratio": float(np.isfinite(tracks["tracks_xyz_ref0"]).all(axis=-1).mean()),
        "visible_ratio": float(np.asarray(tracks["tracks_visibility"], dtype=bool).mean()),
        "mean_confidence_sigmoid": float(np.nanmean(1.0 / (1.0 + np.exp(-tracks["tracks_confidence"])))),
        "mean_static_confidence": float(np.nanmean(tracks["tracks_static_confidence"])),
    }

    if bool(args.save_overlay):
        overlay_path = output_dir / "track_overlay.mp4"
        overlay_ok = _save_track_overlay(video_model_rgb, tracks["tracks_uv_norm"], overlay_path)
        summary["track_overlay_mp4"] = str(overlay_path.resolve()) if overlay_ok else None

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {npz_path}")
    print(f"Summary: {summary_path}")
    print(f"finite_xyz_ratio={summary['finite_xyz_ratio']:.4f}")
    print(f"visible_ratio={summary['visible_ratio']:.4f}")
    print(f"mean_static_confidence={summary['mean_static_confidence']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
