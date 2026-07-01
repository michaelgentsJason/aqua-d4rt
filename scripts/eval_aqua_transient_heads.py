#!/usr/bin/env python3
"""Evaluate Aqua-D4RT transient heads on synthetic clip masks."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_track_3d import _resolve_device, _unwrap_state_dict  # noqa: E402
from src.core import load_checkpoint, load_yaml_config, seed_everything  # noqa: E402
from src.eval.tasks import _encode_model_memory, _run_model_for_queries  # noqa: E402
from src.model import build_model  # noqa: E402


def _load_rgb(path: str | Path, image_hw: tuple[int, int]) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read frame: {path}")
    h, w = image_hw
    if bgr.shape[:2] != (h, w):
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resize_mask(mask: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = image_hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def _load_clip(manifest_path: Path, image_hw: tuple[int, int], max_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_paths = [str(path) for path in manifest["frames"]]
    if max_frames > 0:
        frame_paths = frame_paths[: int(max_frames)]
    frames = [_load_rgb(path, image_hw) for path in frame_paths]
    video = np.stack(frames, axis=0)
    masks = np.load(str(manifest["labels_npz"]))
    dynamic = masks["dynamic_object_mask"][: len(frames)]
    particle = masks["particle_mask"][: len(frames)]
    dynamic = np.stack([_resize_mask(dynamic[t], image_hw) for t in range(dynamic.shape[0])], axis=0)
    particle = np.stack([_resize_mask(particle[t], image_hw) for t in range(particle.shape[0])], axis=0)
    return video, dynamic, particle, manifest


def _grid_queries(num_frames: int, height: int, width: int, stride: int) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    step = max(1, int(stride))
    xs = np.arange(0, width, step, dtype=np.int64)
    ys = np.arange(0, height, step, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    per_frame_xy = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)
    coords = []
    for t in range(int(num_frames)):
        tcol = np.full((per_frame_xy.shape[0], 1), t, dtype=np.int64)
        coords.append(np.concatenate([tcol, per_frame_xy], axis=1))
    coord_txy = np.concatenate(coords, axis=0)
    t_idx = coord_txy[:, 0]
    x = coord_txy[:, 1]
    y = coord_txy[:, 2]
    u = x.astype(np.float32) / float(max(width - 1, 1))
    v = y.astype(np.float32) / float(max(height - 1, 1))
    query = {
        "u": torch.from_numpy(u),
        "v": torch.from_numpy(v),
        "t_src": torch.from_numpy(t_idx).long(),
        "t_tgt": torch.from_numpy(t_idx).long(),
        "t_cam": torch.from_numpy(t_idx).long(),
    }
    return query, coord_txy


def _load_model(config_path: Path, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    cfg = load_yaml_config(config_path)
    model = build_model(cfg["model"])
    payload = load_checkpoint(ckpt_path, map_location="cpu")
    state = _unwrap_state_dict(payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def _binary_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    pred = probs >= float(threshold)
    lab = labels.astype(bool)
    tp = int(np.logical_and(pred, lab).sum())
    fp = int(np.logical_and(pred, ~lab).sum())
    fn = int(np.logical_and(~pred, lab).sum())
    tn = int(np.logical_and(~pred, ~lab).sum())
    precision = tp / float(max(1, tp + fp))
    recall = tp / float(max(1, tp + fn))
    f1 = 2.0 * precision * recall / float(max(1e-12, precision + recall))
    accuracy = (tp + tn) / float(max(1, tp + fp + fn + tn))
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "positive_rate": float(lab.mean()),
        "pred_positive_rate": float(pred.mean()),
        "prob_mean": float(np.mean(probs)),
    }


def _best_f1(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    best = _binary_metrics(probs, labels, threshold=0.5)
    for threshold in np.linspace(0.01, 0.99, num=99):
        metrics = _binary_metrics(probs, labels, float(threshold))
        if metrics["f1"] > best["f1"]:
            best = metrics
    return best


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _write_frame_visuals(
    output_dir: Path,
    video: np.ndarray,
    coord_txy: np.ndarray,
    dynamic_probs: np.ndarray,
    particle_probs: np.ndarray,
    frame_ids: list[int],
) -> None:
    vis_dir = output_dir / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)
    height, width = video.shape[1:3]
    for frame_idx in frame_ids:
        mask = coord_txy[:, 0] == int(frame_idx)
        if not np.any(mask):
            continue
        xy = coord_txy[mask][:, 1:3]
        dyn = dynamic_probs[mask]
        particle = particle_probs[mask]
        dyn_map = np.zeros((height, width), dtype=np.float32)
        particle_map = np.zeros((height, width), dtype=np.float32)
        dyn_map[xy[:, 1], xy[:, 0]] = dyn
        particle_map[xy[:, 1], xy[:, 0]] = particle
        dyn_map = cv2.dilate(dyn_map, np.ones((5, 5), np.uint8))
        particle_map = cv2.dilate(particle_map, np.ones((5, 5), np.uint8))
        base = cv2.cvtColor(video[frame_idx], cv2.COLOR_RGB2BGR)
        overlay = base.copy()
        overlay[dyn_map > 0.5] = (40, 80, 255)
        overlay[particle_map > 0.5] = (255, 230, 80)
        overlay = cv2.addWeighted(overlay, 0.45, base, 0.55, 0.0)
        cv2.imwrite(str(vis_dir / f"frame_{frame_idx:03d}_pred_overlay.png"), overlay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/aqua_synth/watermask_caves_32/manifest.json")
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--grid-stride", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-visuals", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    device = _resolve_device(args.device)
    cfg = load_yaml_config(args.model_config)
    image_hw = tuple(int(v) for v in cfg["model"]["input"].get("image_size", [256, 256]))
    video, dynamic_mask, particle_mask, manifest = _load_clip(Path(args.manifest), image_hw=image_hw, max_frames=int(args.max_frames))
    query_cpu, coord_txy = _grid_queries(video.shape[0], video.shape[1], video.shape[2], stride=int(args.grid_stride))
    labels_dynamic = dynamic_mask[coord_txy[:, 0], coord_txy[:, 2], coord_txy[:, 1]]
    labels_particle = particle_mask[coord_txy[:, 0], coord_txy[:, 2], coord_txy[:, 1]]
    labels_transient = labels_dynamic | labels_particle

    model = _load_model(Path(args.model_config), Path(args.ckpt_path), device=device)
    video_b = torch.from_numpy(video).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
    aspect = torch.tensor([[float(manifest.get("width", video.shape[2])) / float(max(1, manifest.get("height", video.shape[1])))]], device=device)
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
    confidence_probs = _sigmoid(pred["confidence"].numpy())
    static_conf = pred.get("static_confidence")
    if static_conf is None:
        static_probs = confidence_probs * (1.0 - dynamic_probs) * (1.0 - particle_probs)
    else:
        static_probs = static_conf.numpy()

    metrics = {
        "inputs": {
            "manifest": str(Path(args.manifest).resolve()),
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
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if bool(args.save_visuals):
        _write_frame_visuals(
            output_dir=output_dir,
            video=video,
            coord_txy=coord_txy,
            dynamic_probs=dynamic_probs,
            particle_probs=particle_probs,
            frame_ids=[0, video.shape[0] // 4, video.shape[0] // 2, (3 * video.shape[0]) // 4, video.shape[0] - 1],
        )
    for name in ("dynamic_object", "particle", "static"):
        fixed = metrics[name]["threshold_0_5"]
        best = metrics[name]["best_f1"]
        print(
            f"{name}: f1@0.5={fixed['f1']:.4f} p={fixed['precision']:.4f} r={fixed['recall']:.4f} "
            f"best_f1={best['f1']:.4f}@{best['threshold']:.2f}"
        )
    print(f"Saved metrics: {output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
