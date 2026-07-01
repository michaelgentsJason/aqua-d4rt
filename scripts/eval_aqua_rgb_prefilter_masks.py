#!/usr/bin/env python3
"""Evaluate non-oracle RGB prefilter masks against available transient labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aqua_prefilter_utils import binary_mask_metrics, temporal_rgb_pseudo_mask  # noqa: E402
from eval_aqua_transient_heads import _load_rgb  # noqa: E402


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
    frames = [_load_rgb(path, image_hw=image_hw) for path in frame_paths]
    video = np.stack(frames, axis=0)
    masks = np.load(str(manifest["labels_npz"]))
    dynamic = masks["dynamic_object_mask"][: len(frames)]
    particle = masks["particle_mask"][: len(frames)]
    dynamic = np.stack([_resize_mask(dynamic[t], image_hw) for t in range(dynamic.shape[0])], axis=0)
    particle = np.stack([_resize_mask(particle[t], image_hw) for t in range(particle.shape[0])], axis=0)
    return video, dynamic, particle, manifest


def _load_or_make_pseudo_mask(
    *,
    video: np.ndarray,
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    if not bool(args.recompute):
        pseudo_path = manifest.get("pseudo_mask_npz")
        if isinstance(pseudo_path, str) and pseudo_path:
            path = Path(pseudo_path)
            if path.exists():
                data = np.load(str(path))
                for key in ("pseudo_dynamic_mask", "dynamic_object_mask", "transient_mask"):
                    if key in data:
                        return data[key].astype(bool)[: video.shape[0]], f"loaded:{key}"
    return (
        temporal_rgb_pseudo_mask(
            video,
            percentile=float(args.temporal_percentile),
            min_threshold=float(args.temporal_min_threshold),
            blur_kernel=int(args.temporal_blur_kernel),
            morph_kernel=int(args.temporal_morph_kernel),
            dilate_iterations=int(args.temporal_dilate_iterations),
            min_component_area=int(args.temporal_min_component_area),
            max_mask_fraction=float(args.temporal_max_mask_fraction),
        ),
        "temporal_rgb_median_residual",
    )


def _mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return intersection / float(max(1, union))


def _write_visual(
    output_dir: Path,
    name: str,
    video: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> str | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if video.shape[0] == 0:
        return None
    picks = np.linspace(0, video.shape[0] - 1, num=min(6, video.shape[0])).round().astype(int)
    rows: list[np.ndarray] = []
    for idx in picks:
        frame = cv2.cvtColor(video[int(idx)], cv2.COLOR_RGB2BGR)
        gt = frame.copy()
        gt_overlay = gt.copy()
        gt_overlay[gt_mask[int(idx)].astype(bool)] = (40, 80, 255)
        gt = cv2.addWeighted(gt_overlay, 0.35, gt, 0.65, 0.0)
        pred = frame.copy()
        pred_overlay = pred.copy()
        pred_overlay[pred_mask[int(idx)].astype(bool)] = (40, 220, 120)
        pred = cv2.addWeighted(pred_overlay, 0.35, pred, 0.65, 0.0)
        bar = np.full((24, frame.shape[1] * 3, 3), 24, dtype=np.uint8)
        cv2.putText(bar, f"frame {idx:03d} | input / GT transient / RGB pseudo", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
        rows.append(np.concatenate([bar, np.concatenate([frame, gt, pred], axis=1)], axis=0))
    sheet = np.concatenate(rows, axis=0)
    path = output_dir / f"{name}_rgb_prefilter_mask_eval.png"
    cv2.imwrite(str(path), sheet)
    return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=None)
    parser.add_argument("--manifest-list", action="append", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--recompute", action="store_true", help="Ignore cached pseudo masks in manifests.")
    parser.add_argument("--temporal-percentile", type=float, default=92.0)
    parser.add_argument("--temporal-min-threshold", type=float, default=18.0)
    parser.add_argument("--temporal-blur-kernel", type=int, default=5)
    parser.add_argument("--temporal-morph-kernel", type=int, default=5)
    parser.add_argument("--temporal-dilate-iterations", type=int, default=1)
    parser.add_argument("--temporal-min-component-area", type=int, default=20)
    parser.add_argument("--temporal-max-mask-fraction", type=float, default=0.45)
    parser.add_argument("--save-visuals", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifests = _resolve_manifests(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_hw = (int(args.image_height), int(args.image_width))

    per_clip: list[dict[str, Any]] = []
    all_pred: list[np.ndarray] = []
    all_dyn: list[np.ndarray] = []
    all_particle: list[np.ndarray] = []
    all_transient: list[np.ndarray] = []
    for idx, manifest_path in enumerate(manifests):
        print(f"[{idx + 1}/{len(manifests)}] {manifest_path}", flush=True)
        video, dynamic, particle, manifest = _load_clip(manifest_path, image_hw=image_hw, max_frames=int(args.max_frames))
        pred, source = _load_or_make_pseudo_mask(video=video, manifest=manifest, args=args)
        pred = np.stack([_resize_mask(pred[t], image_hw) for t in range(pred.shape[0])], axis=0)
        transient = dynamic | particle
        name = manifest_path.parent.name or manifest_path.stem
        visual_path = _write_visual(output_dir / "visuals", name, video, transient, pred) if bool(args.save_visuals) else None
        item = {
            "manifest": str(manifest_path.resolve()),
            "clip": name,
            "dataset": manifest.get("dataset"),
            "language": manifest.get("language"),
            "num_frames": int(video.shape[0]),
            "image_hw": [int(video.shape[1]), int(video.shape[2])],
            "pseudo_source": source,
            "coverage": {
                "dynamic_object": float(dynamic.mean()),
                "particle": float(particle.mean()),
                "transient": float(transient.mean()),
                "pseudo": float(pred.mean()),
            },
            "metrics_vs_dynamic": binary_mask_metrics(pred, dynamic),
            "metrics_vs_particle": binary_mask_metrics(pred, particle),
            "metrics_vs_transient": binary_mask_metrics(pred, transient),
            "iou_vs_transient": float(_mask_iou(pred, transient)),
            "visual": visual_path,
        }
        per_clip.append(item)
        all_pred.append(pred.reshape(-1))
        all_dyn.append(dynamic.reshape(-1))
        all_particle.append(particle.reshape(-1))
        all_transient.append(transient.reshape(-1))

    pred_all = np.concatenate(all_pred, axis=0).astype(bool)
    dyn_all = np.concatenate(all_dyn, axis=0).astype(bool)
    particle_all = np.concatenate(all_particle, axis=0).astype(bool)
    transient_all = np.concatenate(all_transient, axis=0).astype(bool)
    aggregate = {
        "inputs": {
            "manifests": [str(path.resolve()) for path in manifests],
            "num_clips": len(manifests),
            "image_hw": [int(image_hw[0]), int(image_hw[1])],
            "max_frames": int(args.max_frames),
            "recompute": bool(args.recompute),
        },
        "coverage": {
            "dynamic_object": float(dyn_all.mean()),
            "particle": float(particle_all.mean()),
            "transient": float(transient_all.mean()),
            "pseudo": float(pred_all.mean()),
        },
        "metrics_vs_dynamic": binary_mask_metrics(pred_all, dyn_all),
        "metrics_vs_particle": binary_mask_metrics(pred_all, particle_all),
        "metrics_vs_transient": binary_mask_metrics(pred_all, transient_all),
        "iou_vs_transient": float(_mask_iou(pred_all, transient_all)),
        "per_clip": per_clip,
        "notes": [
            "This evaluates the non-oracle RGB prefilter mask only; it does not run D4RT.",
            "For WebUOT, GT is a tracked-target bounding-box mask, not full fish instance segmentation.",
        ],
    }
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    with (output_dir / "per_clip_metrics.jsonl").open("w", encoding="utf-8") as f:
        for item in per_clip:
            f.write(json.dumps(item) + "\n")
    print(f"Saved aggregate: {output_dir / 'aggregate_metrics.json'}")
    metrics = aggregate["metrics_vs_transient"]
    print(
        "RGB prefilter vs transient: "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} iou={aggregate['iou_vs_transient']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
