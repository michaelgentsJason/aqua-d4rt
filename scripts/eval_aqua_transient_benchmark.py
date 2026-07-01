#!/usr/bin/env python3
"""Evaluate Aqua-D4RT transient heads over many synthetic clip manifests."""

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
    _best_f1,
    _binary_metrics,
    _grid_queries,
    _load_clip,
    _load_model,
    _sigmoid,
)
from infer_track_3d import _resolve_device  # noqa: E402
from src.core import load_yaml_config, seed_everything  # noqa: E402
from src.eval.tasks import _encode_model_memory, _run_model_for_queries  # noqa: E402


def _read_manifest_list(path: str | Path) -> list[str]:
    p = Path(path)
    items: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
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


def _static_curve(static_probs: np.ndarray, transient_labels: np.ndarray, thresholds: list[float]) -> list[dict[str, Any]]:
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


def _evaluate_manifest(
    *,
    manifest_path: Path,
    model: torch.nn.Module,
    model_config: Path,
    image_hw: tuple[int, int],
    device: torch.device,
    max_frames: int,
    grid_stride: int,
    query_chunk_size: int,
    static_thresholds: list[float],
    calibrated_thresholds: dict[str, float],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    video, dynamic_mask, particle_mask, manifest = _load_clip(
        manifest_path,
        image_hw=image_hw,
        max_frames=int(max_frames),
    )
    query_cpu, coord_txy = _grid_queries(video.shape[0], video.shape[1], video.shape[2], stride=int(grid_stride))
    labels_dynamic = dynamic_mask[coord_txy[:, 0], coord_txy[:, 2], coord_txy[:, 1]]
    labels_particle = particle_mask[coord_txy[:, 0], coord_txy[:, 2], coord_txy[:, 1]]
    labels_transient = labels_dynamic | labels_particle
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

    dynamic_probs = _sigmoid(pred["dynamic_object_logit"].numpy())
    particle_probs = _sigmoid(pred["particle_logit"].numpy())
    confidence_probs = _sigmoid(pred["confidence"].numpy())
    if "static_confidence" in pred:
        static_probs = pred["static_confidence"].numpy()
    else:
        static_probs = confidence_probs * (1.0 - dynamic_probs) * (1.0 - particle_probs)

    metrics = {
        "manifest": str(manifest_path.resolve()),
        "model_config": str(model_config.resolve()),
        "grid_stride": int(grid_stride),
        "num_queries": int(coord_txy.shape[0]),
        "num_frames": int(video.shape[0]),
        "image_hw": [int(video.shape[1]), int(video.shape[2])],
        "mask_coverage_eval_grid": {
            "dynamic_object": float(labels_dynamic.mean()),
            "particle": float(labels_particle.mean()),
            "transient": float(labels_transient.mean()),
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
        "static_contamination_curve": _static_curve(static_probs, labels_transient, static_thresholds),
    }
    if "dynamic_object" in calibrated_thresholds:
        metrics["dynamic_object"]["calibrated_threshold"] = _binary_metrics(
            dynamic_probs,
            labels_dynamic,
            threshold=calibrated_thresholds["dynamic_object"],
        )
    if "particle" in calibrated_thresholds:
        metrics["particle"]["calibrated_threshold"] = _binary_metrics(
            particle_probs,
            labels_particle,
            threshold=calibrated_thresholds["particle"],
        )
    if "static" in calibrated_thresholds:
        metrics["static"]["calibrated_threshold"] = _binary_metrics(
            static_probs,
            ~labels_transient,
            threshold=calibrated_thresholds["static"],
        )
        metrics["static_calibrated_contamination"] = _static_curve(
            static_probs,
            labels_transient,
            [calibrated_thresholds["static"]],
        )[0]
    arrays = {
        "dynamic_probs": dynamic_probs.astype(np.float32),
        "particle_probs": particle_probs.astype(np.float32),
        "static_probs": static_probs.astype(np.float32),
        "labels_dynamic": labels_dynamic.astype(bool),
        "labels_particle": labels_particle.astype(bool),
        "labels_transient": labels_transient.astype(bool),
    }
    return metrics, arrays


def _aggregate(
    per_clip: list[dict[str, Any]],
    arrays: list[dict[str, np.ndarray]],
    thresholds: list[float],
    calibrated_thresholds: dict[str, float],
) -> dict[str, Any]:
    dynamic_probs = np.concatenate([item["dynamic_probs"] for item in arrays], axis=0)
    particle_probs = np.concatenate([item["particle_probs"] for item in arrays], axis=0)
    static_probs = np.concatenate([item["static_probs"] for item in arrays], axis=0)
    labels_dynamic = np.concatenate([item["labels_dynamic"] for item in arrays], axis=0)
    labels_particle = np.concatenate([item["labels_particle"] for item in arrays], axis=0)
    labels_transient = np.concatenate([item["labels_transient"] for item in arrays], axis=0)
    aggregate = {
        "num_clips": len(per_clip),
        "num_queries": int(dynamic_probs.shape[0]),
        "mask_coverage_eval_grid": {
            "dynamic_object": float(labels_dynamic.mean()),
            "particle": float(labels_particle.mean()),
            "transient": float(labels_transient.mean()),
        },
        "dynamic_object": {
            "threshold_0_5": _binary_metrics(dynamic_probs, labels_dynamic, threshold=0.5),
            "best_f1": _best_f1(dynamic_probs, labels_dynamic),
            "per_clip_f1_at_0_5_mean": float(np.mean([m["dynamic_object"]["threshold_0_5"]["f1"] for m in per_clip])),
        },
        "particle": {
            "threshold_0_5": _binary_metrics(particle_probs, labels_particle, threshold=0.5),
            "best_f1": _best_f1(particle_probs, labels_particle),
            "per_clip_f1_at_0_5_mean": float(np.mean([m["particle"]["threshold_0_5"]["f1"] for m in per_clip])),
        },
        "static": {
            "threshold_0_5": _binary_metrics(static_probs, ~labels_transient, threshold=0.5),
            "best_f1": _best_f1(static_probs, ~labels_transient),
            "per_clip_f1_at_0_5_mean": float(np.mean([m["static"]["threshold_0_5"]["f1"] for m in per_clip])),
        },
        "static_contamination_curve": _static_curve(static_probs, labels_transient, thresholds),
    }
    if "dynamic_object" in calibrated_thresholds:
        aggregate["dynamic_object"]["calibrated_threshold"] = _binary_metrics(
            dynamic_probs,
            labels_dynamic,
            threshold=calibrated_thresholds["dynamic_object"],
        )
    if "particle" in calibrated_thresholds:
        aggregate["particle"]["calibrated_threshold"] = _binary_metrics(
            particle_probs,
            labels_particle,
            threshold=calibrated_thresholds["particle"],
        )
    if "static" in calibrated_thresholds:
        aggregate["static"]["calibrated_threshold"] = _binary_metrics(
            static_probs,
            ~labels_transient,
            threshold=calibrated_thresholds["static"],
        )
        aggregate["static_calibrated_contamination"] = _static_curve(
            static_probs,
            labels_transient,
            [calibrated_thresholds["static"]],
        )[0]
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=None, help="Manifest path or comma-separated paths.")
    parser.add_argument("--manifest-list", action="append", default=None, help="Text file with one manifest path per line.")
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--grid-stride", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--static-thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--dynamic-threshold", type=float, default=None)
    parser.add_argument("--particle-threshold", type=float, default=None)
    parser.add_argument("--static-threshold", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    device = _resolve_device(args.device)
    model_config = Path(args.model_config)
    cfg = load_yaml_config(model_config)
    image_hw = tuple(int(v) for v in cfg["model"]["input"].get("image_size", [256, 256]))
    thresholds = [float(item) for item in str(args.static_thresholds).split(",") if item.strip()]
    calibrated_thresholds: dict[str, float] = {}
    if args.dynamic_threshold is not None:
        calibrated_thresholds["dynamic_object"] = float(args.dynamic_threshold)
    if args.particle_threshold is not None:
        calibrated_thresholds["particle"] = float(args.particle_threshold)
    if args.static_threshold is not None:
        calibrated_thresholds["static"] = float(args.static_threshold)
    manifests = _resolve_manifests(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(model_config, Path(args.ckpt_path), device=device)

    per_clip: list[dict[str, Any]] = []
    arrays: list[dict[str, np.ndarray]] = []
    jsonl_path = output_dir / "per_clip_metrics.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for idx, manifest_path in enumerate(manifests):
            print(f"[{idx + 1}/{len(manifests)}] {manifest_path}", flush=True)
            metrics, clip_arrays = _evaluate_manifest(
                manifest_path=manifest_path,
                model=model,
                model_config=model_config,
                image_hw=image_hw,
                device=device,
                max_frames=int(args.max_frames),
                grid_stride=int(args.grid_stride),
                query_chunk_size=int(args.query_chunk_size),
                static_thresholds=thresholds,
                calibrated_thresholds=calibrated_thresholds,
            )
            per_clip.append(metrics)
            arrays.append(clip_arrays)
            fp.write(json.dumps(metrics) + "\n")
            fp.flush()

    aggregate = _aggregate(per_clip, arrays, thresholds, calibrated_thresholds)
    summary = {
        "inputs": {
            "manifests": [str(path.resolve()) for path in manifests],
            "model_config": str(model_config.resolve()),
            "ckpt_path": str(Path(args.ckpt_path).resolve()),
            "grid_stride": int(args.grid_stride),
            "max_frames": int(args.max_frames),
            "device": str(device),
            "calibrated_thresholds": calibrated_thresholds,
        },
        "aggregate": aggregate,
    }
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    brief = {
        name: {
            "f1_at_0_5": aggregate[name]["threshold_0_5"]["f1"],
            "precision_at_0_5": aggregate[name]["threshold_0_5"]["precision"],
            "recall_at_0_5": aggregate[name]["threshold_0_5"]["recall"],
            "best_f1": aggregate[name]["best_f1"]["f1"],
            "calibrated_f1": aggregate[name].get("calibrated_threshold", {}).get("f1"),
            "calibrated_threshold": aggregate[name].get("calibrated_threshold", {}).get("threshold"),
        }
        for name in ("dynamic_object", "particle", "static")
    }
    (output_dir / "summary_brief.json").write_text(json.dumps(brief, indent=2), encoding="utf-8")
    for name, values in brief.items():
        print(
            f"{name}: f1@0.5={values['f1_at_0_5']:.4f} "
            f"p={values['precision_at_0_5']:.4f} r={values['recall_at_0_5']:.4f} "
            f"best_f1={values['best_f1']:.4f}",
            flush=True,
        )
    print(f"Saved aggregate: {output_dir / 'aggregate_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
