#!/usr/bin/env python3
"""Build keypoint-level training data for Aqua-D4RT retention scoring."""

from __future__ import annotations

import argparse
import csv
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

from aqua_retention_utils import (  # noqa: E402
    build_retention_candidate_table,
    extract_keypoint_context,
    resolve_manifests,
    save_candidate_npz,
    summarize_candidate_table,
    aqua_dense_score_maps,
)
from eval_aqua_transient_heads import _load_clip, _load_model  # noqa: E402
from infer_track_3d import _resolve_device  # noqa: E402
from src.core import load_yaml_config, seed_everything  # noqa: E402


def _merge_tables(tables: list[dict[str, Any]]) -> dict[str, Any]:
    if not tables:
        return {
            "features": np.zeros((0, 0), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
            "candidate_meta": np.zeros((0, 9), dtype=np.float32),
            "feature_names": [],
            "candidate_meta_names": [],
            "summary": {},
        }
    features = np.concatenate([item["features"] for item in tables], axis=0)
    labels = np.concatenate([item["labels"] for item in tables], axis=0)
    candidate_meta = np.concatenate([item["candidate_meta"] for item in tables], axis=0)
    return {
        "features": features,
        "labels": labels,
        "candidate_meta": candidate_meta,
        "feature_names": tables[0]["feature_names"],
        "candidate_meta_names": tables[0]["candidate_meta_names"],
        "summary": summarize_candidate_table(features, labels, candidate_meta),
    }


def _write_summary_csv(path: Path, per_clip: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    rows = []
    for item in per_clip:
        row = {
            "clip_index": item["clip_index"],
            "clip_name": item["clip_name"],
            "manifest": item["manifest"],
        }
        row.update(item["candidate_summary"])
        row.update(
            {
                "num_keypoints": item["keypoint_summary"]["num_keypoints"],
                "num_pairs": item["keypoint_summary"]["num_pairs"],
                "num_pairs_with_model": item["keypoint_summary"]["num_pairs_with_model"],
                "rejected_pixel_fraction": item["rejected_pixel_fraction"],
                "gt_transient_fraction": item["gt_transient_fraction"],
            }
        )
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "clip_index",
            "clip_name",
            "manifest",
            "num_candidates",
            "positive_rate",
            "gt_transient_rate",
            "stable_rate",
            "mean_inlier_support",
            "num_keypoints",
            "num_pairs",
            "num_pairs_with_model",
            "rejected_pixel_fraction",
            "gt_transient_fraction",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=None)
    parser.add_argument("--manifest-list", action="append", default=None)
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="retention_candidates.npz")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--aqua-grid-stride", type=int, default=4)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--dynamic-threshold", type=float, default=0.79)
    parser.add_argument("--particle-threshold", type=float, default=0.83)
    parser.add_argument("--static-threshold", type=float, default=0.55)
    parser.add_argument("--detector", default="orb", choices=("orb", "sift"))
    parser.add_argument("--max-features", type=int, default=1200)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--min-positive-inlier-support", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    manifests = resolve_manifests(
        manifest=args.manifest,
        manifest_list=args.manifest_list,
        max_clips=int(args.max_clips),
    )
    device = _resolve_device(args.device)
    cfg = load_yaml_config(args.model_config)
    image_hw = tuple(int(v) for v in cfg["model"]["input"].get("image_size", [256, 256]))
    model = _load_model(Path(args.model_config), Path(args.ckpt_path), device=device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables: list[dict[str, Any]] = []
    per_clip: list[dict[str, Any]] = []
    rejected_name = f"aqua_static_conf_ge_{float(args.static_threshold):.3f}".replace(".", "p")

    for clip_index, manifest_path in enumerate(manifests):
        print(f"[{clip_index + 1}/{len(manifests)}] {manifest_path}")
        video, dynamic_mask, particle_mask, manifest = _load_clip(
            manifest_path,
            image_hw=image_hw,
            max_frames=int(args.max_frames),
        )
        transient_gt = (dynamic_mask | particle_mask).astype(bool)
        aqua = aqua_dense_score_maps(
            model=model,
            video=video,
            manifest=manifest,
            device=device,
            grid_stride=int(args.aqua_grid_stride),
            query_chunk_size=int(args.query_chunk_size),
            dynamic_threshold=float(args.dynamic_threshold),
            particle_threshold=float(args.particle_threshold),
            static_thresholds=[float(args.static_threshold)],
        )
        rejected_mask = aqua["rejected_masks"][rejected_name]
        keypoint_context = extract_keypoint_context(
            video_rgb=video,
            detector_name=str(args.detector),
            max_features=int(args.max_features),
            ratio=float(args.ratio),
            frame_step=int(args.frame_step),
        )
        table = build_retention_candidate_table(
            video_rgb=video,
            transient_gt=transient_gt,
            score_maps=aqua["score_maps"],
            rejected_mask=rejected_mask,
            keypoint_context=keypoint_context,
            dynamic_threshold=float(args.dynamic_threshold),
            particle_threshold=float(args.particle_threshold),
            static_threshold=float(args.static_threshold),
            min_positive_inlier_support=int(args.min_positive_inlier_support),
            clip_index=clip_index,
        )
        tables.append(table)
        per_clip.append(
            {
                "clip_index": int(clip_index),
                "manifest": str(manifest_path.resolve()),
                "clip_name": str(manifest.get("name", manifest_path.parent.name)),
                "dataset": str(manifest.get("dataset", "unknown")),
                "num_frames": int(video.shape[0]),
                "image_hw": [int(video.shape[1]), int(video.shape[2])],
                "gt_transient_fraction": float(transient_gt.mean()),
                "rejected_variant": rejected_name,
                "rejected_pixel_fraction": float(rejected_mask.mean()),
                "candidate_summary": table["summary"],
                "keypoint_summary": {
                    "num_keypoints": int(keypoint_context["num_keypoints"]),
                    "num_pairs": int(keypoint_context["num_pairs"]),
                    "num_pairs_with_model": int(keypoint_context["num_pairs_with_model"]),
                },
                "aqua_meta": aqua["meta"],
            }
        )
        print(
            "  candidates={num_candidates} pos={positive_rate:.3f} gt_trans={gt_transient_rate:.3f} "
            "stable={stable_rate:.3f}".format(**table["summary"])
        )

    merged = _merge_tables(tables)
    metadata = {
        "model_config": str(Path(args.model_config).resolve()),
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "num_manifests": len(manifests),
        "manifest_paths": [str(path.resolve()) for path in manifests],
        "image_hw": [int(image_hw[0]), int(image_hw[1])],
        "max_frames": int(args.max_frames),
        "aqua_grid_stride": int(args.aqua_grid_stride),
        "dynamic_threshold": float(args.dynamic_threshold),
        "particle_threshold": float(args.particle_threshold),
        "static_threshold": float(args.static_threshold),
        "rejected_variant": rejected_name,
        "detector": str(args.detector),
        "max_features": int(args.max_features),
        "ratio": float(args.ratio),
        "frame_step": int(args.frame_step),
        "min_positive_inlier_support": int(args.min_positive_inlier_support),
        "label_definition": (
            "positive iff candidate keypoint lies in Aqua-rejected region, is GT-static, "
            "and has adjacent-frame essential-matrix inlier support. GT is used only for training/evaluation."
        ),
    }
    save_candidate_npz(output_dir / str(args.output_name), merged, metadata=metadata)
    result = {"metadata": metadata, "aggregate": merged["summary"], "per_clip": per_clip}
    (output_dir / "candidate_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_summary_csv(output_dir / "candidate_summary.csv", per_clip, merged["summary"])

    print("Aggregate retention candidate summary:")
    for key, value in merged["summary"].items():
        print(f"- {key}: {value}")
    print(f"Saved: {output_dir / str(args.output_name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
