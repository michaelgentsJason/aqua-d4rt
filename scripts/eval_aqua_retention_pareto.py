#!/usr/bin/env python3
"""Evaluate Aqua-D4RT learned retention as a SLAM-front-end Pareto sweep."""

from __future__ import annotations

import argparse
import csv
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

from aqua_prefilter_utils import temporal_rgb_pseudo_mask  # noqa: E402
from aqua_retention_utils import (  # noqa: E402
    build_retention_candidate_table,
    extract_keypoint_context,
    load_retention_scorer,
    resolve_manifests,
    retention_mask_from_candidates,
    score_retention_candidates,
    aqua_dense_score_maps,
)
from eval_aqua_downstream_slam_proxy import (  # noqa: E402
    _detect_features,
    _match_descriptors,
    _pair_metrics,
    _resize_mask_stack,
    _safe_mean,
    _sample_mask,
    _slam_aware_retention_mask,
    _summarize_variant,
)
from eval_aqua_transient_heads import _load_clip, _load_model  # noqa: E402
from infer_track_3d import _resolve_device  # noqa: E402
from src.core import load_yaml_config, seed_everything  # noqa: E402


def _load_pseudo_mask(manifest: dict[str, Any], video: np.ndarray) -> tuple[np.ndarray, str]:
    pseudo_path = manifest.get("pseudo_mask_npz")
    if pseudo_path and Path(str(pseudo_path)).exists():
        payload = np.load(str(pseudo_path))
        for key in ("pseudo_dynamic_mask", "dynamic_object_mask", "transient_mask"):
            if key in payload:
                return payload[key].astype(bool)[: video.shape[0]], str(pseudo_path)
    return temporal_rgb_pseudo_mask(video).astype(bool), "computed_temporal_rgb"


def _make_detector(name: str, max_features: int) -> tuple[Any, int]:
    detector_name = str(name).lower()
    if detector_name == "sift" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=int(max_features)), cv2.NORM_L2
    if detector_name != "orb":
        print(f"Detector {name!r} unavailable; falling back to ORB.")
    return cv2.ORB_create(nfeatures=int(max_features), fastThreshold=12), cv2.NORM_HAMMING


def _evaluate_variants(
    *,
    video: np.ndarray,
    transient_gt: np.ndarray,
    variants: dict[str, np.ndarray],
    detector_name: str,
    max_features: int,
    ratio: float,
    frame_step: int,
) -> dict[str, Any]:
    detector, norm = _make_detector(detector_name, max_features)
    h, w = video.shape[1:3]
    focal = float(max(h, w))
    camera_matrix = np.asarray([[focal, 0.0, 0.5 * (w - 1)], [0.0, focal, 0.5 * (h - 1)], [0.0, 0.0, 1.0]])
    frame_ids = list(range(0, video.shape[0], max(1, int(frame_step))))
    if frame_ids[-1] != video.shape[0] - 1:
        frame_ids.append(video.shape[0] - 1)

    out: dict[str, Any] = {}
    for name, static_mask in variants.items():
        keypoints_by_frame: dict[int, list[cv2.KeyPoint]] = {}
        descriptors_by_frame: dict[int, np.ndarray | None] = {}
        frame_metrics: list[dict[str, Any]] = []
        for frame_idx in frame_ids:
            keypoints, descriptors = _detect_features(
                frame_rgb=video[frame_idx],
                static_mask=static_mask[frame_idx],
                detector=detector,
                max_features=int(max_features),
            )
            pts = np.asarray([kp.pt for kp in keypoints], dtype=np.float32) if keypoints else np.zeros((0, 2), dtype=np.float32)
            contam = _sample_mask(transient_gt, pts, frame_idx)
            keypoints_by_frame[frame_idx] = keypoints
            descriptors_by_frame[frame_idx] = descriptors
            frame_metrics.append(
                {
                    "frame": int(frame_idx),
                    "num_features": int(len(keypoints)),
                    "contaminated_features": int(contam.sum()),
                    "feature_contamination": float(contam.mean()) if contam.size else 0.0,
                    "static_mask_fraction": float(static_mask[frame_idx].mean()),
                    "gt_transient_fraction": float(transient_gt[frame_idx].mean()),
                }
            )

        pair_metrics: list[dict[str, Any]] = []
        for f0, f1 in zip(frame_ids[:-1], frame_ids[1:]):
            pair_metrics.append(
                _pair_metrics(
                    keypoints0=keypoints_by_frame[f0],
                    descriptors0=descriptors_by_frame[f0],
                    keypoints1=keypoints_by_frame[f1],
                    descriptors1=descriptors_by_frame[f1],
                    transient_mask=transient_gt,
                    frame0=f0,
                    frame1=f1,
                    norm=norm,
                    ratio=float(ratio),
                    camera_matrix=camera_matrix,
                )
            )
        out[name] = {
            "summary": _summarize_variant(frame_metrics, pair_metrics),
            "frame_metrics": frame_metrics,
            "pair_metrics": pair_metrics,
        }
    return out


def _aggregate(per_clip: list[dict[str, Any]]) -> dict[str, Any]:
    variants: list[str] = []
    for clip in per_clip:
        for variant in clip["variants"]:
            if variant not in variants:
                variants.append(variant)
    out: dict[str, Any] = {"num_clips": len(per_clip), "variants": {}}
    for variant in variants:
        summaries = [clip["variants"][variant]["summary"] for clip in per_clip if variant in clip["variants"]]
        total_features = int(sum(int(item["total_features"]) for item in summaries))
        total_matches = int(sum(int(item["total_matches"]) for item in summaries))
        total_pairs = int(sum(int(item["pairs"]) for item in summaries))
        success_pairs = int(round(sum(float(item["essential_success_rate"]) * int(item["pairs"]) for item in summaries)))
        feature_contam_num = sum(float(item["feature_contamination"]) * int(item["total_features"]) for item in summaries)
        out["variants"][variant] = {
            "clips": len(summaries),
            "total_features": total_features,
            "features_per_frame_mean": _safe_mean([float(item["features_per_frame_mean"]) for item in summaries]),
            "feature_contamination": float(feature_contam_num) / float(max(1, total_features)),
            "total_matches": total_matches,
            "matches_per_pair_mean": _safe_mean([float(item["matches_per_pair_mean"]) for item in summaries]),
            "match_contamination_mean": _safe_mean([float(item["match_contamination_mean"]) for item in summaries]),
            "essential_success_rate": float(success_pairs) / float(max(1, total_pairs)),
            "essential_inliers_per_pair_mean": _safe_mean([float(item["essential_inliers_per_pair_mean"]) for item in summaries]),
            "essential_inlier_rate_mean": _safe_mean([float(item["essential_inlier_rate_mean"]) for item in summaries]),
            "static_mask_fraction_mean": _safe_mean(
                [
                    float(np.mean([frame["static_mask_fraction"] for frame in clip["variants"][variant]["frame_metrics"]]))
                    for clip in per_clip
                    if variant in clip["variants"]
                ]
            ),
        }
    out["mean_mask_coverage"] = {
        key: _safe_mean([float(clip["mask_coverage"][key]) for clip in per_clip])
        for key in ("dynamic_object", "particle", "transient")
    }
    return out


def _write_summary_csv(path: Path, aggregate: dict[str, Any]) -> None:
    rows = []
    for variant, metrics in aggregate["variants"].items():
        row = {"variant": variant}
        row.update(metrics)
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["variant"])
        writer.writeheader()
        writer.writerows(rows)


def _parse_thresholds(value: str | None, default: float) -> list[float]:
    if not value:
        return [float(default)]
    return [float(part) for part in str(value).split(",") if part.strip()]


def _evaluate_manifest(
    *,
    manifest_path: Path,
    model: torch.nn.Module,
    scorer: dict[str, Any] | None,
    image_hw: tuple[int, int],
    device: torch.device,
    max_frames: int,
    aqua_grid_stride: int,
    query_chunk_size: int,
    dynamic_threshold: float,
    particle_threshold: float,
    static_threshold: float,
    detector_name: str,
    max_features: int,
    ratio: float,
    frame_step: int,
    include_oracle: bool,
    include_temporal_rgb: bool,
    include_rule_retention: bool,
    learned_thresholds: list[float],
    retention_patch_radius: int,
    retention_min_inlier_support: int,
    retention_max_features_per_frame: int,
    retention_max_fraction: float,
    clip_index: int,
) -> dict[str, Any]:
    video, dynamic_mask, particle_mask, manifest = _load_clip(
        manifest_path,
        image_hw=image_hw,
        max_frames=int(max_frames),
    )
    transient_gt = (dynamic_mask | particle_mask).astype(bool)
    rejected_name = f"aqua_static_conf_ge_{float(static_threshold):.3f}".replace(".", "p")
    aqua = aqua_dense_score_maps(
        model=model,
        video=video,
        manifest=manifest,
        device=device,
        grid_stride=int(aqua_grid_stride),
        query_chunk_size=int(query_chunk_size),
        dynamic_threshold=float(dynamic_threshold),
        particle_threshold=float(particle_threshold),
        static_thresholds=[float(static_threshold)],
    )
    aqua_mask = aqua["rejected_masks"][rejected_name]
    variants: dict[str, np.ndarray] = {
        "raw_all_pixels": np.ones(transient_gt.shape, dtype=bool),
        rejected_name: ~aqua_mask,
    }
    retention_meta: dict[str, Any] = {}

    if include_rule_retention:
        retain_mask, meta = _slam_aware_retention_mask(
            video_rgb=video,
            rejected_mask=aqua_mask,
            detector_name=str(detector_name),
            max_features=int(max_features),
            ratio=float(ratio),
            frame_step=int(frame_step),
            patch_radius=int(retention_patch_radius),
            min_inlier_support=int(retention_min_inlier_support),
            max_retained_features_per_frame=int(retention_max_features_per_frame),
            max_retained_fraction=float(retention_max_fraction),
        )
        name = f"{rejected_name}_rule_slam_retain"
        variants[name] = (~aqua_mask) | retain_mask
        retention_meta[name] = meta

    if scorer is not None:
        keypoint_context = extract_keypoint_context(
            video_rgb=video,
            detector_name=str(detector_name),
            max_features=int(max_features),
            ratio=float(ratio),
            frame_step=int(frame_step),
        )
        table = build_retention_candidate_table(
            video_rgb=video,
            transient_gt=transient_gt,
            score_maps=aqua["score_maps"],
            rejected_mask=aqua_mask,
            keypoint_context=keypoint_context,
            dynamic_threshold=float(dynamic_threshold),
            particle_threshold=float(particle_threshold),
            static_threshold=float(static_threshold),
            min_positive_inlier_support=int(retention_min_inlier_support),
            clip_index=int(clip_index),
        )
        scores = score_retention_candidates(
            table["features"],
            scorer,
            device=device,
            feature_names=table["feature_names"],
        )
        for threshold in learned_thresholds:
            retain_mask, meta = retention_mask_from_candidates(
                candidate_meta=table["candidate_meta"],
                scores=scores,
                rejected_mask=aqua_mask,
                score_threshold=float(threshold),
                patch_radius=int(retention_patch_radius),
                max_features_per_frame=int(retention_max_features_per_frame),
                max_fraction=float(retention_max_fraction),
            )
            name = f"{rejected_name}_learned_retain_t{float(threshold):.2f}".replace(".", "p")
            variants[name] = (~aqua_mask) | retain_mask
            meta.update(table["summary"])
            retention_meta[name] = meta

    pseudo_source = None
    if include_temporal_rgb:
        pseudo_mask, pseudo_source = _load_pseudo_mask(manifest, video)
        pseudo_mask = _resize_mask_stack(pseudo_mask[: video.shape[0]], image_hw)
        variants["temporal_rgb_static"] = ~pseudo_mask
    if include_oracle:
        variants["oracle_gt_static"] = ~transient_gt

    result_variants = _evaluate_variants(
        video=video,
        transient_gt=transient_gt,
        variants=variants,
        detector_name=str(detector_name),
        max_features=int(max_features),
        ratio=float(ratio),
        frame_step=int(frame_step),
    )
    return {
        "manifest": str(manifest_path.resolve()),
        "clip_name": str(manifest.get("name", manifest_path.parent.name)),
        "dataset": str(manifest.get("dataset", "unknown")),
        "num_frames": int(video.shape[0]),
        "image_hw": [int(video.shape[1]), int(video.shape[2])],
        "mask_coverage": {
            "dynamic_object": float(dynamic_mask.mean()),
            "particle": float(particle_mask.mean()),
            "transient": float(transient_gt.mean()),
        },
        "aqua_mask_meta": aqua["meta"],
        "rejected_variant": rejected_name,
        "pseudo_mask_source": pseudo_source,
        "retention_meta": retention_meta,
        "variants": result_variants,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=None)
    parser.add_argument("--manifest-list", action="append", default=None)
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--scorer-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--aqua-grid-stride", type=int, default=4)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--dynamic-threshold", type=float, default=0.79)
    parser.add_argument("--particle-threshold", type=float, default=0.83)
    parser.add_argument("--static-threshold", type=float, default=0.55)
    parser.add_argument("--detector", default="orb", choices=("orb", "sift"))
    parser.add_argument("--max-features", type=int, default=1200)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--learned-thresholds", default=None)
    parser.add_argument("--include-rule-retention", action="store_true")
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--no-temporal-rgb", action="store_true")
    parser.add_argument("--retention-patch-radius", type=int, default=3)
    parser.add_argument("--retention-min-inlier-support", type=int, default=1)
    parser.add_argument("--retention-max-features-per-frame", type=int, default=80)
    parser.add_argument("--retention-max-fraction", type=float, default=0.06)
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
    scorer = load_retention_scorer(args.scorer_path, device=device) if args.scorer_path else None
    default_threshold = scorer["threshold"] if scorer is not None else 0.5
    learned_thresholds = _parse_thresholds(args.learned_thresholds, default=float(default_threshold))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_clip: list[dict[str, Any]] = []
    for idx, manifest_path in enumerate(manifests):
        print(f"[{idx + 1}/{len(manifests)}] {manifest_path}")
        per_clip.append(
            _evaluate_manifest(
                manifest_path=manifest_path,
                model=model,
                scorer=scorer,
                image_hw=image_hw,
                device=device,
                max_frames=int(args.max_frames),
                aqua_grid_stride=int(args.aqua_grid_stride),
                query_chunk_size=int(args.query_chunk_size),
                dynamic_threshold=float(args.dynamic_threshold),
                particle_threshold=float(args.particle_threshold),
                static_threshold=float(args.static_threshold),
                detector_name=str(args.detector),
                max_features=int(args.max_features),
                ratio=float(args.ratio),
                frame_step=int(args.frame_step),
                include_oracle=not bool(args.no_oracle),
                include_temporal_rgb=not bool(args.no_temporal_rgb),
                include_rule_retention=bool(args.include_rule_retention),
                learned_thresholds=learned_thresholds,
                retention_patch_radius=int(args.retention_patch_radius),
                retention_min_inlier_support=int(args.retention_min_inlier_support),
                retention_max_features_per_frame=int(args.retention_max_features_per_frame),
                retention_max_fraction=float(args.retention_max_fraction),
                clip_index=int(idx),
            )
        )

    aggregate = _aggregate(per_clip)
    metadata = {
        "model_config": str(Path(args.model_config).resolve()),
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "scorer_path": str(Path(args.scorer_path).resolve()) if args.scorer_path else None,
        "num_manifests": len(manifests),
        "detector": str(args.detector),
        "max_features": int(args.max_features),
        "ratio": float(args.ratio),
        "frame_step": int(args.frame_step),
        "aqua_grid_stride": int(args.aqua_grid_stride),
        "dynamic_threshold": float(args.dynamic_threshold),
        "particle_threshold": float(args.particle_threshold),
        "static_threshold": float(args.static_threshold),
        "learned_thresholds": learned_thresholds,
        "rule_retention_included": bool(args.include_rule_retention),
        "retention": {
            "patch_radius": int(args.retention_patch_radius),
            "min_inlier_support": int(args.retention_min_inlier_support),
            "max_features_per_frame": int(args.retention_max_features_per_frame),
            "max_fraction": float(args.retention_max_fraction),
        },
        "webuot_caveat": "WebUOT masks are tracked-target bbox masks, not complete fish instance masks.",
    }
    (output_dir / "per_clip_metrics.json").write_text(json.dumps(per_clip, indent=2), encoding="utf-8")
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps({"metadata": metadata, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )
    _write_summary_csv(output_dir / "summary_table.csv", aggregate)

    print("Aqua retention Pareto summary:")
    for variant, metrics in aggregate["variants"].items():
        print(
            f"- {variant}: feat/frame={metrics['features_per_frame_mean']:.1f} "
            f"feat_contam={metrics['feature_contamination']:.4f} "
            f"match_contam={metrics['match_contamination_mean']:.4f} "
            f"E_success={metrics['essential_success_rate']:.3f}"
        )
    print(f"Saved: {output_dir / 'aggregate_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
