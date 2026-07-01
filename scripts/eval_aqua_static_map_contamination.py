#!/usr/bin/env python3
"""Evaluate transient contamination in Aqua-D4RT query point maps.

The script treats D4RT query-level xyz predictions as a lightweight point map.
It reports both point-count contamination and voxelized map contamination
against dataset transient labels. Synthetic clips have fish + particle masks;
WebUOT clips currently provide tracked-target bbox masks only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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

from aqua_prefilter_utils import query_labels_from_mask, temporal_rgb_pseudo_mask  # noqa: E402
from aqua_retention_utils import effective_aqua_scores  # noqa: E402
from eval_aqua_transient_heads import _grid_queries, _load_clip, _load_model, _sigmoid  # noqa: E402
from infer_track_3d import _resolve_device  # noqa: E402
from src.core import load_yaml_config, seed_everything  # noqa: E402
from src.eval.tasks import _encode_model_memory, _run_model_for_queries  # noqa: E402


def _read_manifest_list(path: str | Path) -> list[str]:
    items: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            items.append(item)
    return items


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _parse_external_masks(values: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values or []:
        parts = [part.strip() for part in str(value).split(",") if part.strip()]
        for part in parts:
            if "=" not in part:
                raise ValueError(f"External mask must be NAME=DIR, got: {part!r}")
            name, directory = part.split("=", 1)
            name = name.strip()
            if not name:
                raise ValueError(f"External mask name is empty in {part!r}")
            if not all(ch.isalnum() or ch in {"_", "-"} for ch in name):
                raise ValueError(f"External mask name must be alnum/_/-, got: {name!r}")
            out[name] = Path(directory.strip())
    return out


def _resolve_manifests(args: argparse.Namespace) -> list[Path]:
    items: list[str] = []
    if args.manifest:
        for value in args.manifest:
            items.extend(part.strip() for part in str(value).split(",") if part.strip())
    if args.manifest_list:
        for value in args.manifest_list:
            items.extend(_read_manifest_list(value))
    out: list[Path] = []
    seen: set[str] = set()
    for item in items:
        path = Path(item)
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            out.append(path)
    if args.max_clips > 0:
        out = out[: int(args.max_clips)]
    if not out:
        raise ValueError("Provide --manifest or --manifest-list.")
    return out


def _load_pseudo_mask(manifest: dict[str, Any], video: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    pseudo_path = manifest.get("pseudo_mask_npz")
    if pseudo_path and Path(str(pseudo_path)).exists():
        payload = np.load(str(pseudo_path))
        for key in ("pseudo_dynamic_mask", "dynamic_object_mask", "transient_mask"):
            if key in payload:
                return payload[key].astype(bool)[: video.shape[0]], str(pseudo_path)
    return temporal_rgb_pseudo_mask(video).astype(bool), "computed_temporal_rgb"


def _resize_mask_stack(mask: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = image_hw
    if mask.shape[1:3] == (h, w):
        return mask.astype(bool)
    out = [
        cv2.resize(mask[t].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        for t in range(mask.shape[0])
    ]
    return np.stack(out, axis=0)


def _load_external_pred_mask(
    *,
    mask_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    image_hw: tuple[int, int],
    num_frames: int,
) -> tuple[np.ndarray, str]:
    clip_names = [
        str(manifest.get("name", "")),
        str(manifest_path.parent.name),
        str(manifest_path.stem),
    ]
    candidates: list[Path] = []
    for name in clip_names:
        if not name:
            continue
        candidates.append(mask_dir / f"{name}.npz")
        candidates.append(mask_dir / f"{_safe_stem(name)}.npz")
    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    for candidate in unique_candidates:
        if not candidate.exists():
            continue
        payload = np.load(str(candidate), allow_pickle=False)
        if "pred_mask" not in payload:
            raise KeyError(f"{candidate} does not contain key 'pred_mask'")
        mask = np.asarray(payload["pred_mask"]).astype(bool)
        if mask.ndim != 3:
            raise ValueError(f"{candidate}: pred_mask must have shape T,H,W, got {mask.shape}")
        mask = mask[: int(num_frames)]
        if mask.shape[0] < int(num_frames):
            pad = np.zeros((int(num_frames) - mask.shape[0], mask.shape[1], mask.shape[2]), dtype=bool)
            mask = np.concatenate([mask, pad], axis=0)
        return _resize_mask_stack(mask, image_hw), str(candidate.resolve())
    expected = ", ".join(str(path) for path in unique_candidates[:4])
    raise FileNotFoundError(f"No external mask cache found for {manifest_path}. Tried: {expected}")


def _safe_rate(num: int, den: int) -> float:
    return float(num) / float(max(1, den))


def _point_metrics(keep: np.ndarray, labels_transient: np.ndarray, total_valid: int) -> dict[str, Any]:
    keep_b = keep.astype(bool)
    transient = labels_transient.astype(bool)
    static = ~transient
    kept = int(keep_b.sum())
    kept_transient = int(np.logical_and(keep_b, transient).sum())
    kept_static = int(np.logical_and(keep_b, static).sum())
    total_static = int(static.sum())
    total_transient = int(transient.sum())
    return {
        "total_valid_points": int(total_valid),
        "total_static_points": total_static,
        "total_transient_points": total_transient,
        "kept_points": kept,
        "kept_rate": _safe_rate(kept, total_valid),
        "kept_static_points": kept_static,
        "kept_transient_points": kept_transient,
        "point_contamination": _safe_rate(kept_transient, kept),
        "point_static_precision": 1.0 - _safe_rate(kept_transient, kept),
        "point_static_retention": _safe_rate(kept_static, total_static),
        "point_transient_rejection": 1.0 - _safe_rate(kept_transient, total_transient),
    }


def _voxel_ids(xyz: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    size = max(float(voxel_size), 1e-8)
    quantized = np.floor(np.asarray(xyz, dtype=np.float64) / size).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    return quantized, inverse.astype(np.int64)


def _voxel_metrics(xyz: np.ndarray, keep: np.ndarray, labels_transient: np.ndarray, voxel_size: float) -> dict[str, Any]:
    keep_b = keep.astype(bool)
    transient = labels_transient.astype(bool)
    static = ~transient
    if xyz.shape[0] == 0:
        return {
            "voxel_size": float(voxel_size),
            "total_voxels": 0,
            "kept_voxels": 0,
            "kept_contaminated_voxels": 0,
            "voxel_contamination_any": 0.0,
            "static_support_voxels": 0,
            "kept_static_support_voxels": 0,
            "voxel_static_support_retention": 0.0,
            "clean_static_only_voxels": 0,
            "kept_clean_static_only_voxels": 0,
            "clean_static_only_voxel_retention": 0.0,
        }

    _, inverse = _voxel_ids(xyz, voxel_size)
    n_voxels = int(inverse.max()) + 1
    has_static_all = np.bincount(inverse, weights=static.astype(np.float32), minlength=n_voxels) > 0
    has_transient_all = np.bincount(inverse, weights=transient.astype(np.float32), minlength=n_voxels) > 0
    clean_static_only = has_static_all & ~has_transient_all

    kept_inverse = inverse[keep_b]
    if kept_inverse.size == 0:
        kept_voxel_mask = np.zeros((n_voxels,), dtype=bool)
        has_static_kept = np.zeros((n_voxels,), dtype=bool)
        has_transient_kept = np.zeros((n_voxels,), dtype=bool)
    else:
        kept_voxel_mask = np.bincount(kept_inverse, minlength=n_voxels) > 0
        has_static_kept = np.bincount(
            kept_inverse,
            weights=static[keep_b].astype(np.float32),
            minlength=n_voxels,
        ) > 0
        has_transient_kept = np.bincount(
            kept_inverse,
            weights=transient[keep_b].astype(np.float32),
            minlength=n_voxels,
        ) > 0

    kept_voxels = int(kept_voxel_mask.sum())
    kept_contaminated = int(np.logical_and(kept_voxel_mask, has_transient_kept).sum())
    static_support_voxels = int(has_static_all.sum())
    kept_static_support = int(np.logical_and(has_static_kept, has_static_all).sum())
    clean_static_only_voxels = int(clean_static_only.sum())
    kept_clean_static_only = int(np.logical_and(kept_voxel_mask, clean_static_only).sum())
    return {
        "voxel_size": float(voxel_size),
        "total_voxels": int(n_voxels),
        "kept_voxels": kept_voxels,
        "kept_contaminated_voxels": kept_contaminated,
        "voxel_contamination_any": _safe_rate(kept_contaminated, kept_voxels),
        "static_support_voxels": static_support_voxels,
        "kept_static_support_voxels": kept_static_support,
        "voxel_static_support_retention": _safe_rate(kept_static_support, static_support_voxels),
        "clean_static_only_voxels": clean_static_only_voxels,
        "kept_clean_static_only_voxels": kept_clean_static_only,
        "clean_static_only_voxel_retention": _safe_rate(kept_clean_static_only, clean_static_only_voxels),
    }


def _rank_detection_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Compute AUROC/AP for transient query detection without sklearn."""

    y = np.asarray(labels).astype(bool).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]
    positives = int(y.sum())
    negatives = int((~y).sum())
    if y.size == 0 or positives == 0 or negatives == 0:
        return {
            "num_queries": int(y.size),
            "num_positive": positives,
            "num_negative": negatives,
            "auroc": None,
            "average_precision": None,
        }

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = s[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    pos_rank_sum = float(ranks[y].sum())
    auroc = (pos_rank_sum - positives * (positives + 1) / 2.0) / float(max(1, positives * negatives))

    desc = np.argsort(-s, kind="mergesort")
    y_desc = y[desc]
    tp = np.cumsum(y_desc.astype(np.float64))
    fp = np.cumsum((~y_desc).astype(np.float64))
    precision = tp / np.maximum(1.0, tp + fp)
    average_precision = float(precision[y_desc].sum() / float(max(1, positives)))
    return {
        "num_queries": int(y.size),
        "num_positive": positives,
        "num_negative": negatives,
        "auroc": float(auroc),
        "average_precision": average_precision,
    }


def _merge_metrics(point: dict[str, Any], voxel: dict[str, Any]) -> dict[str, Any]:
    out = dict(point)
    out.update(voxel)
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
    dynamic_threshold: float,
    particle_threshold: float,
    static_thresholds: list[float],
    static_score_modes: list[str],
    confidence_threshold: float,
    visibility_threshold: float,
    voxel_size: float,
    include_rgb_prefilter: bool,
    include_oracle: bool,
    external_masks: dict[str, Path],
) -> dict[str, Any]:
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

    xyz = pred["xyz_3d"].numpy().astype(np.float32)
    dynamic_probs = _sigmoid(pred["dynamic_object_logit"].numpy())
    particle_probs = _sigmoid(pred["particle_logit"].numpy())
    confidence_probs = _sigmoid(pred["confidence"].numpy())
    visibility_probs = _sigmoid(pred["visibility"].numpy())
    if "static_confidence" in pred:
        raw_static_probs = pred["static_confidence"].numpy().astype(np.float32)
    else:
        raw_static_probs = confidence_probs * (1.0 - dynamic_probs) * (1.0 - particle_probs)

    score_modes: dict[str, dict[str, np.ndarray]] = {
        mode: effective_aqua_scores(
            dynamic_probs=dynamic_probs,
            particle_probs=particle_probs,
            confidence_probs=confidence_probs,
            static_probs=raw_static_probs,
            static_score_mode=mode,
        )
        for mode in static_score_modes
    }
    full_scores = score_modes.get("full") or next(iter(score_modes.values()))
    static_probs = full_scores["static_confidence"]

    finite = np.isfinite(xyz).all(axis=1)
    finite &= np.isfinite(static_probs)
    finite &= confidence_probs >= float(confidence_threshold)
    finite &= visibility_probs >= float(visibility_threshold)
    if not np.any(finite):
        raise RuntimeError(f"No valid D4RT query points for {manifest_path}")

    xyz_v = xyz[finite]
    labels_v = labels_transient[finite].astype(bool)
    dyn_v = labels_dynamic[finite].astype(bool)
    particle_v = labels_particle[finite].astype(bool)
    total_valid = int(finite.sum())

    variants: dict[str, np.ndarray] = {
        "all_d4rt_points": np.ones((total_valid,), dtype=bool),
    }
    for mode, scores in score_modes.items():
        dyn_mode = scores["dynamic_prob"][finite]
        particle_mode = scores["particle_prob"][finite]
        static_mode = scores["static_confidence"][finite]
        suffix = "" if mode == "full" else f"_{mode}"
        variants[f"aqua_pred_transient_filter{suffix}"] = ~(
            (dyn_mode >= float(dynamic_threshold)) | (particle_mode >= float(particle_threshold))
        )
        for threshold in static_thresholds:
            name = f"aqua_static_conf_ge_{threshold:.3f}{suffix}".replace(".", "p")
            variants[name] = static_mode >= float(threshold)
    if include_oracle:
        variants["oracle_gt_static"] = ~labels_v
    pseudo_source = None
    if include_rgb_prefilter:
        pseudo_mask, pseudo_source = _load_pseudo_mask(manifest, video)
        if pseudo_mask is not None:
            pseudo_mask = _resize_mask_stack(pseudo_mask[: video.shape[0]], image_hw)
            pseudo_labels = query_labels_from_mask(pseudo_mask, coord_txy)[finite]
            variants["temporal_rgb_prefilter_static"] = ~pseudo_labels
    external_mask_sources: dict[str, str] = {}
    for name, mask_dir in external_masks.items():
        external_mask, source_path = _load_external_pred_mask(
            mask_dir=mask_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            image_hw=image_hw,
            num_frames=int(video.shape[0]),
        )
        external_labels = query_labels_from_mask(external_mask, coord_txy)[finite]
        variants[f"{name}_static"] = ~external_labels
        external_mask_sources[name] = source_path

    variant_metrics: dict[str, Any] = {}
    for name, keep in variants.items():
        point = _point_metrics(keep=keep, labels_transient=labels_v, total_valid=total_valid)
        voxel = _voxel_metrics(xyz=xyz_v, keep=keep, labels_transient=labels_v, voxel_size=float(voxel_size))
        variant_metrics[name] = _merge_metrics(point, voxel)

    return {
        "manifest": str(manifest_path.resolve()),
        "clip_name": str(manifest.get("name", manifest_path.parent.name)),
        "dataset": str(manifest.get("dataset", "unknown")),
        "num_frames": int(video.shape[0]),
        "image_hw": [int(video.shape[1]), int(video.shape[2])],
        "grid_stride": int(grid_stride),
        "num_queries": int(coord_txy.shape[0]),
        "num_valid_queries": total_valid,
        "mask_coverage_eval_grid": {
            "dynamic_object": float(dyn_v.mean()) if total_valid else 0.0,
            "particle": float(particle_v.mean()) if total_valid else 0.0,
            "transient": float(labels_v.mean()) if total_valid else 0.0,
        },
        "score_stats": {
            "dynamic_prob_mean": float(np.mean(dynamic_probs[finite])),
            "particle_prob_mean": float(np.mean(particle_probs[finite])),
            "confidence_prob_mean": float(np.mean(confidence_probs[finite])),
            "visibility_prob_mean": float(np.mean(visibility_probs[finite])),
            "static_confidence_mean": float(np.mean(static_probs[finite])),
            "raw_dynamic_prob_mean": float(np.mean(dynamic_probs[finite])),
            "raw_particle_prob_mean": float(np.mean(particle_probs[finite])),
            "raw_static_confidence_mean": float(np.mean(raw_static_probs[finite])),
        },
        "transient_query_detection": _rank_detection_metrics(
            scores=1.0 - static_probs[finite],
            labels=labels_v,
        ),
        "static_score_modes": list(static_score_modes),
        "variant_metrics": variant_metrics,
        "pseudo_mask_source": pseudo_source,
        "external_mask_sources": external_mask_sources,
        "model_config": str(model_config.resolve()),
    }


def _aggregate_variant(per_clip: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    point_keys = [
        "total_valid_points",
        "total_static_points",
        "total_transient_points",
        "kept_points",
        "kept_static_points",
        "kept_transient_points",
    ]
    voxel_keys = [
        "total_voxels",
        "kept_voxels",
        "kept_contaminated_voxels",
        "static_support_voxels",
        "kept_static_support_voxels",
        "clean_static_only_voxels",
        "kept_clean_static_only_voxels",
    ]
    sums: dict[str, int] = {}
    for key in point_keys + voxel_keys:
        sums[key] = int(sum(int(item["variant_metrics"][variant].get(key, 0)) for item in per_clip))
    return {
        **sums,
        "kept_rate": _safe_rate(sums["kept_points"], sums["total_valid_points"]),
        "point_contamination": _safe_rate(sums["kept_transient_points"], sums["kept_points"]),
        "point_static_precision": 1.0 - _safe_rate(sums["kept_transient_points"], sums["kept_points"]),
        "point_static_retention": _safe_rate(sums["kept_static_points"], sums["total_static_points"]),
        "point_transient_rejection": 1.0 - _safe_rate(sums["kept_transient_points"], sums["total_transient_points"]),
        "voxel_contamination_any": _safe_rate(sums["kept_contaminated_voxels"], sums["kept_voxels"]),
        "voxel_static_support_retention": _safe_rate(sums["kept_static_support_voxels"], sums["static_support_voxels"]),
        "clean_static_only_voxel_retention": _safe_rate(
            sums["kept_clean_static_only_voxels"],
            sums["clean_static_only_voxels"],
        ),
    }


def _aggregate(per_clip: list[dict[str, Any]]) -> dict[str, Any]:
    variants: list[str] = []
    for item in per_clip:
        for variant in item["variant_metrics"]:
            if variant not in variants:
                variants.append(variant)
    aurocs = [
        item.get("transient_query_detection", {}).get("auroc")
        for item in per_clip
        if item.get("transient_query_detection", {}).get("auroc") is not None
    ]
    aps = [
        item.get("transient_query_detection", {}).get("average_precision")
        for item in per_clip
        if item.get("transient_query_detection", {}).get("average_precision") is not None
    ]
    return {
        "num_clips": len(per_clip),
        "variants": {variant: _aggregate_variant(per_clip, variant) for variant in variants},
        "mean_mask_coverage_eval_grid": {
            key: float(np.mean([item["mask_coverage_eval_grid"][key] for item in per_clip]))
            for key in ("dynamic_object", "particle", "transient")
        },
        "transient_query_detection": {
            "mean_auroc": float(np.mean(aurocs)) if aurocs else None,
            "mean_average_precision": float(np.mean(aps)) if aps else None,
            "num_valid_clips": int(min(len(aurocs), len(aps))),
            "interpretation": "AUROC/AP use 1-static_confidence as the transient query score.",
        },
    }


def _write_summary_csv(path: Path, aggregate: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for variant, metrics in aggregate["variants"].items():
        rows.append(
            {
                "variant": variant,
                "kept_points": metrics["kept_points"],
                "kept_rate": metrics["kept_rate"],
                "point_contamination": metrics["point_contamination"],
                "point_static_retention": metrics["point_static_retention"],
                "point_transient_rejection": metrics["point_transient_rejection"],
                "kept_voxels": metrics["kept_voxels"],
                "voxel_contamination_any": metrics["voxel_contamination_any"],
                "voxel_static_support_retention": metrics["voxel_static_support_retention"],
                "clean_static_only_voxel_retention": metrics["clean_static_only_voxel_retention"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["variant"])
        writer.writeheader()
        writer.writerows(rows)


def _parse_threshold_from_variant(variant: str) -> float | None:
    marker = "aqua_static_conf_ge_"
    if not variant.startswith(marker):
        return None
    rest = variant[len(marker) :]
    token = rest.split("_", 1)[0].replace("p", ".")
    try:
        return float(token)
    except ValueError:
        return None


def _write_threshold_curve(path: Path, aggregate: dict[str, Any], contamination_target: float = 0.005) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant, metrics in aggregate["variants"].items():
        threshold = _parse_threshold_from_variant(variant)
        if threshold is None:
            continue
        rows.append(
            {
                "variant": variant,
                "threshold": threshold,
                "point_contamination": float(metrics["point_contamination"]),
                "point_static_retention": float(metrics["point_static_retention"]),
                "point_transient_rejection": float(metrics["point_transient_rejection"]),
                "kept_rate": float(metrics["kept_rate"]),
                "voxel_contamination_any": float(metrics["voxel_contamination_any"]),
                "voxel_static_support_retention": float(metrics["voxel_static_support_retention"]),
            }
        )
    rows.sort(key=lambda item: (str(item["variant"]).split("_full", 1)[-1], float(item["threshold"])))
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    feasible = [row for row in rows if float(row["point_contamination"]) <= float(contamination_target)]
    best = max(feasible, key=lambda row: float(row["point_static_retention"])) if feasible else None
    return {
        "num_threshold_points": len(rows),
        "contamination_target": float(contamination_target),
        "retention_at_contamination_le_target": float(best["point_static_retention"]) if best else None,
        "threshold_at_contamination_le_target": float(best["threshold"]) if best else None,
        "variant_at_contamination_le_target": str(best["variant"]) if best else None,
        "threshold_curve_csv": str(path.resolve()) if rows else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=None)
    parser.add_argument("--manifest-list", action="append", default=None)
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--grid-stride", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--dynamic-threshold", type=float, default=0.79)
    parser.add_argument("--particle-threshold", type=float, default=0.83)
    parser.add_argument("--static-thresholds", default="0.11,0.55")
    parser.add_argument("--static-score-modes", default="full")
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--visibility-threshold", type=float, default=0.0)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--include-rgb-prefilter", action="store_true")
    parser.add_argument(
        "--external-transient-mask",
        action="append",
        default=None,
        metavar="NAME=DIR",
        help="Reuse cached transient masks saved as DIR/<clip>.npz:pred_mask and add NAME_static as a baseline.",
    )
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    start_time = time.perf_counter()
    manifests = _resolve_manifests(args)
    static_thresholds = [float(part) for part in str(args.static_thresholds).split(",") if part.strip()]
    static_score_modes = [part.strip() for part in str(args.static_score_modes).split(",") if part.strip()]
    external_masks = _parse_external_masks(args.external_transient_mask)

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cfg = load_yaml_config(args.model_config)
    image_hw = tuple(int(v) for v in cfg["model"]["input"].get("image_size", [256, 256]))
    model = _load_model(Path(args.model_config), Path(args.ckpt_path), device=device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_clip: list[dict[str, Any]] = []
    for idx, manifest_path in enumerate(manifests):
        print(f"[{idx + 1}/{len(manifests)}] {manifest_path}")
        per_clip.append(
            _evaluate_manifest(
                manifest_path=manifest_path,
                model=model,
                model_config=Path(args.model_config),
                image_hw=image_hw,
                device=device,
                max_frames=int(args.max_frames),
                grid_stride=int(args.grid_stride),
                query_chunk_size=int(args.query_chunk_size),
                dynamic_threshold=float(args.dynamic_threshold),
                particle_threshold=float(args.particle_threshold),
                static_thresholds=static_thresholds,
                static_score_modes=static_score_modes,
                confidence_threshold=float(args.confidence_threshold),
                visibility_threshold=float(args.visibility_threshold),
                voxel_size=float(args.voxel_size),
                include_rgb_prefilter=bool(args.include_rgb_prefilter),
                include_oracle=not bool(args.no_oracle),
                external_masks=external_masks,
            )
        )

    aggregate = _aggregate(per_clip)
    curve_summary = _write_threshold_curve(output_dir / "threshold_curve.csv", aggregate)
    wall_seconds = float(time.perf_counter() - start_time)
    total_frames = int(sum(int(item["num_frames"]) for item in per_clip))
    peak_vram_gb = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0**3))
        if device.type == "cuda" and torch.cuda.is_available()
        else 0.0
    )
    metadata = {
        "model_config": str(Path(args.model_config).resolve()),
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "num_manifests": len(manifests),
        "grid_stride": int(args.grid_stride),
        "voxel_size": float(args.voxel_size),
        "dynamic_threshold": float(args.dynamic_threshold),
        "particle_threshold": float(args.particle_threshold),
        "static_thresholds": static_thresholds,
        "static_score_modes": static_score_modes,
        "confidence_threshold": float(args.confidence_threshold),
        "visibility_threshold": float(args.visibility_threshold),
        "external_transient_masks": {name: str(path.resolve()) for name, path in external_masks.items()},
        "threshold_curve": curve_summary,
        "runtime": {
            "wall_seconds": wall_seconds,
            "clips_per_second": float(len(manifests)) / float(max(wall_seconds, 1e-9)),
            "frames_per_second": float(total_frames) / float(max(wall_seconds, 1e-9)),
            "peak_vram_gb": peak_vram_gb,
        },
        "interpretation_note": (
            "Map metrics are computed on D4RT query-level xyz predictions, not on a full SLAM-fused global map. "
            "WebUOT labels are bbox-level tracked-target masks and may under-label other fish."
        ),
    }
    (output_dir / "per_clip_metrics.json").write_text(json.dumps(per_clip, indent=2), encoding="utf-8")
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps({"metadata": metadata, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )
    _write_summary_csv(output_dir / "summary_table.csv", aggregate)

    print("Static point/map contamination summary:")
    for variant, metrics in aggregate["variants"].items():
        print(
            f"- {variant}: kept={metrics['kept_points']} "
            f"point_contam={metrics['point_contamination']:.4f} "
            f"point_ret={metrics['point_static_retention']:.4f} "
            f"voxel_contam={metrics['voxel_contamination_any']:.4f} "
            f"voxel_ret={metrics['voxel_static_support_retention']:.4f}"
        )
    print(f"Saved: {output_dir / 'aggregate_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
