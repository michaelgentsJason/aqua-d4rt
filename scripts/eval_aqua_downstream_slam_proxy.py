#!/usr/bin/env python3
"""Evaluate lightweight downstream visual-SLAM/SfM front-end metrics.

COLMAP is the preferred downstream benchmark once installed. This script keeps
the validation runnable without COLMAP by measuring feature-map cleanliness,
pairwise feature matching, and essential-matrix RANSAC success under different
static filtering masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
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

from aqua_prefilter_utils import temporal_rgb_pseudo_mask  # noqa: E402
from aqua_retention_utils import effective_aqua_scores  # noqa: E402
from eval_aqua_transient_heads import _load_clip, _load_model, _sigmoid  # noqa: E402
from infer_track_3d import _resolve_device  # noqa: E402
from src.core import load_yaml_config, seed_everything  # noqa: E402
from src.eval.tasks import _encode_model_memory, _run_model_for_queries  # noqa: E402


def _read_manifest_list(path: str | Path) -> list[str]:
    out: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            out.append(item)
    return out


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _parse_external_masks(values: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values or []:
        for part in [part.strip() for part in str(value).split(",") if part.strip()]:
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
    paths: list[Path] = []
    seen: set[str] = set()
    for item in items:
        path = Path(item)
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            paths.append(path)
    if args.max_clips > 0:
        paths = paths[: int(args.max_clips)]
    if not paths:
        raise ValueError("Provide --manifest or --manifest-list.")
    return paths


def _resize_mask_stack(mask: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = image_hw
    if mask.shape[1:3] == (h, w):
        return mask.astype(bool)
    return np.stack(
        [
            cv2.resize(mask[t].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            for t in range(mask.shape[0])
        ],
        axis=0,
    )


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


def _sample_mask(mask_t_hw: np.ndarray, points_xy: np.ndarray, frame_idx: int) -> np.ndarray:
    if points_xy.size == 0:
        return np.zeros((0,), dtype=bool)
    h, w = mask_t_hw.shape[1:3]
    xy = np.rint(points_xy).astype(np.int64)
    x = np.clip(xy[:, 0], 0, w - 1)
    y = np.clip(xy[:, 1], 0, h - 1)
    t = int(np.clip(frame_idx, 0, mask_t_hw.shape[0] - 1))
    return mask_t_hw[t, y, x].astype(bool)


def _load_pseudo_mask(manifest: dict[str, Any], video: np.ndarray) -> tuple[np.ndarray, str]:
    pseudo_path = manifest.get("pseudo_mask_npz")
    if pseudo_path and Path(str(pseudo_path)).exists():
        payload = np.load(str(pseudo_path))
        for key in ("pseudo_dynamic_mask", "dynamic_object_mask", "transient_mask"):
            if key in payload:
                return payload[key].astype(bool)[: video.shape[0]], str(pseudo_path)
    return temporal_rgb_pseudo_mask(video).astype(bool), "computed_temporal_rgb"


def _aqua_dense_transient_masks(
    *,
    model: torch.nn.Module,
    video: np.ndarray,
    manifest: dict[str, Any],
    device: torch.device,
    grid_stride: int,
    query_chunk_size: int,
    dynamic_threshold: float,
    particle_threshold: float,
    static_thresholds: list[float],
    static_score_mode: str = "full",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    t, h, w = video.shape[:3]
    step = max(1, int(grid_stride))
    xs = np.arange(0, w, step, dtype=np.int64)
    ys = np.arange(0, h, step, dtype=np.int64)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    per_frame_xy = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)
    coords: list[np.ndarray] = []
    for frame_idx in range(t):
        tc = np.full((per_frame_xy.shape[0], 1), frame_idx, dtype=np.int64)
        coords.append(np.concatenate([tc, per_frame_xy], axis=1))
    coord_txy = np.concatenate(coords, axis=0)

    u = coord_txy[:, 1].astype(np.float32) / float(max(w - 1, 1))
    v = coord_txy[:, 2].astype(np.float32) / float(max(h - 1, 1))
    query_cpu = {
        "u": torch.from_numpy(u),
        "v": torch.from_numpy(v),
        "t_src": torch.from_numpy(coord_txy[:, 0]).long(),
        "t_tgt": torch.from_numpy(coord_txy[:, 0]).long(),
        "t_cam": torch.from_numpy(coord_txy[:, 0]).long(),
    }

    video_b = torch.from_numpy(video).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
    aspect = torch.tensor(
        [[float(manifest.get("width", w)) / float(max(1, manifest.get("height", h)))]],
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
        raw_static_probs = pred["static_confidence"].numpy().astype(np.float32)
    else:
        raw_static_probs = confidence_probs * (1.0 - dynamic_probs) * (1.0 - particle_probs)
    effective = effective_aqua_scores(
        dynamic_probs=dynamic_probs,
        particle_probs=particle_probs,
        confidence_probs=confidence_probs,
        static_probs=raw_static_probs,
        static_score_mode=static_score_mode,
    )
    dynamic_probs_eff = effective["dynamic_prob"]
    particle_probs_eff = effective["particle_prob"]
    static_probs = effective["static_confidence"]
    pred_transient = (dynamic_probs_eff >= float(dynamic_threshold)) | (particle_probs_eff >= float(particle_threshold))
    suffix = "" if str(static_score_mode) == "full" else f"_{static_score_mode}"
    transient_variants: dict[str, np.ndarray] = {f"aqua_pred_transient_filter{suffix}": pred_transient}
    for threshold in static_thresholds:
        name = f"aqua_static_conf_ge_{threshold:.3f}{suffix}".replace(".", "p")
        transient_variants[name] = pred_transient | (static_probs < float(threshold))

    coarse_variants = {name: np.zeros((t, len(ys), len(xs)), dtype=np.uint8) for name in transient_variants}
    per_frame = int(per_frame_xy.shape[0])
    for frame_idx in range(t):
        start = frame_idx * per_frame
        for name, transient in transient_variants.items():
            coarse_variants[name][frame_idx] = transient[start : start + per_frame].reshape(len(ys), len(xs)).astype(np.uint8)

    kernel = np.ones((max(1, step), max(1, step)), dtype=np.uint8)
    dense_variants: dict[str, np.ndarray] = {}
    for name, coarse in coarse_variants.items():
        dense_frames: list[np.ndarray] = []
        for frame_idx in range(t):
            dense = cv2.resize(coarse[frame_idx], (w, h), interpolation=cv2.INTER_NEAREST)
            dense = cv2.dilate(dense, kernel, iterations=1)
            dense_frames.append(dense.astype(bool))
        dense_variants[name] = np.stack(dense_frames, axis=0)
    meta = {
        "grid_stride": int(grid_stride),
        "static_score_mode": str(static_score_mode),
        "dynamic_threshold": float(dynamic_threshold),
        "particle_threshold": float(particle_threshold),
        "static_thresholds": [float(v) for v in static_thresholds],
        "dynamic_prob_mean": float(np.mean(dynamic_probs_eff)),
        "particle_prob_mean": float(np.mean(particle_probs_eff)),
        "static_confidence_mean": float(np.mean(static_probs)),
        "raw_dynamic_prob_mean": float(np.mean(dynamic_probs)),
        "raw_particle_prob_mean": float(np.mean(particle_probs)),
        "raw_static_confidence_mean": float(np.mean(raw_static_probs)),
    }
    return dense_variants, meta


def _make_detector(name: str, max_features: int) -> tuple[Any, int]:
    detector_name = str(name).lower()
    if detector_name == "sift" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=int(max_features)), cv2.NORM_L2
    if detector_name != "orb":
        print(f"Detector {name!r} unavailable; falling back to ORB.")
    return cv2.ORB_create(nfeatures=int(max_features), fastThreshold=12), cv2.NORM_HAMMING


def _detect_features(
    frame_rgb: np.ndarray,
    static_mask: np.ndarray,
    detector: Any,
    max_features: int,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    mask_u8 = static_mask.astype(np.uint8) * 255
    keypoints, descriptors = detector.detectAndCompute(gray, mask_u8)
    if keypoints is None:
        keypoints = []
    if len(keypoints) > int(max_features):
        order = np.argsort([-kp.response for kp in keypoints])[: int(max_features)]
        keypoints = [keypoints[int(i)] for i in order]
        descriptors = descriptors[order] if descriptors is not None else None
    return keypoints, descriptors


def _match_descriptors(desc0: np.ndarray | None, desc1: np.ndarray | None, norm: int, ratio: float) -> list[cv2.DMatch]:
    if desc0 is None or desc1 is None or len(desc0) < 2 or len(desc1) < 2:
        return []
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    raw = matcher.knnMatch(desc0, desc1, k=2)
    good: list[cv2.DMatch] = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance <= float(ratio) * n.distance:
            good.append(m)
    return good


def _slam_aware_retention_mask(
    *,
    video_rgb: np.ndarray,
    rejected_mask: np.ndarray,
    detector_name: str,
    max_features: int,
    ratio: float,
    frame_step: int,
    patch_radius: int,
    min_inlier_support: int,
    max_retained_features_per_frame: int,
    max_retained_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover small patches around geometrically stable features rejected by Aqua.

    This is an evaluation-time SLAM-aware retention baseline. It does not use
    transient ground truth. A feature can be retained only if it lies in the
    rejected Aqua mask and participates in adjacent-frame essential-matrix
    inlier matches.
    """

    if video_rgb.ndim != 4 or rejected_mask.shape != video_rgb.shape[:3]:
        raise ValueError(f"shape mismatch: video={video_rgb.shape}, rejected={rejected_mask.shape}")

    detector, norm = _make_detector(detector_name, max_features)
    t, h, w = video_rgb.shape[:3]
    frame_ids = list(range(0, t, max(1, int(frame_step))))
    if frame_ids[-1] != t - 1:
        frame_ids.append(t - 1)

    keypoints_by_frame: dict[int, list[cv2.KeyPoint]] = {}
    descriptors_by_frame: dict[int, np.ndarray | None] = {}
    support_by_frame: dict[int, np.ndarray] = {}
    for frame_idx in frame_ids:
        keypoints, descriptors = _detect_features(
            frame_rgb=video_rgb[frame_idx],
            static_mask=np.ones((h, w), dtype=bool),
            detector=detector,
            max_features=int(max_features),
        )
        keypoints_by_frame[frame_idx] = keypoints
        descriptors_by_frame[frame_idx] = descriptors
        support_by_frame[frame_idx] = np.zeros((len(keypoints),), dtype=np.int32)

    focal = float(max(h, w))
    camera_matrix = np.asarray([[focal, 0.0, 0.5 * (w - 1)], [0.0, focal, 0.5 * (h - 1)], [0.0, 0.0, 1.0]])
    num_pairs = 0
    num_pairs_with_model = 0
    num_inlier_matches = 0
    for f0, f1 in zip(frame_ids[:-1], frame_ids[1:]):
        matches = _match_descriptors(descriptors_by_frame[f0], descriptors_by_frame[f1], norm, ratio)
        num_pairs += 1
        if len(matches) < 8:
            continue
        pts0 = np.asarray([keypoints_by_frame[f0][m.queryIdx].pt for m in matches], dtype=np.float32)
        pts1 = np.asarray([keypoints_by_frame[f1][m.trainIdx].pt for m in matches], dtype=np.float32)
        try:
            _, inlier_mask = cv2.findEssentialMat(
                pts0,
                pts1,
                cameraMatrix=camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0,
            )
        except cv2.error:
            inlier_mask = None
        if inlier_mask is None:
            continue
        inliers = inlier_mask.reshape(-1).astype(bool)
        if int(inliers.sum()) < 8:
            continue
        num_pairs_with_model += 1
        num_inlier_matches += int(inliers.sum())
        for match, is_inlier in zip(matches, inliers):
            if not bool(is_inlier):
                continue
            support_by_frame[f0][match.queryIdx] += 1
            support_by_frame[f1][match.trainIdx] += 1

    retain = np.zeros(rejected_mask.shape, dtype=bool)
    radius = max(0, int(patch_radius))
    max_features_frame = max(0, int(max_retained_features_per_frame))
    max_fraction = float(np.clip(max_retained_fraction, 0.0, 1.0))
    retained_features_total = 0
    candidate_features_total = 0
    for frame_idx in frame_ids:
        candidates: list[tuple[int, float, int, int]] = []
        for kp_idx, kp in enumerate(keypoints_by_frame[frame_idx]):
            support = int(support_by_frame[frame_idx][kp_idx])
            if support < int(min_inlier_support):
                continue
            x = int(np.clip(round(kp.pt[0]), 0, w - 1))
            y = int(np.clip(round(kp.pt[1]), 0, h - 1))
            if not bool(rejected_mask[frame_idx, y, x]):
                continue
            score = float(support) * max(float(kp.response), 1e-6)
            candidates.append((support, score, x, y))
        candidate_features_total += len(candidates)
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if max_features_frame > 0:
            candidates = candidates[:max_features_frame]

        frame_retain = np.zeros((h, w), dtype=np.uint8)
        kept = 0
        for _, _, x, y in candidates:
            if radius <= 0:
                frame_retain[y, x] = 1
            else:
                cv2.circle(frame_retain, (x, y), radius, 1, thickness=-1)
            kept += 1
            if max_fraction > 0.0 and float(frame_retain.mean()) >= max_fraction:
                break
        retained_features_total += kept
        retain[frame_idx] = (frame_retain.astype(bool) & rejected_mask[frame_idx])

    meta = {
        "detector": str(detector_name),
        "frame_step": int(frame_step),
        "patch_radius": int(patch_radius),
        "min_inlier_support": int(min_inlier_support),
        "max_retained_features_per_frame": int(max_retained_features_per_frame),
        "max_retained_fraction": float(max_retained_fraction),
        "num_frames": int(t),
        "num_eval_frames": len(frame_ids),
        "num_pairs": int(num_pairs),
        "num_pairs_with_model": int(num_pairs_with_model),
        "num_inlier_matches": int(num_inlier_matches),
        "candidate_features_total": int(candidate_features_total),
        "retained_features_total": int(retained_features_total),
        "retained_pixel_fraction": float(retain.mean()) if retain.size else 0.0,
    }
    return retain, meta


def _pair_metrics(
    *,
    keypoints0: list[cv2.KeyPoint],
    descriptors0: np.ndarray | None,
    keypoints1: list[cv2.KeyPoint],
    descriptors1: np.ndarray | None,
    transient_mask: np.ndarray,
    frame0: int,
    frame1: int,
    norm: int,
    ratio: float,
    camera_matrix: np.ndarray,
) -> dict[str, Any]:
    matches = _match_descriptors(descriptors0, descriptors1, norm, ratio)
    n_matches = len(matches)
    if n_matches == 0:
        return {
            "frame0": int(frame0),
            "frame1": int(frame1),
            "matches": 0,
            "match_contamination": 0.0,
            "essential_success": False,
            "essential_inliers": 0,
            "essential_inlier_rate": 0.0,
        }

    pts0 = np.asarray([keypoints0[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts1 = np.asarray([keypoints1[m.trainIdx].pt for m in matches], dtype=np.float32)
    contam0 = _sample_mask(transient_mask, pts0, frame0)
    contam1 = _sample_mask(transient_mask, pts1, frame1)
    contam = contam0 | contam1

    essential_success = False
    inliers = 0
    inlier_rate = 0.0
    if n_matches >= 8:
        try:
            _, mask = cv2.findEssentialMat(
                pts0,
                pts1,
                cameraMatrix=camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0,
            )
            if mask is not None:
                mask_b = mask.reshape(-1).astype(bool)
                inliers = int(mask_b.sum())
                inlier_rate = float(inliers) / float(max(1, n_matches))
                essential_success = inliers >= 8
        except cv2.error:
            essential_success = False

    return {
        "frame0": int(frame0),
        "frame1": int(frame1),
        "matches": int(n_matches),
        "match_contamination": float(contam.mean()) if contam.size else 0.0,
        "essential_success": bool(essential_success),
        "essential_inliers": int(inliers),
        "essential_inlier_rate": float(inlier_rate),
    }


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _summarize_variant(frame_metrics: list[dict[str, Any]], pair_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    total_features = int(sum(int(item["num_features"]) for item in frame_metrics))
    contaminated = int(sum(int(item["contaminated_features"]) for item in frame_metrics))
    total_matches = int(sum(int(item["matches"]) for item in pair_metrics))
    successes = int(sum(1 for item in pair_metrics if bool(item["essential_success"])))
    return {
        "frames": len(frame_metrics),
        "pairs": len(pair_metrics),
        "total_features": total_features,
        "features_per_frame_mean": _safe_mean([float(item["num_features"]) for item in frame_metrics]),
        "feature_contamination": float(contaminated) / float(max(1, total_features)),
        "total_matches": total_matches,
        "matches_per_pair_mean": _safe_mean([float(item["matches"]) for item in pair_metrics]),
        "match_contamination_mean": _safe_mean([float(item["match_contamination"]) for item in pair_metrics]),
        "essential_success_rate": float(successes) / float(max(1, len(pair_metrics))),
        "essential_inliers_per_pair_mean": _safe_mean([float(item["essential_inliers"]) for item in pair_metrics]),
        "essential_inlier_rate_mean": _safe_mean([float(item["essential_inlier_rate"]) for item in pair_metrics]),
    }


def _evaluate_manifest(
    *,
    manifest_path: Path,
    model: torch.nn.Module,
    image_hw: tuple[int, int],
    device: torch.device,
    max_frames: int,
    frame_step: int,
    aqua_grid_stride: int,
    query_chunk_size: int,
    dynamic_threshold: float,
    particle_threshold: float,
    static_thresholds: list[float],
    static_score_modes: list[str],
    detector_name: str,
    max_features: int,
    ratio: float,
    include_oracle: bool,
    include_temporal_rgb: bool,
    enable_slam_aware_retention: bool,
    retention_patch_radius: int,
    retention_min_inlier_support: int,
    retention_max_features_per_frame: int,
    retention_max_fraction: float,
    external_masks: dict[str, Path],
) -> dict[str, Any]:
    video, dynamic_mask, particle_mask, manifest = _load_clip(
        manifest_path,
        image_hw=image_hw,
        max_frames=int(max_frames),
    )
    transient_gt = (dynamic_mask | particle_mask).astype(bool)
    aqua_masks: dict[str, np.ndarray] = {}
    aqua_meta: dict[str, Any] = {}
    for mode in static_score_modes:
        mode_masks, mode_meta = _aqua_dense_transient_masks(
            model=model,
            video=video,
            manifest=manifest,
            device=device,
            grid_stride=int(aqua_grid_stride),
            query_chunk_size=int(query_chunk_size),
            dynamic_threshold=float(dynamic_threshold),
            particle_threshold=float(particle_threshold),
            static_thresholds=static_thresholds,
            static_score_mode=str(mode),
        )
        aqua_masks.update(mode_masks)
        aqua_meta[str(mode)] = mode_meta
    variants: dict[str, np.ndarray] = {
        "raw_all_pixels": np.ones(transient_gt.shape, dtype=bool),
    }
    for name, mask in aqua_masks.items():
        variants[name] = ~mask
    retention_meta: dict[str, Any] = {}
    if bool(enable_slam_aware_retention):
        for name, mask in aqua_masks.items():
            retain_mask, meta = _slam_aware_retention_mask(
                video_rgb=video,
                rejected_mask=mask,
                detector_name=detector_name,
                max_features=int(max_features),
                ratio=float(ratio),
                frame_step=int(frame_step),
                patch_radius=int(retention_patch_radius),
                min_inlier_support=int(retention_min_inlier_support),
                max_retained_features_per_frame=int(retention_max_features_per_frame),
                max_retained_fraction=float(retention_max_fraction),
            )
            retained_name = f"{name}_slam_retain"
            variants[retained_name] = (~mask) | retain_mask
            retention_meta[retained_name] = meta
    pseudo_source = None
    if include_oracle:
        variants["oracle_gt_static"] = ~transient_gt
    if include_temporal_rgb:
        pseudo_mask, pseudo_source = _load_pseudo_mask(manifest, video)
        pseudo_mask = _resize_mask_stack(pseudo_mask[: video.shape[0]], image_hw)
        variants["temporal_rgb_static"] = ~pseudo_mask
    external_mask_sources: dict[str, str] = {}
    for name, mask_dir in external_masks.items():
        external_mask, source_path = _load_external_pred_mask(
            mask_dir=mask_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            image_hw=image_hw,
            num_frames=int(video.shape[0]),
        )
        variants[f"{name}_static"] = ~external_mask
        external_mask_sources[name] = source_path

    detector, norm = _make_detector(detector_name, max_features)
    h, w = video.shape[1:3]
    focal = float(max(h, w))
    camera_matrix = np.asarray([[focal, 0.0, 0.5 * (w - 1)], [0.0, focal, 0.5 * (h - 1)], [0.0, 0.0, 1.0]])
    frame_ids = list(range(0, video.shape[0], max(1, int(frame_step))))
    if frame_ids[-1] != video.shape[0] - 1:
        frame_ids.append(video.shape[0] - 1)

    result_variants: dict[str, Any] = {}
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

        result_variants[name] = {
            "summary": _summarize_variant(frame_metrics, pair_metrics),
            "frame_metrics": frame_metrics,
            "pair_metrics": pair_metrics,
        }

    return {
        "manifest": str(manifest_path.resolve()),
        "clip_name": str(manifest.get("name", manifest_path.parent.name)),
        "dataset": str(manifest.get("dataset", "unknown")),
        "num_frames": int(video.shape[0]),
        "image_hw": [int(h), int(w)],
        "frame_ids": frame_ids,
        "mask_coverage": {
            "dynamic_object": float(dynamic_mask.mean()),
            "particle": float(particle_mask.mean()),
            "transient": float(transient_gt.mean()),
        },
        "aqua_mask_meta": aqua_meta,
        "slam_aware_retention_meta": retention_meta,
        "pseudo_mask_source": pseudo_source,
        "external_mask_sources": external_mask_sources,
        "variants": result_variants,
    }


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
        }
    out["mean_mask_coverage"] = {
        key: _safe_mean([float(clip["mask_coverage"][key]) for clip in per_clip])
        for key in ("dynamic_object", "particle", "transient")
    }
    return out


def _write_summary_csv(path: Path, aggregate: dict[str, Any]) -> None:
    rows = []
    for variant, metrics in aggregate["variants"].items():
        rows.append(
            {
                "variant": variant,
                "total_features": metrics["total_features"],
                "features_per_frame_mean": metrics["features_per_frame_mean"],
                "feature_contamination": metrics["feature_contamination"],
                "total_matches": metrics["total_matches"],
                "matches_per_pair_mean": metrics["matches_per_pair_mean"],
                "match_contamination_mean": metrics["match_contamination_mean"],
                "essential_success_rate": metrics["essential_success_rate"],
                "essential_inliers_per_pair_mean": metrics["essential_inliers_per_pair_mean"],
                "essential_inlier_rate_mean": metrics["essential_inlier_rate_mean"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["variant"])
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--aqua-grid-stride", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--dynamic-threshold", type=float, default=0.79)
    parser.add_argument("--particle-threshold", type=float, default=0.83)
    parser.add_argument("--static-thresholds", default="0.11,0.55")
    parser.add_argument("--static-score-modes", default="full")
    parser.add_argument("--detector", default="orb", choices=("orb", "sift"))
    parser.add_argument("--max-features", type=int, default=1200)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--no-temporal-rgb", action="store_true")
    parser.add_argument(
        "--external-transient-mask",
        action="append",
        default=None,
        metavar="NAME=DIR",
        help="Reuse cached transient masks saved as DIR/<clip>.npz:pred_mask and add NAME_static as a baseline.",
    )
    parser.add_argument("--enable-slam-aware-retention", action="store_true")
    parser.add_argument("--retention-patch-radius", type=int, default=5)
    parser.add_argument("--retention-min-inlier-support", type=int, default=1)
    parser.add_argument("--retention-max-features-per-frame", type=int, default=300)
    parser.add_argument("--retention-max-fraction", type=float, default=0.18)
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
                image_hw=image_hw,
                device=device,
                max_frames=int(args.max_frames),
                frame_step=int(args.frame_step),
                aqua_grid_stride=int(args.aqua_grid_stride),
                query_chunk_size=int(args.query_chunk_size),
                dynamic_threshold=float(args.dynamic_threshold),
                particle_threshold=float(args.particle_threshold),
                static_thresholds=static_thresholds,
                static_score_modes=static_score_modes,
                detector_name=str(args.detector),
                max_features=int(args.max_features),
                ratio=float(args.ratio),
                include_oracle=not bool(args.no_oracle),
                include_temporal_rgb=not bool(args.no_temporal_rgb),
                enable_slam_aware_retention=bool(args.enable_slam_aware_retention),
                retention_patch_radius=int(args.retention_patch_radius),
                retention_min_inlier_support=int(args.retention_min_inlier_support),
                retention_max_features_per_frame=int(args.retention_max_features_per_frame),
                retention_max_fraction=float(args.retention_max_fraction),
                external_masks=external_masks,
            )
        )

    aggregate = _aggregate(per_clip)
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
        "detector": str(args.detector),
        "max_features": int(args.max_features),
        "ratio": float(args.ratio),
        "frame_step": int(args.frame_step),
        "aqua_grid_stride": int(args.aqua_grid_stride),
        "dynamic_threshold": float(args.dynamic_threshold),
        "particle_threshold": float(args.particle_threshold),
        "static_thresholds": static_thresholds,
        "static_score_modes": static_score_modes,
        "external_transient_masks": {name: str(path.resolve()) for name, path in external_masks.items()},
        "runtime": {
            "wall_seconds": wall_seconds,
            "clips_per_second": float(len(manifests)) / float(max(wall_seconds, 1e-9)),
            "frames_per_second": float(total_frames) / float(max(wall_seconds, 1e-9)),
            "peak_vram_gb": peak_vram_gb,
        },
        "colmap_available": shutil.which("colmap") is not None,
        "pycolmap_available": False,
        "slam_aware_retention": {
            "enabled": bool(args.enable_slam_aware_retention),
            "patch_radius": int(args.retention_patch_radius),
            "min_inlier_support": int(args.retention_min_inlier_support),
            "max_features_per_frame": int(args.retention_max_features_per_frame),
            "max_fraction": float(args.retention_max_fraction),
        },
        "interpretation_note": (
            "These are front-end proxy metrics for SfM/visual SLAM: feature contamination, pairwise matches, "
            "and essential-matrix RANSAC success. They are not a replacement for a full COLMAP/ORB-SLAM3 run."
        ),
    }
    try:
        import pycolmap  # noqa: F401

        metadata["pycolmap_available"] = True
    except Exception:
        metadata["pycolmap_available"] = False

    (output_dir / "per_clip_metrics.json").write_text(json.dumps(per_clip, indent=2), encoding="utf-8")
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps({"metadata": metadata, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )
    _write_summary_csv(output_dir / "summary_table.csv", aggregate)

    print("Downstream SLAM/SfM proxy summary:")
    for variant, metrics in aggregate["variants"].items():
        print(
            f"- {variant}: feat/frame={metrics['features_per_frame_mean']:.1f} "
            f"feat_contam={metrics['feature_contamination']:.4f} "
            f"match/pair={metrics['matches_per_pair_mean']:.1f} "
            f"match_contam={metrics['match_contamination_mean']:.4f} "
            f"E_success={metrics['essential_success_rate']:.3f}"
        )
    print(f"COLMAP available: {metadata['colmap_available']}")
    print(f"Saved: {output_dir / 'aggregate_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
