#!/usr/bin/env python3
"""Utilities for Aqua-D4RT SLAM-aware retention experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from eval_aqua_transient_heads import _sigmoid  # noqa: E402
from src.eval.tasks import _encode_model_memory, _run_model_for_queries  # noqa: E402


FEATURE_NAMES = [
    "aqua_dynamic_prob",
    "aqua_particle_prob",
    "aqua_static_confidence",
    "aqua_confidence_prob",
    "aqua_transient_prob_max",
    "aqua_rejection_margin",
    "kp_response",
    "kp_size",
    "kp_octave",
    "kp_response_rank",
    "match_support",
    "inlier_support",
    "inlier_ratio",
    "mean_match_distance_norm",
    "mean_inlier_distance_norm",
    "mean_inlier_flow_norm",
    "mean_inlier_patch_ncc",
]

POSE_RETENTION_FEATURE_NAMES = FEATURE_NAMES + [
    "gt_pose_match_support",
    "gt_pose_inlier_support",
    "gt_pose_inlier_ratio",
    "gt_pose_sampson_mean_log1p_px",
    "gt_pose_sampson_min_log1p_px",
    "gt_pose_consistency_mean",
]

RETENTION_META_NAMES = [
    "clip_index",
    "frame",
    "x",
    "y",
    "gt_transient",
    "geometry_stable",
    "match_support",
    "inlier_support",
    "kp_response",
]

POSE_RETENTION_META_NAMES = RETENTION_META_NAMES + [
    "gt_pose_stable",
    "gt_pose_match_support",
    "gt_pose_inlier_support",
    "gt_pose_sampson_mean_px",
    "gt_pose_sampson_min_px",
    "gt_pose_consistency_mean",
]

STATIC_SCORE_MODES = ("full", "no_dynamic", "no_particle", "confidence_only")


def read_manifest_list(path: str | Path) -> list[str]:
    items: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            items.append(item)
    return items


def resolve_manifests(
    *,
    manifest: list[str] | None,
    manifest_list: list[str] | None,
    max_clips: int = 0,
) -> list[Path]:
    items: list[str] = []
    if manifest:
        for value in manifest:
            items.extend(part.strip() for part in str(value).split(",") if part.strip())
    if manifest_list:
        for value in manifest_list:
            items.extend(read_manifest_list(value))
    out: list[Path] = []
    seen: set[str] = set()
    for item in items:
        path = Path(item)
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            out.append(path)
    if max_clips > 0:
        out = out[: int(max_clips)]
    if not out:
        raise ValueError("Provide --manifest or --manifest-list.")
    return out


def frame_ids_for(num_frames: int, frame_step: int) -> list[int]:
    if num_frames <= 0:
        return []
    frame_ids = list(range(0, int(num_frames), max(1, int(frame_step))))
    if frame_ids[-1] != int(num_frames) - 1:
        frame_ids.append(int(num_frames) - 1)
    return frame_ids


def _dense_from_coarse(coarse_tyx: np.ndarray, image_hw: tuple[int, int], interpolation: int) -> np.ndarray:
    h, w = image_hw
    dense = [
        cv2.resize(coarse_tyx[t].astype(np.float32), (w, h), interpolation=interpolation)
        for t in range(coarse_tyx.shape[0])
    ]
    return np.stack(dense, axis=0).astype(np.float32)


def effective_aqua_scores(
    *,
    dynamic_probs: np.ndarray,
    particle_probs: np.ndarray,
    confidence_probs: np.ndarray,
    static_probs: np.ndarray | None = None,
    static_score_mode: str = "full",
) -> dict[str, np.ndarray]:
    """Apply component ablations to Aqua transient/static scores.

    The ablations are inference-time diagnostics, not retrained models:
    `no_dynamic` removes dynamic-object probabilities from the static product
    and transient mask; `no_particle` removes particle probabilities; and
    `confidence_only` falls back to base D4RT confidence.
    """

    mode = str(static_score_mode).strip().lower()
    if mode not in STATIC_SCORE_MODES:
        raise ValueError(f"Unsupported static_score_mode={static_score_mode!r}; expected one of {STATIC_SCORE_MODES}")
    dyn = np.asarray(dynamic_probs, dtype=np.float32)
    particle = np.asarray(particle_probs, dtype=np.float32)
    conf = np.asarray(confidence_probs, dtype=np.float32)
    if mode == "full":
        eff_dyn = dyn
        eff_particle = particle
        eff_static = np.asarray(static_probs, dtype=np.float32) if static_probs is not None else conf * (1.0 - dyn) * (1.0 - particle)
    elif mode == "no_dynamic":
        eff_dyn = np.zeros_like(dyn, dtype=np.float32)
        eff_particle = particle
        eff_static = conf * (1.0 - eff_particle)
    elif mode == "no_particle":
        eff_dyn = dyn
        eff_particle = np.zeros_like(particle, dtype=np.float32)
        eff_static = conf * (1.0 - eff_dyn)
    else:
        eff_dyn = np.zeros_like(dyn, dtype=np.float32)
        eff_particle = np.zeros_like(particle, dtype=np.float32)
        eff_static = conf
    return {
        "dynamic_prob": np.clip(eff_dyn, 0.0, 1.0).astype(np.float32),
        "particle_prob": np.clip(eff_particle, 0.0, 1.0).astype(np.float32),
        "confidence_prob": np.clip(conf, 0.0, 1.0).astype(np.float32),
        "static_confidence": np.clip(eff_static, 0.0, 1.0).astype(np.float32),
    }


def aqua_dense_score_maps(
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
) -> dict[str, Any]:
    """Run Aqua-D4RT on a dense query grid and return image-sized score maps.

    Binary rejected masks intentionally match the downstream proxy convention:
    threshold on the coarse query grid, nearest-neighbor upsample, then dilate by
    one grid cell. Probability maps are bilinearly upsampled for keypoint-level
    feature sampling.
    """

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

    per_frame = int(per_frame_xy.shape[0])
    coarse_shape = (t, len(ys), len(xs))
    dynamic_coarse = dynamic_probs_eff.reshape(coarse_shape)
    particle_coarse = particle_probs_eff.reshape(coarse_shape)
    confidence_coarse = confidence_probs.reshape(coarse_shape)
    static_coarse = static_probs.reshape(coarse_shape)

    score_maps = {
        "dynamic_prob": np.clip(_dense_from_coarse(dynamic_coarse, (h, w), cv2.INTER_LINEAR), 0.0, 1.0),
        "particle_prob": np.clip(_dense_from_coarse(particle_coarse, (h, w), cv2.INTER_LINEAR), 0.0, 1.0),
        "confidence_prob": np.clip(_dense_from_coarse(confidence_coarse, (h, w), cv2.INTER_LINEAR), 0.0, 1.0),
        "static_confidence": np.clip(_dense_from_coarse(static_coarse, (h, w), cv2.INTER_LINEAR), 0.0, 1.0),
    }

    coarse_transient = (dynamic_coarse >= float(dynamic_threshold)) | (particle_coarse >= float(particle_threshold))
    kernel = np.ones((step, step), dtype=np.uint8)
    rejected_masks: dict[str, np.ndarray] = {}
    transient_mask_frames: list[np.ndarray] = []
    for frame_idx in range(t):
        dense = cv2.resize(coarse_transient[frame_idx].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        dense = cv2.dilate(dense, kernel, iterations=1).astype(bool)
        transient_mask_frames.append(dense)
    rejected_masks["aqua_pred_transient_filter"] = np.stack(transient_mask_frames, axis=0)

    for threshold in static_thresholds:
        name = f"aqua_static_conf_ge_{float(threshold):.3f}".replace(".", "p")
        coarse_rejected = coarse_transient | (static_coarse < float(threshold))
        dense_frames = []
        for frame_idx in range(t):
            dense = cv2.resize(coarse_rejected[frame_idx].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            dense = cv2.dilate(dense, kernel, iterations=1).astype(bool)
            dense_frames.append(dense)
        rejected_masks[name] = np.stack(dense_frames, axis=0)

    meta = {
        "grid_stride": int(grid_stride),
        "num_queries": int(coord_txy.shape[0]),
        "queries_per_frame": int(per_frame),
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
    return {"score_maps": score_maps, "rejected_masks": rejected_masks, "meta": meta}


def make_detector(name: str, max_features: int) -> tuple[Any, int]:
    detector_name = str(name).lower()
    if detector_name == "sift" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=int(max_features)), cv2.NORM_L2
    if detector_name != "orb":
        print(f"Detector {name!r} unavailable; falling back to ORB.")
    return cv2.ORB_create(nfeatures=int(max_features), fastThreshold=12), cv2.NORM_HAMMING


def detect_features(
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


def match_descriptors(desc0: np.ndarray | None, desc1: np.ndarray | None, norm: int, ratio: float) -> list[cv2.DMatch]:
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


def _patch_ncc(gray0: np.ndarray, gray1: np.ndarray, xy0: tuple[float, float], xy1: tuple[float, float], radius: int = 3) -> float:
    r = max(1, int(radius))
    x0, y0 = int(round(xy0[0])), int(round(xy0[1]))
    x1, y1 = int(round(xy1[0])), int(round(xy1[1]))
    h, w = gray0.shape[:2]
    if x0 - r < 0 or x0 + r >= w or y0 - r < 0 or y0 + r >= h:
        return 0.0
    if x1 - r < 0 or x1 + r >= w or y1 - r < 0 or y1 + r >= h:
        return 0.0
    p0 = gray0[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1].astype(np.float32)
    p1 = gray1[y1 - r : y1 + r + 1, x1 - r : x1 + r + 1].astype(np.float32)
    p0 = p0 - float(p0.mean())
    p1 = p1 - float(p1.mean())
    den = float(np.linalg.norm(p0) * np.linalg.norm(p1))
    if den <= 1e-6:
        return 0.0
    return float(np.clip(float(np.sum(p0 * p1)) / den, -1.0, 1.0))


def _quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        q = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        q = q / n
    x, y, z, w = [float(v) for v in q]
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = [float(v) for v in np.asarray(vec, dtype=np.float64).reshape(3)]
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _pose_vec(row: dict[str, Any], key: str, scalar_keys: tuple[str, str, str] | None = None) -> np.ndarray:
    if key in row:
        return np.asarray(row[key], dtype=np.float64)
    if scalar_keys is not None and all(name in row for name in scalar_keys):
        return np.asarray([float(row[name]) for name in scalar_keys], dtype=np.float64)
    raise KeyError(key)


def _essential_candidates_from_pose_rows(row0: dict[str, Any], row1: dict[str, Any]) -> list[np.ndarray]:
    """Return plausible essential matrices from two GT pose rows.

    Tank pose files provide reliable camera centers, but the stored quaternion
    convention needs a dedicated frame audit before it can be treated as a
    paper-facing orientation metric. For v3 supervision we therefore compute
    both common rotation conventions and use the lower Sampson residual.
    """

    c0 = _pose_vec(row0, "position", ("pos_x", "pos_y", "pos_z")).reshape(3)
    c1 = _pose_vec(row1, "position", ("pos_x", "pos_y", "pos_z")).reshape(3)
    q0 = _pose_vec(row0, "quaternion_xyzw", ("orient_qx", "orient_qy", "orient_qz")).reshape(-1)
    q1 = _pose_vec(row1, "quaternion_xyzw", ("orient_qx", "orient_qy", "orient_qz")).reshape(-1)
    if q0.size == 3 and "orient_qw" in row0:
        q0 = np.concatenate([q0, np.asarray([float(row0["orient_qw"])], dtype=np.float64)])
    if q1.size == 3 and "orient_qw" in row1:
        q1 = np.concatenate([q1, np.asarray([float(row1["orient_qw"])], dtype=np.float64)])
    if q0.size != 4 or q1.size != 4:
        return []

    r0 = _quat_xyzw_to_rot(q0)
    r1 = _quat_xyzw_to_rot(q1)
    candidates: list[np.ndarray] = []

    for rel_rot, rel_t in (
        (r1.T @ r0, r1.T @ (c0 - c1)),  # camera-to-world quaternion convention
        (r1 @ r0.T, r1 @ (c0 - c1)),  # world-to-camera quaternion convention
    ):
        norm = float(np.linalg.norm(rel_t))
        if norm <= 1e-9:
            continue
        e_mat = _skew(rel_t / norm) @ rel_rot
        e_norm = float(np.linalg.norm(e_mat))
        if e_norm > 1e-12:
            candidates.append(e_mat / e_norm)
    return candidates


def _sampson_errors_px(
    pts0: np.ndarray,
    pts1: np.ndarray,
    essentials: list[np.ndarray],
    *,
    focal: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    if pts0.size == 0 or pts1.size == 0 or not essentials:
        return np.full((pts0.shape[0],), np.inf, dtype=np.float32)
    x0 = np.stack(
        [
            (pts0[:, 0].astype(np.float64) - float(cx)) / float(max(focal, 1e-6)),
            (pts0[:, 1].astype(np.float64) - float(cy)) / float(max(focal, 1e-6)),
            np.ones((pts0.shape[0],), dtype=np.float64),
        ],
        axis=1,
    )
    x1 = np.stack(
        [
            (pts1[:, 0].astype(np.float64) - float(cx)) / float(max(focal, 1e-6)),
            (pts1[:, 1].astype(np.float64) - float(cy)) / float(max(focal, 1e-6)),
            np.ones((pts1.shape[0],), dtype=np.float64),
        ],
        axis=1,
    )
    best = np.full((pts0.shape[0],), np.inf, dtype=np.float64)
    for e_mat in essentials:
        ex0 = x0 @ e_mat.T
        etx1 = x1 @ e_mat
        residual = np.sum(x1 * ex0, axis=1)
        denom = ex0[:, 0] * ex0[:, 0] + ex0[:, 1] * ex0[:, 1] + etx1[:, 0] * etx1[:, 0] + etx1[:, 1] * etx1[:, 1]
        err = (residual * residual) / np.maximum(denom, 1e-12)
        best = np.minimum(best, np.sqrt(np.maximum(err, 0.0)) * float(focal))
    return best.astype(np.float32)


def extract_keypoint_context(
    *,
    video_rgb: np.ndarray,
    detector_name: str,
    max_features: int,
    ratio: float,
    frame_step: int,
    essential_threshold: float = 1.0,
    pose_rows: list[dict[str, Any]] | None = None,
    pose_sampson_threshold_px: float = 2.0,
) -> dict[str, Any]:
    detector, norm = make_detector(detector_name, max_features)
    t, h, w = video_rgb.shape[:3]
    frame_ids = frame_ids_for(t, frame_step)
    keypoints_by_frame: dict[int, list[cv2.KeyPoint]] = {}
    descriptors_by_frame: dict[int, np.ndarray | None] = {}
    gray_by_frame: dict[int, np.ndarray] = {}
    stats_by_frame: dict[int, dict[str, np.ndarray]] = {}

    for frame_idx in frame_ids:
        keypoints, descriptors = detect_features(
            frame_rgb=video_rgb[frame_idx],
            static_mask=np.ones((h, w), dtype=bool),
            detector=detector,
            max_features=int(max_features),
        )
        keypoints_by_frame[frame_idx] = keypoints
        descriptors_by_frame[frame_idx] = descriptors
        gray_by_frame[frame_idx] = cv2.cvtColor(video_rgb[frame_idx], cv2.COLOR_RGB2GRAY)
        n = len(keypoints)
        stats_by_frame[frame_idx] = {
            "match_support": np.zeros((n,), dtype=np.float32),
            "inlier_support": np.zeros((n,), dtype=np.float32),
            "sum_match_distance": np.zeros((n,), dtype=np.float32),
            "sum_inlier_distance": np.zeros((n,), dtype=np.float32),
            "sum_inlier_flow": np.zeros((n,), dtype=np.float32),
            "sum_inlier_patch_ncc": np.zeros((n,), dtype=np.float32),
            "gt_pose_match_support": np.zeros((n,), dtype=np.float32),
            "gt_pose_inlier_support": np.zeros((n,), dtype=np.float32),
            "sum_gt_pose_sampson_px": np.zeros((n,), dtype=np.float32),
            "min_gt_pose_sampson_px": np.full((n,), np.inf, dtype=np.float32),
            "sum_gt_pose_consistency": np.zeros((n,), dtype=np.float32),
        }

    focal = float(max(h, w))
    camera_matrix = np.asarray([[focal, 0.0, 0.5 * (w - 1)], [0.0, focal, 0.5 * (h - 1)], [0.0, 0.0, 1.0]])
    dist_norm = 256.0 if norm == cv2.NORM_HAMMING else 512.0
    pair_metrics: list[dict[str, Any]] = []
    for f0, f1 in zip(frame_ids[:-1], frame_ids[1:]):
        matches = match_descriptors(descriptors_by_frame[f0], descriptors_by_frame[f1], norm, ratio)
        for match in matches:
            stats_by_frame[f0]["match_support"][match.queryIdx] += 1.0
            stats_by_frame[f1]["match_support"][match.trainIdx] += 1.0
            d = float(match.distance) / dist_norm
            stats_by_frame[f0]["sum_match_distance"][match.queryIdx] += d
            stats_by_frame[f1]["sum_match_distance"][match.trainIdx] += d

        pose_errors_px: np.ndarray | None = None
        pose_inlier_count = 0
        if pose_rows is not None and len(matches) > 0 and max(f0, f1) < len(pose_rows):
            pts0_pose = np.asarray([keypoints_by_frame[f0][m.queryIdx].pt for m in matches], dtype=np.float32)
            pts1_pose = np.asarray([keypoints_by_frame[f1][m.trainIdx].pt for m in matches], dtype=np.float32)
            try:
                essentials = _essential_candidates_from_pose_rows(pose_rows[f0], pose_rows[f1])
                pose_errors_px = _sampson_errors_px(
                    pts0_pose,
                    pts1_pose,
                    essentials,
                    focal=focal,
                    cx=0.5 * float(w - 1),
                    cy=0.5 * float(h - 1),
                )
            except Exception:
                pose_errors_px = None
            if pose_errors_px is not None and pose_errors_px.size == len(matches):
                threshold_px = float(max(1e-6, pose_sampson_threshold_px))
                for match, err_px in zip(matches, pose_errors_px):
                    err = float(err_px) if np.isfinite(float(err_px)) else threshold_px * 100.0
                    consistency = float(np.exp(-min(err / threshold_px, 50.0)))
                    pose_ok = err <= threshold_px
                    if pose_ok:
                        pose_inlier_count += 1
                    for frame_idx, kp_idx in ((f0, match.queryIdx), (f1, match.trainIdx)):
                        stats_by_frame[frame_idx]["gt_pose_match_support"][kp_idx] += 1.0
                        stats_by_frame[frame_idx]["sum_gt_pose_sampson_px"][kp_idx] += err
                        stats_by_frame[frame_idx]["min_gt_pose_sampson_px"][kp_idx] = min(
                            float(stats_by_frame[frame_idx]["min_gt_pose_sampson_px"][kp_idx]),
                            err,
                        )
                        stats_by_frame[frame_idx]["sum_gt_pose_consistency"][kp_idx] += consistency
                        if pose_ok:
                            stats_by_frame[frame_idx]["gt_pose_inlier_support"][kp_idx] += 1.0

        inlier_mask = None
        if len(matches) >= 8:
            pts0 = np.asarray([keypoints_by_frame[f0][m.queryIdx].pt for m in matches], dtype=np.float32)
            pts1 = np.asarray([keypoints_by_frame[f1][m.trainIdx].pt for m in matches], dtype=np.float32)
            try:
                _, raw_mask = cv2.findEssentialMat(
                    pts0,
                    pts1,
                    cameraMatrix=camera_matrix,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=float(essential_threshold),
                )
                if raw_mask is not None:
                    inlier_mask = raw_mask.reshape(-1).astype(bool)
                    if int(inlier_mask.sum()) < 8:
                        inlier_mask = None
            except cv2.error:
                inlier_mask = None

        inlier_count = 0
        if inlier_mask is not None:
            for match, is_inlier in zip(matches, inlier_mask):
                if not bool(is_inlier):
                    continue
                kp0 = keypoints_by_frame[f0][match.queryIdx]
                kp1 = keypoints_by_frame[f1][match.trainIdx]
                flow = float(np.linalg.norm(np.asarray(kp0.pt) - np.asarray(kp1.pt))) / float(max(h, w))
                d = float(match.distance) / dist_norm
                ncc = _patch_ncc(gray_by_frame[f0], gray_by_frame[f1], kp0.pt, kp1.pt)
                for frame_idx, kp_idx in ((f0, match.queryIdx), (f1, match.trainIdx)):
                    stats_by_frame[frame_idx]["inlier_support"][kp_idx] += 1.0
                    stats_by_frame[frame_idx]["sum_inlier_distance"][kp_idx] += d
                    stats_by_frame[frame_idx]["sum_inlier_flow"][kp_idx] += flow
                    stats_by_frame[frame_idx]["sum_inlier_patch_ncc"][kp_idx] += ncc
                inlier_count += 1

        pair_metrics.append(
            {
                "frame0": int(f0),
                "frame1": int(f1),
                "matches": int(len(matches)),
                "essential_inliers": int(inlier_count),
                "essential_success": bool(inlier_count >= 8),
                "gt_pose_inliers": int(pose_inlier_count),
                "gt_pose_sampson_median_px": float(np.median(pose_errors_px[np.isfinite(pose_errors_px)]))
                if pose_errors_px is not None and bool(np.isfinite(pose_errors_px).any())
                else None,
            }
        )

    return {
        "frame_ids": frame_ids,
        "keypoints_by_frame": keypoints_by_frame,
        "descriptors_by_frame": descriptors_by_frame,
        "stats_by_frame": stats_by_frame,
        "norm": int(norm),
        "ratio": float(ratio),
        "pair_metrics": pair_metrics,
        "num_keypoints": int(sum(len(v) for v in keypoints_by_frame.values())),
        "num_pairs": int(len(pair_metrics)),
        "num_pairs_with_model": int(sum(1 for item in pair_metrics if bool(item["essential_success"]))),
    }


def _sample_score(score_map: np.ndarray, frame_idx: int, x: int, y: int) -> float:
    t = int(np.clip(frame_idx, 0, score_map.shape[0] - 1))
    yy = int(np.clip(y, 0, score_map.shape[1] - 1))
    xx = int(np.clip(x, 0, score_map.shape[2] - 1))
    return float(score_map[t, yy, xx])


def build_retention_candidate_table(
    *,
    video_rgb: np.ndarray,
    transient_gt: np.ndarray,
    score_maps: dict[str, np.ndarray],
    rejected_mask: np.ndarray,
    keypoint_context: dict[str, Any],
    dynamic_threshold: float,
    particle_threshold: float,
    static_threshold: float,
    min_positive_inlier_support: int,
    clip_index: int,
) -> dict[str, Any]:
    h, w = video_rgb.shape[1:3]
    features: list[list[float]] = []
    labels: list[int] = []
    meta_rows: list[list[float]] = []
    for frame_idx in keypoint_context["frame_ids"]:
        keypoints = keypoint_context["keypoints_by_frame"][frame_idx]
        stats = keypoint_context["stats_by_frame"][frame_idx]
        responses = np.asarray([max(float(kp.response), 0.0) for kp in keypoints], dtype=np.float32)
        if responses.size:
            order = np.argsort(-responses)
            ranks = np.empty_like(order, dtype=np.float32)
            ranks[order] = np.arange(responses.size, dtype=np.float32)
            response_rank = 1.0 - ranks / float(max(1, responses.size - 1))
        else:
            response_rank = np.zeros((0,), dtype=np.float32)

        for kp_idx, kp in enumerate(keypoints):
            x = int(np.clip(round(kp.pt[0]), 0, w - 1))
            y = int(np.clip(round(kp.pt[1]), 0, h - 1))
            if not bool(rejected_mask[frame_idx, y, x]):
                continue
            dyn = _sample_score(score_maps["dynamic_prob"], frame_idx, x, y)
            particle = _sample_score(score_maps["particle_prob"], frame_idx, x, y)
            static_conf = _sample_score(score_maps["static_confidence"], frame_idx, x, y)
            confidence = _sample_score(score_maps["confidence_prob"], frame_idx, x, y)
            transient_prob = max(dyn, particle)
            rejection_margin = max(
                dyn - float(dynamic_threshold),
                particle - float(particle_threshold),
                float(static_threshold) - static_conf,
            )
            match_support = float(stats["match_support"][kp_idx])
            inlier_support = float(stats["inlier_support"][kp_idx])
            inlier_ratio = inlier_support / max(1.0, match_support)
            mean_match_distance = float(stats["sum_match_distance"][kp_idx]) / max(1.0, match_support)
            mean_inlier_distance = float(stats["sum_inlier_distance"][kp_idx]) / max(1.0, inlier_support)
            mean_inlier_flow = float(stats["sum_inlier_flow"][kp_idx]) / max(1.0, inlier_support)
            mean_inlier_ncc = float(stats["sum_inlier_patch_ncc"][kp_idx]) / max(1.0, inlier_support)
            gt_transient = bool(transient_gt[frame_idx, y, x])
            stable = inlier_support >= float(min_positive_inlier_support)
            label = int((not gt_transient) and stable)
            features.append(
                [
                    dyn,
                    particle,
                    static_conf,
                    confidence,
                    transient_prob,
                    rejection_margin,
                    float(kp.response),
                    float(kp.size),
                    float(kp.octave),
                    float(response_rank[kp_idx]) if response_rank.size else 0.0,
                    match_support,
                    inlier_support,
                    inlier_ratio,
                    mean_match_distance,
                    mean_inlier_distance,
                    mean_inlier_flow,
                    mean_inlier_ncc,
                ]
            )
            labels.append(label)
            meta_rows.append(
                [
                    float(clip_index),
                    float(frame_idx),
                    float(x),
                    float(y),
                    float(gt_transient),
                    float(stable),
                    match_support,
                    inlier_support,
                    float(kp.response),
                ]
            )

    feature_arr = np.asarray(features, dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
    label_arr = np.asarray(labels, dtype=np.int64)
    meta_arr = np.asarray(meta_rows, dtype=np.float32).reshape(-1, 9)
    return {
        "features": feature_arr,
        "labels": label_arr,
        "candidate_meta": meta_arr,
        "feature_names": list(FEATURE_NAMES),
        "candidate_meta_names": list(RETENTION_META_NAMES),
        "summary": summarize_candidate_table(feature_arr, label_arr, meta_arr),
    }


def build_pose_retention_candidate_table(
    *,
    video_rgb: np.ndarray,
    transient_gt: np.ndarray,
    score_maps: dict[str, np.ndarray],
    rejected_mask: np.ndarray,
    keypoint_context: dict[str, Any],
    dynamic_threshold: float,
    particle_threshold: float,
    static_threshold: float,
    min_positive_inlier_support: int,
    min_positive_pose_inlier_support: int,
    pose_sampson_threshold_px: float,
    clip_index: int,
    include_pose_features: bool = False,
) -> dict[str, Any]:
    h, w = video_rgb.shape[1:3]
    features: list[list[float]] = []
    labels: list[int] = []
    meta_rows: list[list[float]] = []
    fallback_pose_error = float(max(1.0, pose_sampson_threshold_px) * 10.0)
    for frame_idx in keypoint_context["frame_ids"]:
        keypoints = keypoint_context["keypoints_by_frame"][frame_idx]
        stats = keypoint_context["stats_by_frame"][frame_idx]
        responses = np.asarray([max(float(kp.response), 0.0) for kp in keypoints], dtype=np.float32)
        if responses.size:
            order = np.argsort(-responses)
            ranks = np.empty_like(order, dtype=np.float32)
            ranks[order] = np.arange(responses.size, dtype=np.float32)
            response_rank = 1.0 - ranks / float(max(1, responses.size - 1))
        else:
            response_rank = np.zeros((0,), dtype=np.float32)

        for kp_idx, kp in enumerate(keypoints):
            x = int(np.clip(round(kp.pt[0]), 0, w - 1))
            y = int(np.clip(round(kp.pt[1]), 0, h - 1))
            if not bool(rejected_mask[frame_idx, y, x]):
                continue
            dyn = _sample_score(score_maps["dynamic_prob"], frame_idx, x, y)
            particle = _sample_score(score_maps["particle_prob"], frame_idx, x, y)
            static_conf = _sample_score(score_maps["static_confidence"], frame_idx, x, y)
            confidence = _sample_score(score_maps["confidence_prob"], frame_idx, x, y)
            transient_prob = max(dyn, particle)
            rejection_margin = max(
                dyn - float(dynamic_threshold),
                particle - float(particle_threshold),
                float(static_threshold) - static_conf,
            )
            match_support = float(stats["match_support"][kp_idx])
            inlier_support = float(stats["inlier_support"][kp_idx])
            inlier_ratio = inlier_support / max(1.0, match_support)
            mean_match_distance = float(stats["sum_match_distance"][kp_idx]) / max(1.0, match_support)
            mean_inlier_distance = float(stats["sum_inlier_distance"][kp_idx]) / max(1.0, inlier_support)
            mean_inlier_flow = float(stats["sum_inlier_flow"][kp_idx]) / max(1.0, inlier_support)
            mean_inlier_ncc = float(stats["sum_inlier_patch_ncc"][kp_idx]) / max(1.0, inlier_support)
            pose_match_support = float(stats.get("gt_pose_match_support", np.zeros((len(keypoints),), dtype=np.float32))[kp_idx])
            pose_inlier_support = float(stats.get("gt_pose_inlier_support", np.zeros((len(keypoints),), dtype=np.float32))[kp_idx])
            pose_inlier_ratio = pose_inlier_support / max(1.0, pose_match_support)
            sum_pose_error = float(stats.get("sum_gt_pose_sampson_px", np.zeros((len(keypoints),), dtype=np.float32))[kp_idx])
            mean_pose_error = sum_pose_error / max(1.0, pose_match_support) if pose_match_support > 0.0 else fallback_pose_error
            min_pose_error_raw = float(
                stats.get("min_gt_pose_sampson_px", np.full((len(keypoints),), np.inf, dtype=np.float32))[kp_idx]
            )
            min_pose_error = min_pose_error_raw if np.isfinite(min_pose_error_raw) else fallback_pose_error
            pose_consistency = float(stats.get("sum_gt_pose_consistency", np.zeros((len(keypoints),), dtype=np.float32))[kp_idx])
            mean_pose_consistency = pose_consistency / max(1.0, pose_match_support)
            gt_transient = bool(transient_gt[frame_idx, y, x])
            stable = inlier_support >= float(min_positive_inlier_support)
            pose_stable = pose_inlier_support >= float(min_positive_pose_inlier_support)
            low_pose_error = min_pose_error <= float(pose_sampson_threshold_px)
            label = int((not gt_transient) and stable and (pose_stable or low_pose_error))
            base_features = [
                dyn,
                particle,
                static_conf,
                confidence,
                transient_prob,
                rejection_margin,
                float(kp.response),
                float(kp.size),
                float(kp.octave),
                float(response_rank[kp_idx]) if response_rank.size else 0.0,
                match_support,
                inlier_support,
                inlier_ratio,
                mean_match_distance,
                mean_inlier_distance,
                mean_inlier_flow,
                mean_inlier_ncc,
            ]
            if include_pose_features:
                base_features.extend(
                    [
                        pose_match_support,
                        pose_inlier_support,
                        pose_inlier_ratio,
                        float(np.log1p(max(0.0, mean_pose_error))),
                        float(np.log1p(max(0.0, min_pose_error))),
                        mean_pose_consistency,
                    ]
                )
            features.append(base_features)
            labels.append(label)
            meta_rows.append(
                [
                    float(clip_index),
                    float(frame_idx),
                    float(x),
                    float(y),
                    float(gt_transient),
                    float(stable),
                    match_support,
                    inlier_support,
                    float(kp.response),
                    float(pose_stable or low_pose_error),
                    pose_match_support,
                    pose_inlier_support,
                    mean_pose_error,
                    min_pose_error,
                    mean_pose_consistency,
                ]
            )

    feature_names = list(POSE_RETENTION_FEATURE_NAMES if include_pose_features else FEATURE_NAMES)
    feature_arr = np.asarray(features, dtype=np.float32).reshape(-1, len(feature_names))
    label_arr = np.asarray(labels, dtype=np.int64)
    meta_arr = np.asarray(meta_rows, dtype=np.float32).reshape(-1, len(POSE_RETENTION_META_NAMES))
    return {
        "features": feature_arr,
        "labels": label_arr,
        "candidate_meta": meta_arr,
        "feature_names": feature_names,
        "candidate_meta_names": list(POSE_RETENTION_META_NAMES),
        "summary": summarize_candidate_table(feature_arr, label_arr, meta_arr),
    }


def summarize_candidate_table(features: np.ndarray, labels: np.ndarray, candidate_meta: np.ndarray) -> dict[str, Any]:
    if labels.size == 0:
        return {
            "num_candidates": 0,
            "positive_rate": 0.0,
            "gt_transient_rate": 0.0,
            "stable_rate": 0.0,
            "mean_inlier_support": 0.0,
            "pose_stable_rate": 0.0,
            "mean_gt_pose_inlier_support": 0.0,
            "mean_gt_pose_sampson_px": 0.0,
        }
    out = {
        "num_candidates": int(labels.size),
        "positive_rate": float(labels.mean()),
        "gt_transient_rate": float(candidate_meta[:, 4].mean()) if candidate_meta.size else 0.0,
        "stable_rate": float(candidate_meta[:, 5].mean()) if candidate_meta.size else 0.0,
        "mean_inlier_support": float(candidate_meta[:, 7].mean()) if candidate_meta.size else 0.0,
    }
    if candidate_meta.size and candidate_meta.shape[1] > 12:
        out.update(
            {
                "pose_stable_rate": float(candidate_meta[:, 9].mean()),
                "mean_gt_pose_inlier_support": float(candidate_meta[:, 11].mean()),
                "mean_gt_pose_sampson_px": float(candidate_meta[:, 12].mean()),
            }
        )
    return out


def save_candidate_npz(path: str | Path, payload: dict[str, Any], metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=payload["features"].astype(np.float32),
        labels=payload["labels"].astype(np.int64),
        candidate_meta=payload["candidate_meta"].astype(np.float32),
        feature_names=np.asarray(payload["feature_names"], dtype=object),
        candidate_meta_names=np.asarray(payload["candidate_meta_names"], dtype=object),
        metadata_json=np.asarray(json.dumps(metadata), dtype=object),
    )


def load_candidate_npz(path: str | Path) -> dict[str, Any]:
    data = np.load(str(path), allow_pickle=True)
    metadata_json = str(data["metadata_json"].item()) if "metadata_json" in data else "{}"
    return {
        "features": data["features"].astype(np.float32),
        "labels": data["labels"].astype(np.int64),
        "candidate_meta": data["candidate_meta"].astype(np.float32),
        "feature_names": [str(v) for v in data["feature_names"].tolist()],
        "candidate_meta_names": [str(v) for v in data["candidate_meta_names"].tolist()],
        "metadata": json.loads(metadata_json),
    }


def build_scorer(input_dim: int, hidden_dim: int = 0, dropout: float = 0.0) -> torch.nn.Module:
    if int(hidden_dim) <= 0:
        return torch.nn.Linear(int(input_dim), 1)
    return torch.nn.Sequential(
        torch.nn.Linear(int(input_dim), int(hidden_dim)),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(float(dropout)),
        torch.nn.Linear(int(hidden_dim), 1),
    )


def load_retention_scorer(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    payload = torch.load(str(path), map_location=device, weights_only=False)
    feature_names = [str(v) for v in payload["feature_names"]]
    model = build_scorer(
        input_dim=len(feature_names),
        hidden_dim=int(payload.get("hidden_dim", 0)),
        dropout=0.0,
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return {
        "model": model,
        "feature_names": feature_names,
        "mean": torch.as_tensor(payload["feature_mean"], dtype=torch.float32, device=device),
        "std": torch.as_tensor(payload["feature_std"], dtype=torch.float32, device=device),
        "threshold": float(payload.get("selected_threshold", 0.5)),
        "metadata": payload.get("metadata", {}),
    }


def align_candidate_features_to_scorer(
    features: np.ndarray,
    *,
    source_feature_names: list[str] | None,
    target_feature_names: list[str],
) -> np.ndarray:
    if features.size == 0:
        return features.reshape(0, len(target_feature_names)).astype(np.float32)
    if source_feature_names is None:
        if features.shape[1] != len(target_feature_names):
            raise ValueError(
                f"Candidate features have dim={features.shape[1]}, scorer expects {len(target_feature_names)}. "
                "Pass feature_names so columns can be aligned by name."
            )
        return features.astype(np.float32)
    source = [str(name) for name in source_feature_names]
    target = [str(name) for name in target_feature_names]
    index = {name: idx for idx, name in enumerate(source)}
    missing = [name for name in target if name not in index]
    if missing:
        raise ValueError(f"Candidate feature schema is missing scorer features: {missing}")
    order = [index[name] for name in target]
    return features[:, order].astype(np.float32)


def score_retention_candidates(
    features: np.ndarray,
    scorer: dict[str, Any],
    device: torch.device | str = "cpu",
    batch_size: int = 65536,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    if features.size == 0:
        return np.zeros((0,), dtype=np.float32)
    features = align_candidate_features_to_scorer(
        features,
        source_feature_names=feature_names,
        target_feature_names=scorer["feature_names"],
    )
    model = scorer["model"]
    mean = scorer["mean"]
    std = scorer["std"]
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], int(batch_size)):
            batch = torch.as_tensor(features[start : start + int(batch_size)], dtype=torch.float32, device=device)
            batch = (batch - mean) / torch.clamp(std, min=1e-6)
            logits = model(batch).reshape(-1)
            out.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def retention_mask_from_candidates(
    *,
    candidate_meta: np.ndarray,
    scores: np.ndarray,
    rejected_mask: np.ndarray,
    score_threshold: float,
    patch_radius: int,
    max_features_per_frame: int,
    max_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    retain = np.zeros(rejected_mask.shape, dtype=bool)
    if candidate_meta.size == 0 or scores.size == 0:
        return retain, {
            "score_threshold": float(score_threshold),
            "patch_radius": int(patch_radius),
            "retained_features_total": 0,
            "candidate_features_total": 0,
            "retained_pixel_fraction": 0.0,
        }

    radius = max(0, int(patch_radius))
    max_frame = max(0, int(max_features_per_frame))
    max_frac = float(np.clip(max_fraction, 0.0, 1.0))
    h, w = rejected_mask.shape[1:3]
    retained_features_total = 0
    candidate_features_total = int(scores.size)
    selected_total = int((scores >= float(score_threshold)).sum())
    for frame_idx in sorted(set(candidate_meta[:, 1].astype(np.int64).tolist())):
        idx = np.where(candidate_meta[:, 1].astype(np.int64) == int(frame_idx))[0]
        idx = idx[scores[idx] >= float(score_threshold)]
        if idx.size == 0:
            continue
        order = idx[np.argsort(-scores[idx])]
        if max_frame > 0:
            order = order[:max_frame]
        frame_retain = np.zeros((h, w), dtype=np.uint8)
        kept = 0
        for row_idx in order:
            x = int(np.clip(round(float(candidate_meta[row_idx, 2])), 0, w - 1))
            y = int(np.clip(round(float(candidate_meta[row_idx, 3])), 0, h - 1))
            if radius <= 0:
                frame_retain[y, x] = 1
            else:
                cv2.circle(frame_retain, (x, y), radius, 1, thickness=-1)
            kept += 1
            if max_frac > 0.0 and float(frame_retain.mean()) >= max_frac:
                break
        retained_features_total += kept
        retain[int(frame_idx)] = frame_retain.astype(bool) & rejected_mask[int(frame_idx)]

    return retain, {
        "score_threshold": float(score_threshold),
        "patch_radius": int(patch_radius),
        "max_features_per_frame": int(max_features_per_frame),
        "max_fraction": float(max_fraction),
        "candidate_features_total": int(candidate_features_total),
        "selected_features_total": int(selected_total),
        "retained_features_total": int(retained_features_total),
        "retained_pixel_fraction": float(retain.mean()) if retain.size else 0.0,
    }


def pose_aware_retention_weight_from_candidates(
    *,
    candidate_meta: np.ndarray,
    features: np.ndarray,
    scores: np.ndarray,
    rejected_mask: np.ndarray,
    score_threshold: float,
    patch_radius: int,
    max_features_per_frame: int,
    max_fraction: float,
    score_power: float = 1.0,
    geometry_power: float = 1.0,
    min_weight: float = 0.15,
    max_weight: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a continuous pose-aware original-image retention weight map.

    Unlike ``retention_mask_from_candidates``, this does not make a binary copy
    decision for every retained patch. Each candidate contributes an
    original-image weight derived from the learned score and geometric support:
    inlier support, inlier ratio, match distance, flow, and patch NCC. The
    resulting map lets soft rendering keep high-support background features
    while still temporally filling weaker rejected pixels.
    """

    retain_weight = np.zeros(rejected_mask.shape, dtype=np.float32)
    retain_mask = np.zeros(rejected_mask.shape, dtype=np.bool_)
    if candidate_meta.size == 0 or features.size == 0 or scores.size == 0:
        return retain_weight, retain_mask, {
            "score_threshold": float(score_threshold),
            "patch_radius": int(patch_radius),
            "candidate_features_total": 0,
            "selected_features_total": 0,
            "retained_features_total": 0,
            "retained_pixel_fraction": 0.0,
            "mean_retention_weight": 0.0,
        }

    radius = max(0, int(patch_radius))
    max_frame = max(0, int(max_features_per_frame))
    max_frac = float(np.clip(max_fraction, 0.0, 1.0))
    min_w = float(np.clip(min_weight, 0.0, 1.0))
    max_w = float(np.clip(max_weight, min_w, 1.0))
    h, w = rejected_mask.shape[1:3]

    score_arr = np.clip(np.asarray(scores, dtype=np.float32), 0.0, 1.0)
    feature_arr = np.asarray(features, dtype=np.float32)
    inlier_support = np.clip(feature_arr[:, 11], 0.0, None)
    inlier_ratio = np.clip(feature_arr[:, 12], 0.0, 1.0)
    mean_match_distance = np.clip(feature_arr[:, 13], 0.0, 1.0)
    mean_inlier_flow = np.clip(feature_arr[:, 15], 0.0, 1.0)
    mean_inlier_ncc = np.clip((feature_arr[:, 16] + 1.0) * 0.5, 0.0, 1.0)

    support_score = np.clip(inlier_support / 3.0, 0.0, 1.0)
    distance_score = 1.0 - mean_match_distance
    flow_score = 1.0 - mean_inlier_flow
    geometry_score = (
        0.35 * support_score
        + 0.25 * inlier_ratio
        + 0.20 * mean_inlier_ncc
        + 0.10 * distance_score
        + 0.10 * flow_score
    )
    geometry_score = np.clip(geometry_score, 0.0, 1.0).astype(np.float32)
    raw_weight = np.power(score_arr, float(score_power)) * np.power(geometry_score, float(geometry_power))
    candidate_weight = min_w + (max_w - min_w) * np.clip(raw_weight, 0.0, 1.0)

    selected = score_arr >= float(score_threshold)
    selected_total = int(selected.sum())
    retained_features_total = 0
    for frame_idx in sorted(set(candidate_meta[:, 1].astype(np.int64).tolist())):
        idx = np.where(candidate_meta[:, 1].astype(np.int64) == int(frame_idx))[0]
        idx = idx[selected[idx]]
        if idx.size == 0:
            continue
        # Prefer candidates that have both learned confidence and geometric
        # stability. This is the pose-aware difference from plain learned soft.
        order = idx[np.argsort(-(score_arr[idx] * geometry_score[idx]))]
        if max_frame > 0:
            order = order[:max_frame]
        frame_weight = retain_weight[int(frame_idx)].copy()
        frame_mask = retain_mask[int(frame_idx)].copy()
        kept = 0
        for row_idx in order:
            x = int(np.clip(round(float(candidate_meta[row_idx, 2])), 0, w - 1))
            y = int(np.clip(round(float(candidate_meta[row_idx, 3])), 0, h - 1))
            patch = np.zeros((h, w), dtype=np.uint8)
            if radius <= 0:
                patch[y, x] = 1
            else:
                cv2.circle(patch, (x, y), radius, 1, thickness=-1)
            patch_bool = patch.astype(bool) & rejected_mask[int(frame_idx)]
            if not bool(patch_bool.any()):
                continue
            frame_weight[patch_bool] = np.maximum(frame_weight[patch_bool], float(candidate_weight[row_idx]))
            frame_mask |= patch_bool
            kept += 1
            if max_frac > 0.0 and float(frame_mask.mean()) >= max_frac:
                break
        retained_features_total += kept
        retain_weight[int(frame_idx)] = frame_weight * rejected_mask[int(frame_idx)].astype(np.float32)
        retain_mask[int(frame_idx)] = frame_mask & rejected_mask[int(frame_idx)]

    valid = retain_weight[retain_mask] if bool(retain_mask.any()) else np.asarray([], dtype=np.float32)
    return retain_weight, retain_mask, {
        "score_threshold": float(score_threshold),
        "patch_radius": int(patch_radius),
        "max_features_per_frame": int(max_features_per_frame),
        "max_fraction": float(max_fraction),
        "score_power": float(score_power),
        "geometry_power": float(geometry_power),
        "min_weight": float(min_w),
        "max_weight": float(max_w),
        "candidate_features_total": int(scores.size),
        "selected_features_total": int(selected_total),
        "retained_features_total": int(retained_features_total),
        "retained_pixel_fraction": float(retain_mask.mean()) if retain_mask.size else 0.0,
        "mean_retention_weight": float(valid.mean()) if valid.size else 0.0,
        "mean_selected_geometry_score": float(geometry_score[selected].mean()) if selected_total else 0.0,
    }
