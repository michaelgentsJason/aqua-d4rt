#!/usr/bin/env python3
"""Evaluate Aqua-D4RT downstream variants against GT camera positions."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import shutil
import sys
import traceback
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

from aqua_prefilter_utils import (  # noqa: E402
    inpaint_video_with_mask,
    soft_temporal_fill_video,
    temporal_rgb_pseudo_mask,
)
from aqua_retention_utils import (  # noqa: E402
    aqua_dense_score_maps,
    build_pose_retention_candidate_table,
    build_retention_candidate_table,
    extract_keypoint_context,
    load_retention_scorer,
    pose_aware_retention_weight_from_candidates,
    retention_mask_from_candidates,
    score_retention_candidates,
)
from eval_aqua_downstream_slam_proxy import (  # noqa: E402
    _detect_features,
    _match_descriptors,
    _pair_metrics,
    _resize_mask_stack,
    _safe_mean,
    _sample_mask,
    _slam_aware_retention_mask,
)
from eval_aqua_pycolmap_validation import _run_pycolmap_variant  # noqa: E402
from eval_aqua_transient_heads import _load_model  # noqa: E402
from infer_track_3d import _resolve_device  # noqa: E402
from src.core import load_yaml_config, seed_everything  # noqa: E402


def _read_manifest_list(path: str | Path) -> list[str]:
    out: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            out.append(item)
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


def _load_rgb(path: str | Path, image_hw: tuple[int, int] | None) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read frame: {path}")
    if image_hw is not None and bgr.shape[:2] != image_hw:
        h, w = image_hw
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resize_mask(mask: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = image_hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def _load_pose_clip(
    manifest_path: Path,
    *,
    image_hw: tuple[int, int] | None,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_paths = [str(path) for path in manifest["frames"]]
    if max_frames > 0:
        frame_paths = frame_paths[: int(max_frames)]
    frames = [_load_rgb(path, image_hw) for path in frame_paths]
    video = np.stack(frames, axis=0)
    labels_path = Path(str(manifest["labels_npz"]))
    masks = np.load(str(labels_path))
    dynamic = masks["dynamic_object_mask"][: len(frames)]
    particle = masks["particle_mask"][: len(frames)]
    image_size = video.shape[1:3]
    dynamic = np.stack([_resize_mask(dynamic[t], image_size) for t in range(dynamic.shape[0])], axis=0)
    particle = np.stack([_resize_mask(particle[t], image_size) for t in range(particle.shape[0])], axis=0)
    pose_rows = _load_pose_rows(Path(str(manifest["frames_csv"])), len(frames))
    return video, dynamic, particle, manifest, pose_rows


def _load_pose_rows(path: Path, num_frames: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "frame_index": int(row["frame_index"]),
                    "image_path": row["image_path"],
                    "timestamp_ns": int(float(row["timestamp_ns"])),
                    "position": np.asarray(
                        [float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])],
                        dtype=np.float64,
                    ),
                    "quaternion_xyzw": np.asarray(
                        [
                            float(row["orient_qx"]),
                            float(row["orient_qy"]),
                            float(row["orient_qz"]),
                            float(row["orient_qw"]),
                        ],
                        dtype=np.float64,
                    ),
                    "transient_mask_path": row.get("transient_mask_path", ""),
                }
            )
    if len(rows) < int(num_frames):
        raise RuntimeError(f"frames.csv has {len(rows)} rows but {num_frames} frames are requested: {path}")
    return rows[: int(num_frames)]


def _frame_ids_for(num_frames: int, frame_stride: int) -> list[int]:
    return list(range(0, int(num_frames), max(1, int(frame_stride))))


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


def _rotation_angle_deg(rot: np.ndarray) -> float:
    trace = float(np.trace(rot))
    value = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(math.degrees(math.acos(value)))


def _umeyama_align(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected [N,3] arrays, got {src.shape} and {dst.shape}")
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 registered poses for Sim(3) alignment")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    x = src - mu_src
    y = dst - mu_dst
    var_src = float(np.sum(x * x) / max(1, src.shape[0]))
    cov = (y.T @ x) / float(max(1, src.shape[0]))
    u, singular, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[-1, -1] = -1.0
    rot = u @ s_mat @ vt
    scale = float(np.trace(np.diag(singular) @ s_mat) / max(var_src, 1e-12))
    trans = mu_dst - scale * (rot @ mu_src)
    aligned = (scale * (rot @ src.T)).T + trans
    return aligned, {
        "scale": scale,
        "rotation": rot.tolist(),
        "translation": trans.tolist(),
        "det_rotation": float(np.linalg.det(rot)),
    }


def _load_reconstruction_poses(sparse_dir: str | Path) -> dict[str, dict[str, Any]]:
    import pycolmap

    recon = pycolmap.Reconstruction(str(sparse_dir))
    poses: dict[str, dict[str, Any]] = {}
    for image_id, image in recon.images.items():
        if not bool(image.has_pose):
            continue
        try:
            cam_from_world = image.cam_from_world()
            rot_cw = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
        except Exception:
            rot_cw = np.eye(3, dtype=np.float64)
        center = np.asarray(image.projection_center(), dtype=np.float64)
        poses[str(image.name)] = {
            "image_id": int(image_id),
            "center": center,
            "rot_cw": rot_cw,
        }
    return poses


def _pose_metrics_for_variant(
    *,
    variant_result: dict[str, Any],
    pose_rows: list[dict[str, Any]],
    frame_stride: int,
) -> dict[str, Any]:
    if not bool(variant_result.get("success", False)) or not variant_result.get("sparse_dir"):
        return {
            "pose_eval_success": False,
            "num_pose_pairs": 0,
            "ate_rmse": None,
            "ate_mean": None,
            "rpe_trans_rmse": None,
            "orientation_rpe_deg_mean": None,
            "alignment": None,
        }
    try:
        poses = _load_reconstruction_poses(variant_result["sparse_dir"])
    except Exception as exc:
        return {
            "pose_eval_success": False,
            "pose_error": str(exc),
            "num_pose_pairs": 0,
            "ate_rmse": None,
            "ate_mean": None,
            "rpe_trans_rmse": None,
            "orientation_rpe_deg_mean": None,
            "alignment": None,
        }

    frame_ids = _frame_ids_for(len(pose_rows), frame_stride)
    pred_centers: list[np.ndarray] = []
    gt_centers: list[np.ndarray] = []
    gt_rots: list[np.ndarray] = []
    pred_rots_cw: list[np.ndarray] = []
    matched_frame_ids: list[int] = []
    for out_idx, src_idx in enumerate(frame_ids):
        name = f"frame_{out_idx:04d}.png"
        if name not in poses:
            continue
        pred_centers.append(poses[name]["center"])
        gt_centers.append(pose_rows[src_idx]["position"])
        gt_rots.append(_quat_xyzw_to_rot(pose_rows[src_idx]["quaternion_xyzw"]))
        pred_rots_cw.append(poses[name]["rot_cw"])
        matched_frame_ids.append(int(src_idx))

    if len(pred_centers) < 3:
        return {
            "pose_eval_success": False,
            "num_pose_pairs": len(pred_centers),
            "matched_frame_ids": matched_frame_ids,
            "ate_rmse": None,
            "ate_mean": None,
            "rpe_trans_rmse": None,
            "orientation_rpe_deg_mean": None,
            "alignment": None,
        }

    pred = np.stack(pred_centers, axis=0)
    gt = np.stack(gt_centers, axis=0)
    try:
        aligned, alignment = _umeyama_align(pred, gt)
    except Exception as exc:
        return {
            "pose_eval_success": False,
            "pose_error": str(exc),
            "num_pose_pairs": len(pred_centers),
            "matched_frame_ids": matched_frame_ids,
            "ate_rmse": None,
            "ate_mean": None,
            "rpe_trans_rmse": None,
            "orientation_rpe_deg_mean": None,
            "alignment": None,
        }

    ate = np.linalg.norm(aligned - gt, axis=1)
    pred_steps = np.linalg.norm(np.diff(aligned, axis=0), axis=1)
    gt_steps = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    rpe_trans = np.abs(pred_steps - gt_steps)

    # This orientation metric is secondary because COLMAP and GT camera frame
    # conventions may differ. Translation ATE/RPE is the main robotics signal.
    orient_errors: list[float] = []
    if len(gt_rots) == len(pred_rots_cw):
        for i in range(len(gt_rots) - 1):
            gt_rel = gt_rots[i + 1].T @ gt_rots[i]
            pred_world_from_cam_i = pred_rots_cw[i].T
            pred_world_from_cam_j = pred_rots_cw[i + 1].T
            pred_rel = pred_world_from_cam_j.T @ pred_world_from_cam_i
            orient_errors.append(_rotation_angle_deg(pred_rel.T @ gt_rel))

    return {
        "pose_eval_success": True,
        "num_pose_pairs": int(len(pred_centers)),
        "matched_frame_ids": matched_frame_ids,
        "ate_rmse": float(np.sqrt(np.mean(ate * ate))) if ate.size else None,
        "ate_mean": float(np.mean(ate)) if ate.size else None,
        "ate_median": float(np.median(ate)) if ate.size else None,
        "ate_max": float(np.max(ate)) if ate.size else None,
        "rpe_trans_rmse": float(np.sqrt(np.mean(rpe_trans * rpe_trans))) if rpe_trans.size else None,
        "rpe_trans_mean": float(np.mean(rpe_trans)) if rpe_trans.size else None,
        "orientation_rpe_deg_mean": float(np.mean(orient_errors)) if orient_errors else None,
        "orientation_rpe_deg_median": float(np.median(orient_errors)) if orient_errors else None,
        "alignment": alignment,
    }


def _variant_frontend_metrics(
    *,
    video: np.ndarray,
    transient_gt: np.ndarray,
    static_masks: dict[str, np.ndarray],
    detector_name: str,
    max_features: int,
    ratio: float,
    frame_step: int,
) -> dict[str, Any]:
    detector_name = str(detector_name).lower()
    if detector_name == "sift" and hasattr(cv2, "SIFT_create"):
        detector, norm = cv2.SIFT_create(nfeatures=int(max_features)), cv2.NORM_L2
    else:
        detector, norm = cv2.ORB_create(nfeatures=int(max_features), fastThreshold=12), cv2.NORM_HAMMING
    h, w = video.shape[1:3]
    focal = float(max(h, w))
    camera_matrix = np.asarray([[focal, 0.0, 0.5 * (w - 1)], [0.0, focal, 0.5 * (h - 1)], [0.0, 0.0, 1.0]])
    frame_ids = _frame_ids_for(video.shape[0], frame_step)
    out: dict[str, Any] = {}
    for name, static_mask in static_masks.items():
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
        total_features = int(sum(item["num_features"] for item in frame_metrics))
        total_matches = int(sum(item["matches"] for item in pair_metrics))
        total_pairs = int(len(pair_metrics))
        success_pairs = int(sum(1 for item in pair_metrics if bool(item["essential_success"])))
        contam_num = int(sum(item["contaminated_features"] for item in frame_metrics))
        out[name] = {
            "total_features": total_features,
            "features_per_frame_mean": _safe_mean([float(item["num_features"]) for item in frame_metrics]),
            "feature_contamination": float(contam_num) / float(max(1, total_features)),
            "total_matches": total_matches,
            "matches_per_pair_mean": _safe_mean([float(item["matches"]) for item in pair_metrics]),
            "match_contamination_mean": _safe_mean([float(item["match_contamination"]) for item in pair_metrics]),
            "essential_success_rate": float(success_pairs) / float(max(1, total_pairs)),
            "essential_inliers_per_pair_mean": _safe_mean([float(item["essential_inliers"]) for item in pair_metrics]),
            "static_mask_fraction_mean": _safe_mean([float(item["static_mask_fraction"]) for item in frame_metrics]),
        }
    return out


def _aqua_maps_for_long_video(
    *,
    model: torch.nn.Module,
    video: np.ndarray,
    manifest: dict[str, Any],
    device: torch.device,
    window_size: int,
    grid_stride: int,
    query_chunk_size: int,
    dynamic_threshold: float,
    particle_threshold: float,
    static_threshold: float,
    static_score_mode: str,
) -> dict[str, Any]:
    all_score: dict[str, list[np.ndarray]] = {
        "dynamic_prob": [],
        "particle_prob": [],
        "confidence_prob": [],
        "static_confidence": [],
    }
    rejected_parts: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    for start in range(0, video.shape[0], max(1, int(window_size))):
        stop = min(video.shape[0], start + max(1, int(window_size)))
        sub_manifest = dict(manifest)
        sub_manifest["frames"] = manifest["frames"][start:stop]
        maps = aqua_dense_score_maps(
            model=model,
            video=video[start:stop],
            manifest=sub_manifest,
            device=device,
            grid_stride=int(grid_stride),
            query_chunk_size=int(query_chunk_size),
            dynamic_threshold=float(dynamic_threshold),
            particle_threshold=float(particle_threshold),
            static_thresholds=[float(static_threshold)],
            static_score_mode=str(static_score_mode),
        )
        for key in all_score:
            all_score[key].append(maps["score_maps"][key])
        rejected_name = f"aqua_static_conf_ge_{float(static_threshold):.3f}".replace(".", "p")
        rejected_parts.append(maps["rejected_masks"][rejected_name])
        metas.append(maps["meta"])
    score_maps = {key: np.concatenate(parts, axis=0) for key, parts in all_score.items()}
    rejected_mask = np.concatenate(rejected_parts, axis=0)
    return {
        "score_maps": score_maps,
        "rejected_mask": rejected_mask,
        "meta": {
            "window_size": int(window_size),
            "num_windows": int(len(metas)),
            "window_meta": metas,
            "dynamic_threshold": float(dynamic_threshold),
            "particle_threshold": float(particle_threshold),
            "static_threshold": float(static_threshold),
            "static_confidence_mean": float(np.mean(score_maps["static_confidence"])),
            "dynamic_prob_mean": float(np.mean(score_maps["dynamic_prob"])),
            "particle_prob_mean": float(np.mean(score_maps["particle_prob"])),
        },
    }


def _adaptive_retention_choice(
    *,
    aqua: dict[str, Any],
    aqua_mask: np.ndarray,
    learned_retain_mask: np.ndarray,
    pose_soft_retain_mask: np.ndarray,
    pose_soft_weight: np.ndarray,
    keypoint_context: dict[str, Any],
    dynamic_threshold: float,
    particle_threshold: float,
    particle_coverage_high: float,
    rejected_fraction_high: float,
    retained_fraction_low: float,
    model_pair_rate_low: float,
) -> dict[str, Any]:
    """Choose hard or pose-soft retention from deployable sequence statistics."""

    score_maps = aqua["score_maps"]
    dynamic_coverage = float(np.mean(score_maps["dynamic_prob"] >= float(dynamic_threshold)))
    particle_coverage = float(np.mean(score_maps["particle_prob"] >= float(particle_threshold)))
    rejected_fraction = float(np.mean(aqua_mask))
    learned_retained_fraction = float(np.mean(learned_retain_mask))
    pose_soft_retained_fraction = float(np.mean(pose_soft_retain_mask))
    if bool(aqua_mask.any()):
        pose_soft_weight_mean_rejected = float(np.mean(pose_soft_weight[aqua_mask]))
    else:
        pose_soft_weight_mean_rejected = 0.0
    num_pairs = int(keypoint_context.get("num_pairs", 0))
    num_pairs_with_model = int(keypoint_context.get("num_pairs_with_model", 0))
    model_pair_rate = float(num_pairs_with_model) / float(max(1, num_pairs))

    reasons: list[str] = []
    choose_pose_soft = False
    if particle_coverage >= float(particle_coverage_high):
        choose_pose_soft = True
        reasons.append("particle_coverage_high")
    if rejected_fraction >= float(rejected_fraction_high):
        choose_pose_soft = True
        reasons.append("rejected_fraction_high")
    if learned_retained_fraction <= float(retained_fraction_low):
        choose_pose_soft = True
        reasons.append("learned_retained_fraction_low")
    if model_pair_rate <= float(model_pair_rate_low):
        choose_pose_soft = True
        reasons.append("raw_model_pair_rate_low")
    if not reasons:
        reasons.append("hard_retention_sufficient")

    chosen = "pose_soft" if choose_pose_soft else "hard"
    return {
        "chosen": chosen,
        "reasons": reasons,
        "dynamic_coverage": dynamic_coverage,
        "particle_coverage": particle_coverage,
        "rejected_fraction": rejected_fraction,
        "learned_retained_fraction": learned_retained_fraction,
        "pose_soft_retained_fraction": pose_soft_retained_fraction,
        "pose_soft_weight_mean_rejected": pose_soft_weight_mean_rejected,
        "raw_model_pair_rate": model_pair_rate,
        "num_pairs": num_pairs,
        "num_pairs_with_model": num_pairs_with_model,
        "thresholds": {
            "particle_coverage_high": float(particle_coverage_high),
            "rejected_fraction_high": float(rejected_fraction_high),
            "retained_fraction_low": float(retained_fraction_low),
            "model_pair_rate_low": float(model_pair_rate_low),
        },
    }


def _evaluate_manifest(
    *,
    manifest_path: Path,
    model: torch.nn.Module,
    retention_scorer: dict[str, Any] | None,
    image_hw: tuple[int, int] | None,
    device: torch.device,
    output_dir: Path,
    max_frames: int,
    frame_stride: int,
    aqua_window_size: int,
    aqua_grid_stride: int,
    query_chunk_size: int,
    dynamic_threshold: float,
    particle_threshold: float,
    static_threshold: float,
    inpaint_radius: float,
    max_image_size: int,
    max_num_features: int,
    pycolmap_gpu: bool,
    max_runtime_seconds: float,
    include_temporal_rgb: bool,
    include_oracle: bool,
    enable_slam_aware_retention: bool,
    enable_learned_retention: bool,
    enable_soft_learned_retention: bool,
    enable_pose_aware_soft_retention: bool,
    enable_adaptive_retention: bool,
    retention_score_threshold: float,
    retention_detector: str,
    retention_patch_radius: int,
    retention_min_inlier_support: int,
    retention_max_features_per_frame: int,
    retention_max_fraction: float,
    adaptive_particle_coverage_high: float,
    adaptive_rejected_fraction_high: float,
    adaptive_retained_fraction_low: float,
    adaptive_model_pair_rate_low: float,
    pose_soft_min_weight: float,
    pose_soft_max_weight: float,
    pose_soft_score_power: float,
    pose_soft_geometry_power: float,
    frontend_detector: str,
    frontend_max_features: int,
    frontend_ratio: float,
    frontend_frame_step: int,
    pycolmap_random_seeds: list[int],
    fixed_initial_pair: str | None,
    fallback_initial_pair: bool,
    static_score_mode: str,
    variant_filter: list[str] | None,
) -> dict[str, Any]:
    video, dynamic_mask, particle_mask, manifest, pose_rows = _load_pose_clip(
        manifest_path,
        image_hw=image_hw,
        max_frames=int(max_frames),
    )
    transient_gt = (dynamic_mask | particle_mask).astype(bool)
    aqua = _aqua_maps_for_long_video(
        model=model,
        video=video,
        manifest=manifest,
        device=device,
        window_size=int(aqua_window_size),
        grid_stride=int(aqua_grid_stride),
        query_chunk_size=int(query_chunk_size),
        dynamic_threshold=float(dynamic_threshold),
        particle_threshold=float(particle_threshold),
        static_threshold=float(static_threshold),
        static_score_mode=str(static_score_mode),
    )
    aqua_mask = aqua["rejected_mask"]
    variants: dict[str, np.ndarray] = {"raw": video}
    static_masks: dict[str, np.ndarray] = {"raw": np.ones(transient_gt.shape, dtype=bool)}
    temporal_mask = np.zeros(transient_gt.shape, dtype=bool)
    if include_temporal_rgb:
        temporal_mask = temporal_rgb_pseudo_mask(video).astype(bool)
        variants["temporal_rgb_inpaint"] = inpaint_video_with_mask(video, temporal_mask, radius=float(inpaint_radius))
        static_masks["temporal_rgb_inpaint"] = ~temporal_mask
    variants["aqua_inpaint"] = inpaint_video_with_mask(video, aqua_mask, radius=float(inpaint_radius))
    static_masks["aqua_inpaint"] = ~aqua_mask
    if include_oracle:
        variants["oracle_gt_inpaint"] = inpaint_video_with_mask(video, transient_gt, radius=float(inpaint_radius))
        static_masks["oracle_gt_inpaint"] = ~transient_gt

    retention_meta: dict[str, Any] = {}
    if bool(enable_slam_aware_retention):
        retain_mask, meta = _slam_aware_retention_mask(
            video_rgb=video,
            rejected_mask=aqua_mask,
            detector_name=str(retention_detector),
            max_features=max(1200, int(max_num_features)),
            ratio=0.75,
            frame_step=max(1, int(frame_stride)),
            patch_radius=int(retention_patch_radius),
            min_inlier_support=int(retention_min_inlier_support),
            max_retained_features_per_frame=int(retention_max_features_per_frame),
            max_retained_fraction=float(retention_max_fraction),
        )
        retained_aqua_mask = aqua_mask & ~retain_mask
        variants["aqua_slam_retain_inpaint"] = inpaint_video_with_mask(
            video,
            retained_aqua_mask,
            radius=float(inpaint_radius),
        )
        static_masks["aqua_slam_retain_inpaint"] = ~retained_aqua_mask
        retention_meta["aqua_slam_retain_inpaint"] = meta

    learned_retain_mask = np.zeros(aqua_mask.shape, dtype=bool)
    pose_soft_retain_mask = np.zeros(aqua_mask.shape, dtype=bool)
    pose_soft_weight = np.zeros(aqua_mask.shape, dtype=np.float32)
    if bool(enable_learned_retention) and retention_scorer is not None:
        scorer_feature_names = [str(name) for name in retention_scorer.get("feature_names", [])]
        needs_gt_pose_features = any(name.startswith("gt_pose_") for name in scorer_feature_names)
        keypoint_context = extract_keypoint_context(
            video_rgb=video,
            detector_name=str(retention_detector),
            max_features=max(1200, int(max_num_features)),
            ratio=0.75,
            frame_step=max(1, int(frame_stride)),
            pose_rows=pose_rows if needs_gt_pose_features else None,
        )
        if needs_gt_pose_features:
            table = build_pose_retention_candidate_table(
                video_rgb=video,
                transient_gt=transient_gt,
                score_maps=aqua["score_maps"],
                rejected_mask=aqua_mask,
                keypoint_context=keypoint_context,
                dynamic_threshold=float(dynamic_threshold),
                particle_threshold=float(particle_threshold),
                static_threshold=float(static_threshold),
                min_positive_inlier_support=int(retention_min_inlier_support),
                min_positive_pose_inlier_support=1,
                pose_sampson_threshold_px=2.0,
                clip_index=0,
                include_pose_features=True,
            )
        else:
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
                clip_index=0,
            )
        scores = score_retention_candidates(
            table["features"],
            retention_scorer,
            device=device,
            feature_names=table["feature_names"],
        )
        learned_retain_mask, learned_meta = retention_mask_from_candidates(
            candidate_meta=table["candidate_meta"],
            scores=scores,
            rejected_mask=aqua_mask,
            score_threshold=float(retention_score_threshold),
            patch_radius=int(retention_patch_radius),
            max_features_per_frame=int(retention_max_features_per_frame),
            max_fraction=float(retention_max_fraction),
        )
        learned_meta.update(table["summary"])
        retained_aqua_mask = aqua_mask & ~learned_retain_mask
        learned_name = f"aqua_learned_retain_t{float(retention_score_threshold):.2f}_inpaint".replace(".", "p")
        variants[learned_name] = inpaint_video_with_mask(
            video,
            retained_aqua_mask,
            radius=float(inpaint_radius),
        )
        static_masks[learned_name] = ~retained_aqua_mask
        retention_meta[learned_name] = learned_meta
        if bool(enable_soft_learned_retention):
            soft_name = f"aqua_learned_retain_t{float(retention_score_threshold):.2f}_soft".replace(".", "p")
            variants[soft_name] = soft_temporal_fill_video(
                video,
                aqua_mask,
                static_confidence=aqua["score_maps"]["static_confidence"],
                retain_mask=learned_retain_mask,
                temporal_radius=2,
                blur_kernel=5,
            )
            static_masks[soft_name] = (~aqua_mask) | learned_retain_mask
        if bool(enable_pose_aware_soft_retention):
            pose_soft_weight, pose_soft_retain_mask, pose_soft_meta = pose_aware_retention_weight_from_candidates(
                candidate_meta=table["candidate_meta"],
                features=table["features"],
                scores=scores,
                rejected_mask=aqua_mask,
                score_threshold=float(retention_score_threshold),
                patch_radius=int(retention_patch_radius),
                max_features_per_frame=int(retention_max_features_per_frame),
                max_fraction=float(retention_max_fraction),
                score_power=float(pose_soft_score_power),
                geometry_power=float(pose_soft_geometry_power),
                min_weight=float(pose_soft_min_weight),
                max_weight=float(pose_soft_max_weight),
            )
            pose_soft_meta.update(table["summary"])
            pose_soft_name = f"aqua_pose_soft_t{float(retention_score_threshold):.2f}".replace(".", "p")
            variants[pose_soft_name] = soft_temporal_fill_video(
                video,
                aqua_mask,
                static_confidence=aqua["score_maps"]["static_confidence"],
                retain_mask=None,
                retain_weight=pose_soft_weight,
                temporal_radius=2,
                blur_kernel=5,
            )
            static_masks[pose_soft_name] = (~aqua_mask) | pose_soft_retain_mask
            retention_meta[pose_soft_name] = pose_soft_meta

        if bool(enable_adaptive_retention) and bool(enable_pose_aware_soft_retention):
            if pose_soft_retain_mask.size == 0 or pose_soft_weight.size == 0:
                raise RuntimeError("Adaptive retention requires pose-aware soft retention outputs.")
            adaptive = _adaptive_retention_choice(
                aqua=aqua,
                aqua_mask=aqua_mask,
                learned_retain_mask=learned_retain_mask,
                pose_soft_retain_mask=pose_soft_retain_mask,
                pose_soft_weight=pose_soft_weight,
                keypoint_context=keypoint_context,
                dynamic_threshold=float(dynamic_threshold),
                particle_threshold=float(particle_threshold),
                particle_coverage_high=float(adaptive_particle_coverage_high),
                rejected_fraction_high=float(adaptive_rejected_fraction_high),
                retained_fraction_low=float(adaptive_retained_fraction_low),
                model_pair_rate_low=float(adaptive_model_pair_rate_low),
            )
            adaptive_name = f"aqua_adaptive_retain_t{float(retention_score_threshold):.2f}".replace(".", "p")
            if adaptive["chosen"] == "pose_soft":
                variants[adaptive_name] = variants[pose_soft_name]
                static_masks[adaptive_name] = static_masks[pose_soft_name]
                source_name = pose_soft_name
            else:
                variants[adaptive_name] = variants[learned_name]
                static_masks[adaptive_name] = static_masks[learned_name]
                source_name = learned_name
            retention_meta[adaptive_name] = {
                "adaptive_choice": adaptive,
                "source_variant": source_name,
                "label": "deployable sequence-level hard/pose-soft retention selector; no GT pose errors used",
            }

    if variant_filter:
        patterns = [str(item) for item in variant_filter if str(item).strip()]

        def _keep_variant(name: str) -> bool:
            return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)

        variants = {name: value for name, value in variants.items() if _keep_variant(name)}
        static_masks = {name: value for name, value in static_masks.items() if _keep_variant(name)}
        retention_meta = {name: value for name, value in retention_meta.items() if _keep_variant(name)}
        if not variants:
            raise ValueError(f"--variant-filter matched no variants. Available patterns: {patterns}")

    clip_name = str(manifest.get("name", manifest_path.parent.name))
    clip_dir = output_dir / clip_name
    variant_results: dict[str, Any] = {}
    for name, variant_video in variants.items():
        for seed in pycolmap_random_seeds:
            result_name = name if len(pycolmap_random_seeds) == 1 else f"{name}__seed{int(seed)}"
            print(f"  pyCOLMAP variant: {result_name}", flush=True)
            try:
                result = _run_pycolmap_variant(
                    video_rgb=variant_video,
                    variant_dir=clip_dir / result_name,
                    frame_stride=int(frame_stride),
                    max_image_size=int(max_image_size),
                    max_num_features=int(max_num_features),
                    use_gpu=bool(pycolmap_gpu),
                    max_runtime_seconds=float(max_runtime_seconds),
                    random_seed=int(seed),
                    fixed_initial_pair=fixed_initial_pair,
                    fallback_initial_pair=bool(fallback_initial_pair),
                )
            except Exception as exc:
                result = {
                    "success": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                    "pycolmap_random_seed": int(seed),
                    "fixed_initial_pair": fixed_initial_pair,
                    "image_count": len(_frame_ids_for(video.shape[0], int(frame_stride))),
                    "best_reconstruction": {
                        "num_images": len(_frame_ids_for(video.shape[0], int(frame_stride))),
                        "num_reg_images": 0,
                        "registration_rate": 0.0,
                        "num_points3D": 0,
                        "num_observations": 0,
                        "mean_observations_per_reg_image": 0.0,
                        "mean_track_length": 0.0,
                        "mean_reprojection_error": 0.0,
                    },
                }
            pose_metrics = _pose_metrics_for_variant(
                variant_result=result,
                pose_rows=pose_rows,
                frame_stride=int(frame_stride),
            )
            image_count = int(result.get("image_count", len(_frame_ids_for(video.shape[0], int(frame_stride)))))
            num_reg_images = int(result.get("best_reconstruction", {}).get("num_reg_images", 0))
            result["input_registration_rate"] = float(num_reg_images) / float(max(1, image_count))
            result["pose_metrics"] = pose_metrics
            result["base_variant"] = name
            result["variant_seed_name"] = result_name
            variant_results[result_name] = result

    frontend_metrics = _variant_frontend_metrics(
        video=video,
        transient_gt=transient_gt,
        static_masks=static_masks,
        detector_name=str(frontend_detector),
        max_features=int(frontend_max_features),
        ratio=float(frontend_ratio),
        frame_step=int(frontend_frame_step),
    )
    for name, metrics in frontend_metrics.items():
        for result_name, result in variant_results.items():
            if result.get("base_variant", result_name) == name:
                result["frontend_metrics"] = metrics

    return {
        "manifest": str(manifest_path.resolve()),
        "clip_name": clip_name,
        "dataset": str(manifest.get("dataset", "unknown")),
        "variant": str(manifest.get("variant", clip_name)),
        "num_frames": int(video.shape[0]),
        "image_hw": [int(video.shape[1]), int(video.shape[2])],
        "frame_stride": int(frame_stride),
        "mask_coverage": {
            "dynamic_object": float(dynamic_mask.mean()),
            "particle": float(particle_mask.mean()),
            "transient": float(transient_gt.mean()),
            "aqua_rejected": float(aqua_mask.mean()),
            "temporal_rgb": float(temporal_mask.mean()),
            "learned_retained": float(learned_retain_mask.mean()),
            "pose_soft_retained": float(pose_soft_retain_mask.mean()),
            "pose_soft_mean_weight": float(pose_soft_weight[pose_soft_retain_mask].mean()) if bool(pose_soft_retain_mask.any()) else 0.0,
        },
        "pycolmap_random_seeds": [int(v) for v in pycolmap_random_seeds],
        "fixed_initial_pair": fixed_initial_pair,
        "fallback_initial_pair": bool(fallback_initial_pair),
        "aqua_meta": aqua["meta"],
        "retention_meta": retention_meta,
        "variants": variant_results,
    }


def _mean_optional(values: list[Any]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def _aggregate(per_clip: list[dict[str, Any]]) -> dict[str, Any]:
    variants: list[str] = []
    for clip in per_clip:
        for variant in clip["variants"]:
            if variant not in variants:
                variants.append(variant)
    out: dict[str, Any] = {"num_clips": len(per_clip), "variants": {}}
    for variant in variants:
        records = [clip["variants"][variant] for clip in per_clip if variant in clip["variants"]]
        recon = [record["best_reconstruction"] for record in records]
        pose = [record.get("pose_metrics", {}) for record in records]
        front = [record.get("frontend_metrics", {}) for record in records if record.get("frontend_metrics")]
        out["variants"][variant] = {
            "clips": len(records),
            "success_rate": float(np.mean([bool(record.get("success", False)) for record in records])) if records else 0.0,
            "pose_eval_success_rate": float(np.mean([bool(item.get("pose_eval_success", False)) for item in pose])) if pose else 0.0,
            "num_reg_images_mean": float(np.mean([r["num_reg_images"] for r in recon])) if recon else 0.0,
            "input_registration_rate_mean": float(np.mean([record.get("input_registration_rate", 0.0) for record in records])) if records else 0.0,
            "model_registration_rate_mean": float(np.mean([r["registration_rate"] for r in recon])) if recon else 0.0,
            "num_points3D_mean": float(np.mean([r["num_points3D"] for r in recon])) if recon else 0.0,
            "mean_track_length_mean": float(np.mean([r["mean_track_length"] for r in recon])) if recon else 0.0,
            "mean_reprojection_error_mean": float(np.mean([r["mean_reprojection_error"] for r in recon])) if recon else 0.0,
            "ate_rmse_mean": _mean_optional([item.get("ate_rmse") for item in pose]),
            "ate_mean_mean": _mean_optional([item.get("ate_mean") for item in pose]),
            "rpe_trans_rmse_mean": _mean_optional([item.get("rpe_trans_rmse") for item in pose]),
            "orientation_rpe_deg_mean": _mean_optional([item.get("orientation_rpe_deg_mean") for item in pose]),
            "feature_contamination_mean": _mean_optional([item.get("feature_contamination") for item in front]),
            "match_contamination_mean": _mean_optional([item.get("match_contamination_mean") for item in front]),
            "essential_success_rate_mean": _mean_optional([item.get("essential_success_rate") for item in front]),
            "features_per_frame_mean": _mean_optional([item.get("features_per_frame_mean") for item in front]),
        }
    out["mean_mask_coverage"] = {
        key: _safe_mean([float(clip["mask_coverage"][key]) for clip in per_clip])
        for key in (
            "dynamic_object",
            "particle",
            "transient",
            "aqua_rejected",
            "temporal_rgb",
            "learned_retained",
            "pose_soft_retained",
            "pose_soft_mean_weight",
        )
    }
    return out


def _aggregate_by_stress_variant(per_clip: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for clip in per_clip:
        stress_name = str(clip.get("variant", clip.get("clip_name", "unknown")))
        groups.setdefault(stress_name, []).append(clip)
    return {name: _aggregate(records) for name, records in sorted(groups.items())}


def _base_variant_name(name: str) -> str:
    if "__seed" in str(name):
        return str(name).split("__seed", 1)[0]
    return str(name)


def _std_optional(values: list[Any]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    return float(np.std(valid, ddof=0)) if valid else None


def _aggregate_seed_stability(per_clip: list[dict[str, Any]]) -> dict[str, Any]:
    records_by_base: dict[str, list[dict[str, Any]]] = {}
    for clip in per_clip:
        for name, record in clip["variants"].items():
            base = str(record.get("base_variant", _base_variant_name(name)))
            records_by_base.setdefault(base, []).append(record)
    out: dict[str, Any] = {}
    metric_getters = {
        "success_rate": lambda r: float(bool(r.get("success", False))),
        "pose_eval_success_rate": lambda r: float(bool(r.get("pose_metrics", {}).get("pose_eval_success", False))),
        "input_registration_rate": lambda r: float(r.get("input_registration_rate", 0.0)),
        "num_points3D": lambda r: float(r.get("best_reconstruction", {}).get("num_points3D", 0.0)),
        "ate_rmse": lambda r: r.get("pose_metrics", {}).get("ate_rmse"),
        "rpe_trans_rmse": lambda r: r.get("pose_metrics", {}).get("rpe_trans_rmse"),
        "feature_contamination": lambda r: r.get("frontend_metrics", {}).get("feature_contamination"),
        "match_contamination": lambda r: r.get("frontend_metrics", {}).get("match_contamination_mean"),
    }
    for base, records in sorted(records_by_base.items()):
        item: dict[str, Any] = {"records": int(len(records))}
        for metric, getter in metric_getters.items():
            values = [getter(record) for record in records]
            item[f"{metric}_mean"] = _mean_optional(values)
            item[f"{metric}_std"] = _std_optional(values)
        out[base] = item
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


def _write_summary_by_stress_csv(path: Path, aggregate_by_stress: dict[str, Any]) -> None:
    rows = []
    for stress_variant, aggregate in aggregate_by_stress.items():
        for system_variant, metrics in aggregate.get("variants", {}).items():
            row = {"stress_variant": stress_variant, "system_variant": system_variant}
            row.update(metrics)
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()) if rows else ["stress_variant", "system_variant"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_seed_stability_csv(path: Path, aggregate_seed_stability: dict[str, Any]) -> None:
    rows = []
    for base_variant, metrics in aggregate_seed_stability.items():
        row = {"base_variant": base_variant}
        row.update(metrics)
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["base_variant"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=None)
    parser.add_argument("--manifest-list", action="append", default=None)
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--retention-scorer-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--aqua-window-size", type=int, default=32)
    parser.add_argument("--aqua-grid-stride", type=int, default=4)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--dynamic-threshold", type=float, default=0.79)
    parser.add_argument("--particle-threshold", type=float, default=0.83)
    parser.add_argument("--static-threshold", type=float, default=0.55)
    parser.add_argument("--static-score-mode", default="full", choices=("full", "no_dynamic", "no_particle", "confidence_only"))
    parser.add_argument("--inpaint-radius", type=float, default=3.0)
    parser.add_argument("--max-image-size", type=int, default=1024)
    parser.add_argument("--max-num-features", type=int, default=4096)
    parser.add_argument("--pycolmap-gpu", action="store_true")
    parser.add_argument("--pycolmap-random-seeds", default=None)
    parser.add_argument("--fixed-initial-pair", default=None, help="'auto' or comma-separated output frame indices, e.g. 0,8")
    parser.add_argument("--no-initial-pair-fallback", action="store_true")
    parser.add_argument("--variant-filter", action="append", default=None, help="Only run matching variant names; supports glob syntax and can be repeated.")
    parser.add_argument("--max-runtime-seconds", type=float, default=60.0)
    parser.add_argument("--no-temporal-rgb", action="store_true")
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--enable-slam-aware-retention", action="store_true")
    parser.add_argument("--enable-learned-retention", action="store_true")
    parser.add_argument("--enable-soft-learned-retention", action="store_true")
    parser.add_argument("--enable-pose-aware-soft-retention", action="store_true")
    parser.add_argument("--enable-adaptive-retention", action="store_true")
    parser.add_argument("--retention-score-threshold", type=float, default=0.5)
    parser.add_argument("--retention-detector", default="orb", choices=("orb", "sift"))
    parser.add_argument("--retention-patch-radius", type=int, default=5)
    parser.add_argument("--retention-min-inlier-support", type=int, default=1)
    parser.add_argument("--retention-max-features-per-frame", type=int, default=300)
    parser.add_argument("--retention-max-fraction", type=float, default=0.18)
    parser.add_argument("--adaptive-particle-coverage-high", type=float, default=0.025)
    parser.add_argument("--adaptive-rejected-fraction-high", type=float, default=0.42)
    parser.add_argument("--adaptive-retained-fraction-low", type=float, default=0.015)
    parser.add_argument("--adaptive-model-pair-rate-low", type=float, default=0.35)
    parser.add_argument("--pose-soft-min-weight", type=float, default=0.10)
    parser.add_argument("--pose-soft-max-weight", type=float, default=0.88)
    parser.add_argument("--pose-soft-score-power", type=float, default=0.75)
    parser.add_argument("--pose-soft-geometry-power", type=float, default=1.25)
    parser.add_argument("--frontend-detector", default="orb", choices=("orb", "sift"))
    parser.add_argument("--frontend-max-features", type=int, default=1200)
    parser.add_argument("--frontend-ratio", type=float, default=0.75)
    parser.add_argument("--frontend-frame-step", type=int, default=2)
    parser.add_argument("--eval-image-height", type=int, default=0)
    parser.add_argument("--eval-image-width", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    if args.pycolmap_random_seeds:
        pycolmap_random_seeds = [int(part) for part in str(args.pycolmap_random_seeds).split(",") if part.strip()]
    else:
        pycolmap_random_seeds = [int(args.seed)]
    try:
        import pycolmap  # noqa: F401
    except Exception as exc:
        raise RuntimeError("pycolmap is required. Use /media/data/u24conda/envs/longlive/bin/python.") from exc
    manifests = _resolve_manifests(args)
    device = _resolve_device(args.device)
    cfg = load_yaml_config(args.model_config)
    if int(args.eval_image_height) > 0 and int(args.eval_image_width) > 0:
        image_hw: tuple[int, int] | None = (int(args.eval_image_height), int(args.eval_image_width))
    else:
        image_hw = tuple(int(v) for v in cfg["model"]["input"].get("image_size", [256, 256]))
    model = _load_model(Path(args.model_config), Path(args.ckpt_path), device=device)
    retention_scorer = (
        load_retention_scorer(args.retention_scorer_path, device=device)
        if args.retention_scorer_path
        else None
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_clip: list[dict[str, Any]] = []
    for idx, manifest_path in enumerate(manifests):
        print(f"[{idx + 1}/{len(manifests)}] {manifest_path}", flush=True)
        per_clip.append(
            _evaluate_manifest(
                manifest_path=manifest_path,
                model=model,
                retention_scorer=retention_scorer,
                image_hw=image_hw,
                device=device,
                output_dir=output_dir,
                max_frames=int(args.max_frames),
                frame_stride=int(args.frame_stride),
                aqua_window_size=int(args.aqua_window_size),
                aqua_grid_stride=int(args.aqua_grid_stride),
                query_chunk_size=int(args.query_chunk_size),
                dynamic_threshold=float(args.dynamic_threshold),
                particle_threshold=float(args.particle_threshold),
                static_threshold=float(args.static_threshold),
                static_score_mode=str(args.static_score_mode),
                inpaint_radius=float(args.inpaint_radius),
                max_image_size=int(args.max_image_size),
                max_num_features=int(args.max_num_features),
                pycolmap_gpu=bool(args.pycolmap_gpu),
                max_runtime_seconds=float(args.max_runtime_seconds),
                include_temporal_rgb=not bool(args.no_temporal_rgb),
                include_oracle=not bool(args.no_oracle),
                enable_slam_aware_retention=bool(args.enable_slam_aware_retention),
                enable_learned_retention=bool(args.enable_learned_retention),
                enable_soft_learned_retention=bool(args.enable_soft_learned_retention),
                enable_pose_aware_soft_retention=bool(args.enable_pose_aware_soft_retention),
                enable_adaptive_retention=bool(args.enable_adaptive_retention),
                retention_score_threshold=float(args.retention_score_threshold),
                retention_detector=str(args.retention_detector),
                retention_patch_radius=int(args.retention_patch_radius),
                retention_min_inlier_support=int(args.retention_min_inlier_support),
                retention_max_features_per_frame=int(args.retention_max_features_per_frame),
                retention_max_fraction=float(args.retention_max_fraction),
                adaptive_particle_coverage_high=float(args.adaptive_particle_coverage_high),
                adaptive_rejected_fraction_high=float(args.adaptive_rejected_fraction_high),
                adaptive_retained_fraction_low=float(args.adaptive_retained_fraction_low),
                adaptive_model_pair_rate_low=float(args.adaptive_model_pair_rate_low),
                pose_soft_min_weight=float(args.pose_soft_min_weight),
                pose_soft_max_weight=float(args.pose_soft_max_weight),
                pose_soft_score_power=float(args.pose_soft_score_power),
                pose_soft_geometry_power=float(args.pose_soft_geometry_power),
                frontend_detector=str(args.frontend_detector),
                frontend_max_features=int(args.frontend_max_features),
                frontend_ratio=float(args.frontend_ratio),
                frontend_frame_step=int(args.frontend_frame_step),
                pycolmap_random_seeds=pycolmap_random_seeds,
                fixed_initial_pair=args.fixed_initial_pair,
                fallback_initial_pair=not bool(args.no_initial_pair_fallback),
                variant_filter=args.variant_filter,
            )
        )

    aggregate = _aggregate(per_clip)
    aggregate_by_stress = _aggregate_by_stress_variant(per_clip)
    aggregate_seed_stability = _aggregate_seed_stability(per_clip)
    metadata = {
        "model_config": str(Path(args.model_config).resolve()),
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "retention_scorer_path": str(Path(args.retention_scorer_path).resolve()) if args.retention_scorer_path else None,
        "num_manifests": len(manifests),
        "image_hw": list(image_hw) if image_hw is not None else None,
        "frame_stride": int(args.frame_stride),
        "aqua_window_size": int(args.aqua_window_size),
        "aqua_grid_stride": int(args.aqua_grid_stride),
        "dynamic_threshold": float(args.dynamic_threshold),
        "particle_threshold": float(args.particle_threshold),
        "static_threshold": float(args.static_threshold),
        "static_score_mode": str(args.static_score_mode),
        "max_runtime_seconds": float(args.max_runtime_seconds),
        "pycolmap_random_seeds": [int(v) for v in pycolmap_random_seeds],
        "fixed_initial_pair": args.fixed_initial_pair,
        "initial_pair_fallback_enabled": not bool(args.no_initial_pair_fallback),
        "variant_filter": args.variant_filter,
        "pose_aware_soft_retention": {
            "enabled": bool(args.enable_pose_aware_soft_retention),
            "min_weight": float(args.pose_soft_min_weight),
            "max_weight": float(args.pose_soft_max_weight),
            "score_power": float(args.pose_soft_score_power),
            "geometry_power": float(args.pose_soft_geometry_power),
        },
        "adaptive_retention": {
            "enabled": bool(args.enable_adaptive_retention),
            "particle_coverage_high": float(args.adaptive_particle_coverage_high),
            "rejected_fraction_high": float(args.adaptive_rejected_fraction_high),
            "retained_fraction_low": float(args.adaptive_retained_fraction_low),
            "model_pair_rate_low": float(args.adaptive_model_pair_rate_low),
            "note": "Sequence-level hard/pose-soft selector from Aqua/ORB statistics; no GT pose-error oracle.",
        },
        "backend": "pycolmap",
        "colmap_cli_available": shutil.which("colmap") is not None,
        "pose_metric_note": "Translation ATE/RPE after Sim(3) alignment are primary; orientation RPE is secondary because COLMAP and GT camera-frame conventions may differ.",
    }
    (output_dir / "per_clip_metrics.json").write_text(json.dumps(per_clip, indent=2, default=_json_default), encoding="utf-8")
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "aggregate": aggregate,
                "aggregate_by_stress_variant": aggregate_by_stress,
                "aggregate_seed_stability": aggregate_seed_stability,
            },
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    _write_summary_csv(output_dir / "summary_table.csv", aggregate)
    _write_summary_by_stress_csv(output_dir / "summary_by_stress_variant.csv", aggregate_by_stress)
    _write_seed_stability_csv(output_dir / "summary_seed_stability.csv", aggregate_seed_stability)

    print("GT-pose pyCOLMAP summary:")
    for variant, metrics in aggregate["variants"].items():
        ate = metrics["ate_rmse_mean"]
        ate_s = "nan" if ate is None else f"{ate:.4f}"
        rpe = metrics["rpe_trans_rmse_mean"]
        rpe_s = "nan" if rpe is None else f"{rpe:.4f}"
        print(
            f"- {variant}: success={metrics['success_rate']:.3f} "
            f"input_reg={metrics['input_registration_rate_mean']:.3f} "
            f"ATE={ate_s} RPE={rpe_s} "
            f"feat_contam={metrics['feature_contamination_mean']}"
        )
    print(f"Saved: {output_dir / 'aggregate_metrics.json'}")
    return 0


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
