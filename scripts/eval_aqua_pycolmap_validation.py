#!/usr/bin/env python3
"""Run pyCOLMAP validation on raw and transient-filtered underwater clips."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
import shutil
import sqlite3
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

from aqua_prefilter_utils import inpaint_video_with_mask, soft_temporal_fill_video, temporal_rgb_pseudo_mask  # noqa: E402
from aqua_retention_utils import (  # noqa: E402
    aqua_dense_score_maps,
    build_retention_candidate_table,
    extract_keypoint_context,
    load_retention_scorer,
    pose_aware_retention_weight_from_candidates,
    retention_mask_from_candidates,
    score_retention_candidates,
)
from eval_aqua_downstream_slam_proxy import (  # noqa: E402
    _resize_mask_stack,
    _slam_aware_retention_mask,
)
from eval_aqua_transient_heads import _load_clip, _load_model  # noqa: E402
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


def _load_temporal_mask(manifest: dict[str, Any], video: np.ndarray, image_hw: tuple[int, int]) -> tuple[np.ndarray, str]:
    pseudo_path = manifest.get("pseudo_mask_npz")
    if pseudo_path and Path(str(pseudo_path)).exists():
        payload = np.load(str(pseudo_path))
        for key in ("pseudo_dynamic_mask", "dynamic_object_mask", "transient_mask"):
            if key in payload:
                return _resize_mask_stack(payload[key].astype(bool)[: video.shape[0]], image_hw), str(pseudo_path)
    return temporal_rgb_pseudo_mask(video).astype(bool), "computed_temporal_rgb"


def _write_video_frames(video_rgb: np.ndarray, image_dir: Path, frame_stride: int) -> list[str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_names: list[str] = []
    for out_idx, src_idx in enumerate(range(0, video_rgb.shape[0], max(1, int(frame_stride)))):
        name = f"frame_{out_idx:04d}.png"
        bgr = cv2.cvtColor(video_rgb[src_idx], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_dir / name), bgr)
        image_names.append(name)
    return image_names


def _summarize_reconstruction(recon: Any) -> dict[str, Any]:
    num_reg_images = int(recon.num_reg_images())
    num_points3d = int(recon.num_points3D())
    num_obs = int(recon.compute_num_observations()) if num_points3d > 0 else 0
    return {
        "summary": str(recon.summary()),
        "num_images": int(recon.num_images()),
        "num_reg_images": num_reg_images,
        "registration_rate": float(num_reg_images) / float(max(1, int(recon.num_images()))),
        "num_points3D": num_points3d,
        "num_observations": num_obs,
        "mean_observations_per_reg_image": float(recon.compute_mean_observations_per_reg_image()) if num_reg_images > 0 else 0.0,
        "mean_track_length": float(recon.compute_mean_track_length()) if num_points3d > 0 else 0.0,
        "mean_reprojection_error": float(recon.compute_mean_reprojection_error()) if num_points3d > 0 else 0.0,
    }


def _pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    max_image_id = 2_147_483_647
    image_id2 = int(pair_id) % max_image_id
    image_id1 = (int(pair_id) - image_id2) // max_image_id
    return int(image_id1), int(image_id2)


def _frame_index_from_image_name(name: str) -> int | None:
    match = re.search(r"frame_(\d+)\.", str(name))
    if match is None:
        return None
    return int(match.group(1))


def _image_id_for_frame_name(database_path: Path, image_name: str) -> int | None:
    import pycolmap

    database = pycolmap.Database.open(str(database_path))
    try:
        image = database.read_image_with_name(str(image_name))
        return int(image.image_id)
    except Exception:
        return None
    finally:
        database.close()


def _best_verified_pair(database_path: Path) -> tuple[int, int, dict[str, Any]] | None:
    with sqlite3.connect(str(database_path)) as connection:
        image_names = {
            int(image_id): str(name)
            for image_id, name in connection.execute("SELECT image_id, name FROM images")
        }
        rows = list(
            connection.execute(
                "SELECT pair_id, rows, config FROM two_view_geometries WHERE rows > 0"
            )
        )

    best: tuple[float, int, int, int, int, str, str, int, int] | None = None
    candidates_seen = 0
    min_inliers = 30
    min_gap = 2
    target_gap = 8
    for pair_id, num_inliers, config in rows:
        num_inliers = int(num_inliers)
        if num_inliers < min_inliers:
            continue
        image_id1, image_id2 = _pair_id_to_image_ids(int(pair_id))
        name1 = image_names.get(int(image_id1), "")
        name2 = image_names.get(int(image_id2), "")
        frame_idx1 = _frame_index_from_image_name(name1)
        frame_idx2 = _frame_index_from_image_name(name2)
        if frame_idx1 is not None and frame_idx2 is not None:
            gap = abs(int(frame_idx2) - int(frame_idx1))
        else:
            gap = abs(int(image_id2) - int(image_id1))
        if gap < min_gap:
            continue
        candidates_seen += 1
        gap_score = min(int(gap), target_gap)
        score = float(num_inliers * gap_score)
        candidate = (
            score,
            int(gap),
            num_inliers,
            int(image_id1),
            int(image_id2),
            name1,
            name2,
            -1 if frame_idx1 is None else int(frame_idx1),
            -1 if frame_idx2 is None else int(frame_idx2),
        )
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    score, gap, num_inliers, image_id1, image_id2, name1, name2, frame_idx1, frame_idx2 = best
    return image_id1, image_id2, {
        "source": "auto_verified_baseline_pair",
        "score": float(score),
        "num_inliers": int(num_inliers),
        "frame_gap": int(gap),
        "image_name1": name1,
        "image_name2": name2,
        "frame_idx1": frame_idx1,
        "frame_idx2": frame_idx2,
        "min_inliers": int(min_inliers),
        "min_gap": int(min_gap),
        "target_gap": int(target_gap),
        "candidates_seen": int(candidates_seen),
    }


def _resolve_initial_pair(
    *,
    database_path: Path,
    fixed_initial_pair: str | None,
) -> tuple[int, int, dict[str, Any]] | None:
    if fixed_initial_pair is None or str(fixed_initial_pair).strip().lower() in {"", "none", "off"}:
        return None
    spec = str(fixed_initial_pair).strip().lower()
    if spec == "auto":
        return _best_verified_pair(database_path)
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("--fixed-initial-pair must be 'auto' or 'i,j'")
    try:
        idx1, idx2 = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--fixed-initial-pair indices must be integers") from exc
    image_id1 = _image_id_for_frame_name(database_path, f"frame_{idx1:04d}.png")
    image_id2 = _image_id_for_frame_name(database_path, f"frame_{idx2:04d}.png")
    if image_id1 is None or image_id2 is None:
        raise ValueError(f"Could not resolve fixed initial pair frames: {idx1},{idx2}")
    return int(image_id1), int(image_id2), {"source": "frame_indices", "frame_idx1": int(idx1), "frame_idx2": int(idx2)}


def _run_pycolmap_variant(
    *,
    video_rgb: np.ndarray,
    variant_dir: Path,
    frame_stride: int,
    max_image_size: int,
    max_num_features: int,
    use_gpu: bool,
    max_runtime_seconds: float,
    random_seed: int = 42,
    fixed_initial_pair: str | None = None,
    fallback_initial_pair: bool = True,
) -> dict[str, Any]:
    import pycolmap

    if variant_dir.exists():
        shutil.rmtree(variant_dir)
    images_dir = variant_dir / "images"
    database_path = variant_dir / "database.db"
    sparse_dir = variant_dir / "sparse"
    image_names = _write_video_frames(video_rgb, images_dir, frame_stride=frame_stride)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "SIMPLE_PINHOLE"
    reader_options.default_focal_length_factor = 1.2

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.use_gpu = bool(use_gpu)
    extraction_options.max_image_size = int(max_image_size)
    extraction_options.num_threads = 4
    extraction_options.sift.max_num_features = int(max_num_features)
    pycolmap.extract_features(
        database_path,
        images_dir,
        image_names=image_names,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=extraction_options,
    )

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.use_gpu = bool(use_gpu)
    matching_options.num_threads = 4
    matching_options.sift.max_ratio = 0.8
    matching_options.sift.cross_check = True
    pycolmap.match_exhaustive(database_path, matching_options=matching_options)

    initial_pair = _resolve_initial_pair(database_path=database_path, fixed_initial_pair=fixed_initial_pair)
    initial_pair_meta: dict[str, Any] | None = None

    def _make_options(
        pair: tuple[int, int, dict[str, Any]] | None,
        *,
        runtime_seconds: float,
    ) -> tuple[Any, dict[str, Any] | None]:
        options = pycolmap.IncrementalPipelineOptions()
        options.multiple_models = False
        options.max_num_models = 1
        options.min_model_size = 3
        options.min_num_matches = 8
        options.max_runtime_seconds = int(round(float(runtime_seconds)))
        options.random_seed = int(random_seed)
        options.extract_colors = True
        options.num_threads = 4
        pair_meta: dict[str, Any] | None = None
        if pair is not None:
            image_id1, image_id2, pair_meta = pair
            pair_meta = dict(pair_meta)
            options.init_image_id1 = int(image_id1)
            options.init_image_id2 = int(image_id2)
            pair_meta.update({"image_id1": int(image_id1), "image_id2": int(image_id2)})
        options.mapper.init_min_num_inliers = 30
        options.mapper.init_min_tri_angle = 2.0
        options.mapper.init_max_forward_motion = 0.99
        options.mapper.abs_pose_min_num_inliers = 15
        options.mapper.abs_pose_min_inlier_ratio = 0.15
        options.mapper.ba_local_num_images = 6
        return options, pair_meta

    runtime_meta = {
        "max_runtime_seconds": float(max_runtime_seconds),
        "initial_pair_try_runtime_seconds": None,
    }
    initial_runtime = float(max_runtime_seconds)
    if initial_pair is not None and bool(fallback_initial_pair):
        initial_runtime = min(float(max_runtime_seconds), max(5.0, min(12.0, float(max_runtime_seconds) * 0.25)))
        runtime_meta["initial_pair_try_runtime_seconds"] = float(initial_runtime)
    options, initial_pair_meta = _make_options(initial_pair, runtime_seconds=initial_runtime)
    reconstructions = pycolmap.incremental_mapping(database_path, images_dir, sparse_dir, options=options)
    fallback_used = False
    if not reconstructions and initial_pair is not None and bool(fallback_initial_pair):
        fallback_used = True
        shutil.rmtree(sparse_dir, ignore_errors=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        fallback_options, _ = _make_options(None, runtime_seconds=float(max_runtime_seconds))
        reconstructions = pycolmap.incremental_mapping(
            database_path,
            images_dir,
            sparse_dir,
            options=fallback_options,
        )
    if not reconstructions:
        return {
            "success": False,
            "pycolmap_random_seed": int(random_seed),
            "fixed_initial_pair": initial_pair_meta,
            "initial_pair_fallback_used": bool(fallback_used),
            "runtime_meta": runtime_meta,
            "image_count": len(image_names),
            "num_models": 0,
            "best_model_id": None,
            "best_reconstruction": {
                "num_images": len(image_names),
                "num_reg_images": 0,
                "registration_rate": 0.0,
                "num_points3D": 0,
                "num_observations": 0,
                "mean_observations_per_reg_image": 0.0,
                "mean_track_length": 0.0,
                "mean_reprojection_error": 0.0,
            },
        }

    best_id, best_recon = max(reconstructions.items(), key=lambda item: int(item[1].num_reg_images()))
    best_dir = sparse_dir / str(best_id)
    best_dir.mkdir(parents=True, exist_ok=True)
    best_recon.write(best_dir)
    return {
        "success": True,
        "pycolmap_random_seed": int(random_seed),
        "fixed_initial_pair": initial_pair_meta,
        "initial_pair_fallback_used": bool(fallback_used),
        "runtime_meta": runtime_meta,
        "image_count": len(image_names),
        "num_models": len(reconstructions),
        "best_model_id": int(best_id),
        "best_reconstruction": _summarize_reconstruction(best_recon),
        "sparse_dir": str(best_dir.resolve()),
    }


def _evaluate_manifest(
    *,
    manifest_path: Path,
    model: torch.nn.Module,
    image_hw: tuple[int, int],
    device: torch.device,
    output_dir: Path,
    max_frames: int,
    frame_stride: int,
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
    enable_slam_aware_retention: bool,
    retention_scorer: dict[str, Any] | None,
    retention_score_threshold: float,
    retention_detector: str,
    retention_patch_radius: int,
    retention_min_inlier_support: int,
    retention_max_features_per_frame: int,
    retention_max_fraction: float,
    enable_pose_aware_soft_retention: bool,
    pose_soft_min_weight: float,
    pose_soft_max_weight: float,
    pose_soft_score_power: float,
    pose_soft_geometry_power: float,
    pycolmap_random_seeds: list[int],
    fixed_initial_pair: str | None,
    fallback_initial_pair: bool,
    variant_filter: list[str] | None,
) -> dict[str, Any]:
    video, dynamic_mask, particle_mask, manifest = _load_clip(
        manifest_path,
        image_hw=image_hw,
        max_frames=int(max_frames),
    )
    transient_gt = (dynamic_mask | particle_mask).astype(bool)
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
    aqua_name = f"aqua_static_conf_ge_{float(static_threshold):.3f}".replace(".", "p")
    aqua_mask = aqua["rejected_masks"][aqua_name]
    temporal_mask, temporal_source = _load_temporal_mask(manifest, video, image_hw)

    variants = {
        "raw": video,
        "temporal_rgb_inpaint": inpaint_video_with_mask(video, temporal_mask, radius=float(inpaint_radius)),
        "aqua_inpaint": inpaint_video_with_mask(video, aqua_mask, radius=float(inpaint_radius)),
        "oracle_gt_inpaint": inpaint_video_with_mask(video, transient_gt, radius=float(inpaint_radius)),
    }
    retention_meta: dict[str, Any] = {}
    if bool(enable_slam_aware_retention):
        retain_mask, retention_meta = _slam_aware_retention_mask(
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
        retention_meta = {"aqua_slam_retain_inpaint": retention_meta}
    if retention_scorer is not None:
        keypoint_context = extract_keypoint_context(
            video_rgb=video,
            detector_name=str(retention_detector),
            max_features=max(1200, int(max_num_features)),
            ratio=0.75,
            frame_step=max(1, int(frame_stride)),
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
            clip_index=0,
        )
        scores = score_retention_candidates(
            table["features"],
            retention_scorer,
            device=device,
            feature_names=table["feature_names"],
        )
        retain_mask, learned_meta = retention_mask_from_candidates(
            candidate_meta=table["candidate_meta"],
            scores=scores,
            rejected_mask=aqua_mask,
            score_threshold=float(retention_score_threshold),
            patch_radius=int(retention_patch_radius),
            max_features_per_frame=int(retention_max_features_per_frame),
            max_fraction=float(retention_max_fraction),
        )
        learned_meta.update(table["summary"])
        retained_aqua_mask = aqua_mask & ~retain_mask
        learned_name = f"aqua_learned_retain_t{float(retention_score_threshold):.2f}_inpaint".replace(".", "p")
        variants[learned_name] = inpaint_video_with_mask(
            video,
            retained_aqua_mask,
            radius=float(inpaint_radius),
        )
        retention_meta[learned_name] = learned_meta
        if bool(enable_pose_aware_soft_retention):
            retain_weight, pose_retain_mask, pose_meta = pose_aware_retention_weight_from_candidates(
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
            pose_meta.update(table["summary"])
            pose_name = f"aqua_pose_soft_t{float(retention_score_threshold):.2f}".replace(".", "p")
            variants[pose_name] = soft_temporal_fill_video(
                video,
                aqua_mask,
                static_confidence=aqua["score_maps"]["static_confidence"],
                retain_weight=retain_weight,
                temporal_radius=2,
                blur_kernel=5,
            )
            retention_meta[pose_name] = pose_meta

    if variant_filter:
        patterns = [str(item) for item in variant_filter if str(item).strip()]

        def _keep_variant(name: str) -> bool:
            return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)

        variants = {name: value for name, value in variants.items() if _keep_variant(name)}
        retention_meta = {name: value for name, value in retention_meta.items() if _keep_variant(name)}
        if not variants:
            raise ValueError(f"--variant-filter matched no variants. Available patterns: {patterns}")

    clip_dir = output_dir / str(manifest.get("name", manifest_path.parent.name))
    clip_results: dict[str, Any] = {}
    for name, variant_video in variants.items():
        for seed in pycolmap_random_seeds:
            result_name = name if len(pycolmap_random_seeds) == 1 else f"{name}__seed{int(seed)}"
            print(f"  pyCOLMAP variant: {result_name}")
            variant_dir = clip_dir / result_name
            try:
                clip_results[result_name] = _run_pycolmap_variant(
                    video_rgb=variant_video,
                    variant_dir=variant_dir,
                    frame_stride=int(frame_stride),
                    max_image_size=int(max_image_size),
                    max_num_features=int(max_num_features),
                    use_gpu=bool(pycolmap_gpu),
                    max_runtime_seconds=float(max_runtime_seconds),
                    random_seed=int(seed),
                    fixed_initial_pair=fixed_initial_pair,
                    fallback_initial_pair=bool(fallback_initial_pair),
                )
                clip_results[result_name]["base_variant"] = name
                clip_results[result_name]["variant_seed_name"] = result_name
            except Exception as exc:
                clip_results[result_name] = {
                    "success": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=6),
                    "pycolmap_random_seed": int(seed),
                    "fixed_initial_pair": fixed_initial_pair,
                    "base_variant": name,
                    "variant_seed_name": result_name,
                    "image_count": int(np.ceil(video.shape[0] / max(1, int(frame_stride)))),
                    "best_reconstruction": {
                        "num_images": int(np.ceil(video.shape[0] / max(1, int(frame_stride)))),
                        "num_reg_images": 0,
                        "registration_rate": 0.0,
                        "num_points3D": 0,
                        "num_observations": 0,
                        "mean_observations_per_reg_image": 0.0,
                        "mean_track_length": 0.0,
                        "mean_reprojection_error": 0.0,
                    },
                }

    return {
        "manifest": str(manifest_path.resolve()),
        "clip_name": str(manifest.get("name", manifest_path.parent.name)),
        "dataset": str(manifest.get("dataset", "unknown")),
        "image_hw": [int(video.shape[1]), int(video.shape[2])],
        "num_frames": int(video.shape[0]),
        "frame_stride": int(frame_stride),
        "pycolmap_random_seeds": [int(v) for v in pycolmap_random_seeds],
        "fixed_initial_pair": fixed_initial_pair,
        "fallback_initial_pair": bool(fallback_initial_pair),
        "mask_coverage": {
            "gt_transient": float(transient_gt.mean()),
            "aqua_transient": float(aqua_mask.mean()),
            "temporal_rgb_transient": float(temporal_mask.mean()),
        },
        "aqua_mask_meta": aqua["meta"],
        "slam_aware_retention_meta": retention_meta,
        "temporal_mask_source": temporal_source,
        "variants": clip_results,
    }


def _aggregate(per_clip: list[dict[str, Any]]) -> dict[str, Any]:
    variants: list[str] = []
    for clip in per_clip:
        for variant in clip["variants"]:
            if variant not in variants:
                variants.append(variant)
    out: dict[str, Any] = {"num_clips": len(per_clip), "variants": {}}
    for variant in variants:
        records = [clip["variants"][variant]["best_reconstruction"] for clip in per_clip if variant in clip["variants"]]
        successes = [clip["variants"][variant].get("success", False) for clip in per_clip if variant in clip["variants"]]
        out["variants"][variant] = {
            "clips": len(records),
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "num_reg_images_mean": float(np.mean([r["num_reg_images"] for r in records])) if records else 0.0,
            "registration_rate_mean": float(np.mean([r["registration_rate"] for r in records])) if records else 0.0,
            "num_points3D_mean": float(np.mean([r["num_points3D"] for r in records])) if records else 0.0,
            "num_observations_mean": float(np.mean([r["num_observations"] for r in records])) if records else 0.0,
            "mean_track_length_mean": float(np.mean([r["mean_track_length"] for r in records])) if records else 0.0,
            "mean_reprojection_error_mean": float(np.mean([r["mean_reprojection_error"] for r in records])) if records else 0.0,
        }
    return out


def _write_summary_csv(path: Path, aggregate: dict[str, Any]) -> None:
    rows = []
    for variant, metrics in aggregate["variants"].items():
        row = {"variant": variant}
        row.update(metrics)
        rows.append(row)
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
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--aqua-grid-stride", type=int, default=4)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--dynamic-threshold", type=float, default=0.79)
    parser.add_argument("--particle-threshold", type=float, default=0.83)
    parser.add_argument("--static-threshold", type=float, default=0.55)
    parser.add_argument("--inpaint-radius", type=float, default=3.0)
    parser.add_argument("--max-image-size", type=int, default=1024)
    parser.add_argument("--max-num-features", type=int, default=4096)
    parser.add_argument("--pycolmap-gpu", action="store_true")
    parser.add_argument("--pycolmap-random-seeds", default=None)
    parser.add_argument("--fixed-initial-pair", default=None, help="'auto' or comma-separated output frame indices, e.g. 0,8")
    parser.add_argument("--no-initial-pair-fallback", action="store_true")
    parser.add_argument("--variant-filter", action="append", default=None, help="Only run matching variant names; supports glob syntax and can be repeated.")
    parser.add_argument("--max-runtime-seconds", type=float, default=60.0)
    parser.add_argument("--enable-slam-aware-retention", action="store_true")
    parser.add_argument("--retention-scorer-path", default=None)
    parser.add_argument("--retention-score-threshold", type=float, default=0.5)
    parser.add_argument("--retention-detector", default="orb", choices=("orb", "sift"))
    parser.add_argument("--retention-patch-radius", type=int, default=5)
    parser.add_argument("--retention-min-inlier-support", type=int, default=1)
    parser.add_argument("--retention-max-features-per-frame", type=int, default=300)
    parser.add_argument("--retention-max-fraction", type=float, default=0.18)
    parser.add_argument("--enable-pose-aware-soft-retention", action="store_true")
    parser.add_argument("--pose-soft-min-weight", type=float, default=0.10)
    parser.add_argument("--pose-soft-max-weight", type=float, default=0.88)
    parser.add_argument("--pose-soft-score-power", type=float, default=0.75)
    parser.add_argument("--pose-soft-geometry-power", type=float, default=1.25)
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
        raise RuntimeError("pycolmap is required. Install with `python -m pip install pycolmap`.") from exc

    manifests = _resolve_manifests(args)
    device = _resolve_device(args.device)
    cfg = load_yaml_config(args.model_config)
    image_hw = tuple(int(v) for v in cfg["model"]["input"].get("image_size", [256, 256]))
    model = _load_model(Path(args.model_config), Path(args.ckpt_path), device=device)
    retention_scorer = load_retention_scorer(args.retention_scorer_path, device=device) if args.retention_scorer_path else None
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
                output_dir=output_dir,
                max_frames=int(args.max_frames),
                frame_stride=int(args.frame_stride),
                aqua_grid_stride=int(args.aqua_grid_stride),
                query_chunk_size=int(args.query_chunk_size),
                dynamic_threshold=float(args.dynamic_threshold),
                particle_threshold=float(args.particle_threshold),
                static_threshold=float(args.static_threshold),
                inpaint_radius=float(args.inpaint_radius),
                max_image_size=int(args.max_image_size),
                max_num_features=int(args.max_num_features),
                pycolmap_gpu=bool(args.pycolmap_gpu),
                max_runtime_seconds=float(args.max_runtime_seconds),
                enable_slam_aware_retention=bool(args.enable_slam_aware_retention),
                retention_scorer=retention_scorer,
                retention_score_threshold=float(args.retention_score_threshold),
                retention_detector=str(args.retention_detector),
                retention_patch_radius=int(args.retention_patch_radius),
                retention_min_inlier_support=int(args.retention_min_inlier_support),
                retention_max_features_per_frame=int(args.retention_max_features_per_frame),
                retention_max_fraction=float(args.retention_max_fraction),
                enable_pose_aware_soft_retention=bool(args.enable_pose_aware_soft_retention),
                pose_soft_min_weight=float(args.pose_soft_min_weight),
                pose_soft_max_weight=float(args.pose_soft_max_weight),
                pose_soft_score_power=float(args.pose_soft_score_power),
                pose_soft_geometry_power=float(args.pose_soft_geometry_power),
                pycolmap_random_seeds=pycolmap_random_seeds,
                fixed_initial_pair=args.fixed_initial_pair,
                fallback_initial_pair=not bool(args.no_initial_pair_fallback),
                variant_filter=args.variant_filter,
            )
        )

    aggregate = _aggregate(per_clip)
    metadata = {
        "model_config": str(Path(args.model_config).resolve()),
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "num_manifests": len(manifests),
        "frame_stride": int(args.frame_stride),
        "aqua_grid_stride": int(args.aqua_grid_stride),
        "dynamic_threshold": float(args.dynamic_threshold),
        "particle_threshold": float(args.particle_threshold),
        "static_threshold": float(args.static_threshold),
        "max_runtime_seconds": float(args.max_runtime_seconds),
        "pycolmap_random_seeds": [int(v) for v in pycolmap_random_seeds],
        "fixed_initial_pair": args.fixed_initial_pair,
        "initial_pair_fallback_enabled": not bool(args.no_initial_pair_fallback),
        "variant_filter": args.variant_filter,
        "retention_scorer_path": str(Path(args.retention_scorer_path).resolve()) if args.retention_scorer_path else None,
        "retention_score_threshold": float(args.retention_score_threshold),
        "colmap_cli_available": shutil.which("colmap") is not None,
        "backend": "pycolmap",
        "slam_aware_retention": {
            "enabled": bool(args.enable_slam_aware_retention),
            "detector": str(args.retention_detector),
            "patch_radius": int(args.retention_patch_radius),
            "min_inlier_support": int(args.retention_min_inlier_support),
            "max_features_per_frame": int(args.retention_max_features_per_frame),
            "max_fraction": float(args.retention_max_fraction),
        },
        "pose_aware_soft_retention": {
            "enabled": bool(args.enable_pose_aware_soft_retention),
            "min_weight": float(args.pose_soft_min_weight),
            "max_weight": float(args.pose_soft_max_weight),
            "score_power": float(args.pose_soft_score_power),
            "geometry_power": float(args.pose_soft_geometry_power),
        },
    }
    (output_dir / "per_clip_metrics.json").write_text(json.dumps(per_clip, indent=2), encoding="utf-8")
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps({"metadata": metadata, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )
    _write_summary_csv(output_dir / "summary_table.csv", aggregate)

    print("pyCOLMAP summary:")
    for variant, metrics in aggregate["variants"].items():
        print(
            f"- {variant}: success={metrics['success_rate']:.3f} "
            f"reg_rate={metrics['registration_rate_mean']:.3f} "
            f"points3D={metrics['num_points3D_mean']:.1f} "
            f"track={metrics['mean_track_length_mean']:.2f} "
            f"reproj={metrics['mean_reprojection_error_mean']:.3f}"
        )
    print(f"Saved: {output_dir / 'aggregate_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
