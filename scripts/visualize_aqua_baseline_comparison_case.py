#!/usr/bin/env python3
"""Create a single-case visual sheet comparing Aqua and prefilter baselines."""

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
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aqua_prefilter_utils import binary_mask_metrics, temporal_rgb_pseudo_mask  # noqa: E402
from eval_aqua_transient_heads import _grid_queries, _load_model, _sigmoid  # noqa: E402
from infer_track_3d import _resolve_device  # noqa: E402
from src.core import load_yaml_config, seed_everything  # noqa: E402
from src.eval.tasks import _encode_model_memory, _run_model_for_queries  # noqa: E402


def _parse_frame_ids(value: str, num_frames: int) -> list[int]:
    if value.strip():
        items = [int(item) for item in value.split(",") if item.strip()]
    else:
        items = np.linspace(0, max(0, num_frames - 1), num=min(6, num_frames)).round().astype(int).tolist()
    return sorted({int(np.clip(item, 0, max(0, num_frames - 1))) for item in items})


def _mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return intersection / float(max(1, union))


def _resize_mask_stack(mask: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = image_hw
    if mask.shape[1:3] == (h, w):
        return mask.astype(bool)
    out = [
        cv2.resize(mask[t].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        for t in range(mask.shape[0])
    ]
    return np.stack(out, axis=0)


def _load_clip(
    manifest_path: Path,
    image_hw: tuple[int, int],
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_paths = [str(path) for path in manifest["frames"]]
    if max_frames > 0:
        frame_paths = frame_paths[: int(max_frames)]
    frames = [cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for path in frame_paths]
    video = np.stack([cv2.resize(frame, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_AREA) for frame in frames], axis=0)
    labels = np.load(str(manifest["labels_npz"]), allow_pickle=False)
    dynamic = labels["dynamic_object_mask"][: len(frames)]
    particle = labels["particle_mask"][: len(frames)]
    dynamic = _resize_mask_stack(dynamic, image_hw)
    particle = _resize_mask_stack(particle, image_hw)
    return video, dynamic, particle, manifest


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _load_cached_mask_stack(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    image_hw: tuple[int, int],
    num_frames: int,
    cache_roots: list[Path],
    keys: tuple[str, ...] = ("pred_mask", "pseudo_dynamic_mask", "dynamic_object_mask", "transient_mask"),
) -> tuple[np.ndarray, str]:
    clip_names = [
        str(manifest.get("name", "")),
        str(manifest_path.parent.name),
        str(manifest_path.stem),
    ]
    candidates: list[Path] = []
    for root in cache_roots:
        for name in clip_names:
            if not name:
                continue
            candidates.append(root / f"{name}.npz")
            candidates.append(root / f"{_safe_stem(name)}.npz")
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
        for key in keys:
            if key in payload:
                mask = np.asarray(payload[key]).astype(bool)
                if mask.ndim != 3:
                    raise ValueError(f"{candidate}: {key} must have shape T,H,W, got {mask.shape}")
                mask = mask[: int(num_frames)]
                if mask.shape[0] < int(num_frames):
                    pad = np.zeros((int(num_frames) - mask.shape[0], mask.shape[1], mask.shape[2]), dtype=bool)
                    mask = np.concatenate([mask, pad], axis=0)
                return _resize_mask_stack(mask, image_hw), str(candidate.resolve())
        raise KeyError(f"{candidate} does not contain any supported mask key: {keys}")
    expected = ", ".join(str(path) for path in unique_candidates[:4])
    raise FileNotFoundError(f"No cached mask found for {manifest_path}. Tried: {expected}")


def _grid_map(
    coord_txy: np.ndarray,
    values: np.ndarray,
    frame_idx: int,
    height: int,
    width: int,
    stride: int,
) -> np.ndarray:
    mask = coord_txy[:, 0] == int(frame_idx)
    out = np.zeros((height, width), dtype=np.float32)
    if not np.any(mask):
        return out
    xy = coord_txy[mask][:, 1:3]
    out[xy[:, 1], xy[:, 0]] = values[mask].astype(np.float32)
    kernel_size = max(3, int(stride))
    return cv2.dilate(out, np.ones((kernel_size, kernel_size), np.uint8))


def _query_mask_from_probs(
    coord_txy: np.ndarray,
    probs: np.ndarray,
    *,
    threshold: float,
    image_hw: tuple[int, int],
    stride: int,
) -> np.ndarray:
    n_frames = int(coord_txy[:, 0].max()) + 1 if coord_txy.size else 0
    h, w = image_hw
    out = np.zeros((n_frames, h, w), dtype=np.bool_)
    for frame_idx in range(n_frames):
        score_map = _grid_map(coord_txy, probs, frame_idx, h, w, stride)
        out[frame_idx] = score_map >= float(threshold)
    return out


def _smooth_query_mask_from_probs(
    coord_txy: np.ndarray,
    probs: np.ndarray,
    *,
    threshold: float,
    image_hw: tuple[int, int],
    blur_kernel: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_frames = int(coord_txy[:, 0].max()) + 1 if coord_txy.size else 0
    h, w = image_hw
    masks = np.zeros((n_frames, h, w), dtype=np.bool_)
    maps = np.zeros((n_frames, h, w), dtype=np.float32)
    kernel = int(max(1, blur_kernel))
    if kernel % 2 == 0:
        kernel += 1
    for frame_idx in range(n_frames):
        sel = coord_txy[:, 0] == int(frame_idx)
        if not np.any(sel):
            continue
        xy = coord_txy[sel][:, 1:3]
        values = probs[sel].astype(np.float32)
        xs = np.unique(xy[:, 0])
        ys = np.unique(xy[:, 1])
        coarse = np.zeros((len(ys), len(xs)), dtype=np.float32)
        x_to_i = {int(x): idx for idx, x in enumerate(xs)}
        y_to_i = {int(y): idx for idx, y in enumerate(ys)}
        for (x, y), value in zip(xy, values):
            coarse[y_to_i[int(y)], x_to_i[int(x)]] = float(value)
        dense = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
        if kernel > 1:
            dense = cv2.GaussianBlur(dense, (kernel, kernel), 0)
        dense = np.clip(dense, 0.0, 1.0).astype(np.float32)
        maps[frame_idx] = dense
        masks[frame_idx] = dense >= float(threshold)
    return masks, maps


def _aqua_predictions(
    *,
    video: np.ndarray,
    manifest: dict[str, Any],
    model_config: Path,
    ckpt_path: Path,
    device: torch.device,
    grid_stride: int,
    query_chunk_size: int,
) -> dict[str, Any]:
    query_cpu, coord_txy = _grid_queries(video.shape[0], video.shape[1], video.shape[2], stride=int(grid_stride))
    model = _load_model(model_config, ckpt_path, device=device)
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
    dynamic_probs = _sigmoid(pred["dynamic_object_logit"].numpy()).astype(np.float32)
    particle_probs = _sigmoid(pred["particle_logit"].numpy()).astype(np.float32)
    confidence_probs = _sigmoid(pred["confidence"].numpy()).astype(np.float32)
    if "static_confidence" in pred:
        static_probs = pred["static_confidence"].numpy().astype(np.float32)
    else:
        static_probs = (confidence_probs * (1.0 - dynamic_probs) * (1.0 - particle_probs)).astype(np.float32)
    return {
        "coord_txy": coord_txy,
        "dynamic_probs": dynamic_probs,
        "particle_probs": particle_probs,
        "confidence_probs": confidence_probs,
        "static_probs": static_probs,
    }


def _draw_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    del h
    bar = np.full((28, w, 3), 24, dtype=np.uint8)
    cv2.putText(bar, title, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([bar, image_bgr], axis=0)


def _overlay_mask(frame_rgb: np.ndarray, mask: np.ndarray, title: str, color_bgr: tuple[int, int, int]) -> np.ndarray:
    base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    overlay = base.copy()
    overlay[mask.astype(bool)] = color_bgr
    out = cv2.addWeighted(overlay, 0.42, base, 0.58, 0.0)
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    out[edge] = tuple(int(v) for v in color_bgr)
    return _draw_title(out, title)


def _heatmap_overlay(frame_rgb: np.ndarray, score_map: np.ndarray, title: str) -> np.ndarray:
    base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    heat = cv2.applyColorMap((np.clip(score_map, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return _draw_title(cv2.addWeighted(base, 0.70, heat, 0.30, 0.0), title)


def _static_keep_overlay(frame_rgb: np.ndarray, static_map: np.ndarray, title: str, threshold: float) -> np.ndarray:
    base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    keep = static_map >= float(threshold)
    out = base.copy()
    out[~keep] = np.clip(out[~keep].astype(np.float32) * 0.24 + np.array([24, 24, 24]), 0, 255).astype(np.uint8)
    edge = cv2.morphologyEx((~keep).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    out[edge] = (40, 220, 120)
    return _draw_title(out, title)


def _resize_panel(panel: np.ndarray, width: int) -> np.ndarray:
    h, w = panel.shape[:2]
    if w == width:
        return panel
    scale = float(width) / float(max(1, w))
    return cv2.resize(panel, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def _make_sheet(
    *,
    video: np.ndarray,
    frame_ids: list[int],
    masks: dict[str, np.ndarray],
    aqua: dict[str, Any],
    static_threshold: float,
    grid_stride: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    h, w = video.shape[1:3]
    for frame_idx in frame_ids:
        dyn_map = _grid_map(aqua["coord_txy"], aqua["dynamic_probs"], frame_idx, h, w, grid_stride)
        static_map = _grid_map(aqua["coord_txy"], aqua["static_probs"], frame_idx, h, w, grid_stride)
        columns = [
            _draw_title(cv2.cvtColor(video[frame_idx], cv2.COLOR_RGB2BGR), f"Input f{frame_idx:02d}"),
            _overlay_mask(video[frame_idx], masks["gt"][frame_idx], "WebUOT bbox GT", (40, 80, 255)),
            _overlay_mask(video[frame_idx], masks["gdino_box"][frame_idx], "GroundingDINO box", (40, 220, 120)),
            _overlay_mask(video[frame_idx], masks["gdino_sam"][frame_idx], "GroundingDINO+SAM", (40, 220, 120)),
            _overlay_mask(video[frame_idx], masks["temporal_rgb"][frame_idx], "Temporal RGB", (40, 220, 120)),
            _overlay_mask(video[frame_idx], masks["sam_box"][frame_idx], "SAM + GT box", (40, 220, 120)),
            _overlay_mask(video[frame_idx], masks["grabcut_box"][frame_idx], "GrabCut + GT box", (40, 220, 120)),
            _heatmap_overlay(video[frame_idx], dyn_map, "Aqua dynamic"),
            _overlay_mask(video[frame_idx], masks["aqua_dynamic_smooth"][frame_idx], "Aqua dyn smooth", (40, 220, 120)),
            _static_keep_overlay(video[frame_idx], static_map, "Aqua static keep", static_threshold),
        ]
        row = np.concatenate(columns, axis=1)
        rows.append(row)
    return np.concatenate(rows, axis=0)


def _write_video(
    *,
    output_path: Path,
    video: np.ndarray,
    masks: dict[str, np.ndarray],
    aqua: dict[str, Any],
    static_threshold: float,
    grid_stride: int,
    fps: float,
) -> bool:
    frames: list[np.ndarray] = []
    h, w = video.shape[1:3]
    for frame_idx in range(video.shape[0]):
        dyn_map = _grid_map(aqua["coord_txy"], aqua["dynamic_probs"], frame_idx, h, w, grid_stride)
        static_map = _grid_map(aqua["coord_txy"], aqua["static_probs"], frame_idx, h, w, grid_stride)
        columns = [
            _draw_title(cv2.cvtColor(video[frame_idx], cv2.COLOR_RGB2BGR), f"Input f{frame_idx:02d}"),
            _overlay_mask(video[frame_idx], masks["gt"][frame_idx], "WebUOT bbox GT", (40, 80, 255)),
            _overlay_mask(video[frame_idx], masks["gdino_box"][frame_idx], "GroundingDINO box", (40, 220, 120)),
            _overlay_mask(video[frame_idx], masks["gdino_sam"][frame_idx], "GroundingDINO+SAM", (40, 220, 120)),
            _heatmap_overlay(video[frame_idx], dyn_map, "Aqua dynamic"),
            _overlay_mask(video[frame_idx], masks["aqua_dynamic_smooth"][frame_idx], "Aqua dyn smooth", (40, 220, 120)),
            _static_keep_overlay(video[frame_idx], static_map, "Aqua static keep", static_threshold),
        ]
        frames.append(np.concatenate([_resize_panel(panel, 220) for panel in columns], axis=1))
    if not frames:
        return False
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        return False
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return output_path.exists() and output_path.stat().st_size > 0


def _metrics_table(masks: dict[str, np.ndarray], gt: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, mask in masks.items():
        if name == "gt":
            continue
        metrics = binary_mask_metrics(mask, gt)
        out[name] = {
            **metrics,
            "iou": float(_mask_iou(mask, gt)),
            "pred_coverage": float(mask.mean()),
            "gt_coverage": float(gt.mean()),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", default="output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu", "auto"))
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--grid-stride", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--dynamic-threshold", type=float, default=0.85)
    parser.add_argument("--smooth-dynamic-threshold", type=float, default=0.85)
    parser.add_argument("--smooth-blur-kernel", type=int, default=9)
    parser.add_argument("--static-threshold", type=float, default=0.55)
    parser.add_argument("--detector-model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--detector-prompt", default="underwater fish.")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--max-boxes-per-frame", type=int, default=8)
    parser.add_argument("--sam-model-id", default="facebook/sam-vit-base")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    device = _resolve_device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_hw = (int(args.image_height), int(args.image_width))

    manifest_path = Path(args.manifest)
    video, dynamic, particle, manifest = _load_clip(
        manifest_path,
        image_hw=image_hw,
        max_frames=int(args.max_frames),
    )
    gt = dynamic | particle

    print("Computing temporal RGB baseline...", flush=True)
    temporal_cache_roots = [
        Path(str(manifest.get("labels_npz", ""))).parent,
        manifest_path.parent / "labels",
    ]
    temporal_cache_roots = [root for root in temporal_cache_roots if str(root)]
    try:
        temporal, temporal_source = _load_cached_mask_stack(
            manifest_path=manifest_path,
            manifest=manifest,
            image_hw=image_hw,
            num_frames=video.shape[0],
            cache_roots=temporal_cache_roots,
            keys=("pseudo_dynamic_mask", "dynamic_object_mask", "transient_mask"),
        )
    except FileNotFoundError:
        temporal = temporal_rgb_pseudo_mask(video)
        temporal_source = "computed_temporal_rgb"

    cache_dir = Path(str(args.cache_dir)).expanduser() if str(args.cache_dir).strip() else None
    base_cache_roots = [
        Path("tmp/aqua_prefilter_masks/webuot238_dynamic100_groundingdino_box_all100_npz/masks"),
        Path("tmp/aqua_prefilter_masks/webuot238_fish30_groundingdino_box_all30_npz/masks"),
    ]
    sam_cache_roots = [
        Path("tmp/aqua_prefilter_masks/webuot238_fish30_groundingdino_sam_all30_npz/masks"),
    ]
    if cache_dir is not None:
        base_cache_roots = [cache_dir] + base_cache_roots
        sam_cache_roots = [cache_dir] + sam_cache_roots

    print("Loading cached baselines...", flush=True)
    gdino_box, gdino_box_source = _load_cached_mask_stack(
        manifest_path=manifest_path,
        manifest=manifest,
        image_hw=image_hw,
        num_frames=video.shape[0],
        cache_roots=base_cache_roots,
        keys=("pred_mask",),
    )
    gdino_sam, gdino_sam_source = _load_cached_mask_stack(
        manifest_path=manifest_path,
        manifest=manifest,
        image_hw=image_hw,
        num_frames=video.shape[0],
        cache_roots=sam_cache_roots,
        keys=("pred_mask",),
    )
    try:
        sam_box, sam_box_source = _load_cached_mask_stack(
            manifest_path=manifest_path,
            manifest=manifest,
            image_hw=image_hw,
            num_frames=video.shape[0],
            cache_roots=[Path("data/real_underwater/webuot238_all238/WebUOT-1M_Test_000022/labels")],
            keys=("webuot_bbox_sam_mask", "sam_box_mask", "pred_mask"),
        )
    except FileNotFoundError:
        sam_box = gdino_sam.copy()
        sam_box_source = "fallback:gdino_sam"
    try:
        grabcut_box, grabcut_box_source = _load_cached_mask_stack(
            manifest_path=manifest_path,
            manifest=manifest,
            image_hw=image_hw,
            num_frames=video.shape[0],
            cache_roots=[Path("data/real_underwater/webuot238_all238/WebUOT-1M_Test_000022/labels")],
            keys=("grabcut_box_mask", "grabcut_mask", "pred_mask"),
        )
    except FileNotFoundError:
        grabcut_box = gdino_box.copy()
        grabcut_box_source = "fallback:gdino_box"

    scores_by_frame = [np.empty((0,), dtype=np.float32) for _ in range(video.shape[0])]
    labels_by_frame: list[list[str]] = [[] for _ in range(video.shape[0])]
    if bool(args.allow_download):
        try:
            print("Attempting online detector/SAM inference...", flush=True)
            from eval_aqua_box_prefilter_masks import _SamBoxPredictor, _grabcut_box_mask, _sam_box_mask  # noqa: E402
            from eval_aqua_detector_sam_prefilter_masks import (  # noqa: E402
                _GroundingDinoDetector,
                _boxes_to_mask,
                _detect_video,
                _sam_from_detector_boxes,
            )

            detector = _GroundingDinoDetector(
                model_id=str(args.detector_model_id),
                device=str(device),
                box_threshold=float(args.box_threshold),
                text_threshold=float(args.text_threshold),
                nms_iou=0.50,
                max_boxes=int(args.max_boxes_per_frame),
                min_box_size=4.0,
                local_files_only=False,
            )
            sam_predictor = _SamBoxPredictor(model_id=str(args.sam_model_id), device=str(device))
            boxes_by_frame, scores_by_frame, labels_by_frame = _detect_video(
                video,
                detector=detector,
                prompt=str(args.detector_prompt),
            )
            gdino_box = _boxes_to_mask(boxes_by_frame, image_hw=image_hw, dilation_px=0)
            gdino_sam = _sam_from_detector_boxes(
                video,
                boxes_by_frame,
                predictor=sam_predictor,
                dilation_px=0,
                fallback_to_box=False,
            )
            grabcut_box = _grabcut_box_mask(
                video,
                boxes,
                valid,
                dilation_px=0,
                iterations=3,
                fallback_to_box=True,
            )
            sam_box = _sam_box_mask(
                video,
                boxes,
                valid,
                predictor=sam_predictor,
                dilation_px=0,
            )
        except Exception as exc:  # pragma: no cover
            print(f"Online detector/SAM unavailable, keeping cached masks: {exc}", flush=True)

    print("Running Aqua-D4RT heads...", flush=True)
    aqua = _aqua_predictions(
        video=video,
        manifest=manifest,
        model_config=Path(args.model_config),
        ckpt_path=Path(args.ckpt_path),
        device=device,
        grid_stride=int(args.grid_stride),
        query_chunk_size=int(args.query_chunk_size),
    )
    aqua_dynamic_mask = _query_mask_from_probs(
        aqua["coord_txy"],
        aqua["dynamic_probs"],
        threshold=float(args.dynamic_threshold),
        image_hw=image_hw,
        stride=int(args.grid_stride),
    )
    aqua_dynamic_smooth, aqua_dynamic_smooth_maps = _smooth_query_mask_from_probs(
        aqua["coord_txy"],
        aqua["dynamic_probs"],
        threshold=float(args.smooth_dynamic_threshold),
        image_hw=image_hw,
        blur_kernel=int(args.smooth_blur_kernel),
    )
    del aqua_dynamic_smooth_maps
    masks = {
        "gt": gt,
        "temporal_rgb": temporal,
        "gdino_box": gdino_box,
        "gdino_sam": gdino_sam,
        "sam_box": sam_box,
        "grabcut_box": grabcut_box,
        "aqua_dynamic": aqua_dynamic_mask,
        "aqua_dynamic_smooth": aqua_dynamic_smooth,
    }
    frame_ids = _parse_frame_ids(str(args.frame_ids), num_frames=video.shape[0])
    sheet = _make_sheet(
        video=video,
        frame_ids=frame_ids,
        masks=masks,
        aqua=aqua,
        static_threshold=float(args.static_threshold),
        grid_stride=int(args.grid_stride),
    )
    sheet_path = output_dir / "comparison_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)

    video_path: str | None = None
    if bool(args.save_video):
        path = output_dir / "comparison_video.mp4"
        ok = _write_video(
            output_path=path,
            video=video,
            masks=masks,
            aqua=aqua,
            static_threshold=float(args.static_threshold),
            grid_stride=int(args.grid_stride),
            fps=float(args.fps),
        )
        video_path = str(path.resolve()) if ok else None

    score_h, score_w = image_hw
    aqua_stats = {
        "dynamic_prob_mean": float(np.mean(aqua["dynamic_probs"])),
        "dynamic_mask_coverage": float(aqua_dynamic_mask.mean()),
        "static_confidence_mean": float(np.mean(aqua["static_probs"])),
        "static_keep_coverage": float(
            np.mean(
                [
                    (_grid_map(aqua["coord_txy"], aqua["static_probs"], idx, score_h, score_w, int(args.grid_stride)) >= float(args.static_threshold)).mean()
                    for idx in range(video.shape[0])
                ]
            )
        ),
    }
    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "clip": Path(args.manifest).parent.name,
        "num_frames": int(video.shape[0]),
        "frame_ids": frame_ids,
        "image_hw": [int(image_hw[0]), int(image_hw[1])],
        "detector": {
            "model_id": str(args.detector_model_id),
            "prompt": str(args.detector_prompt),
            "box_threshold": float(args.box_threshold),
            "text_threshold": float(args.text_threshold),
            "mean_boxes_per_frame": float(np.mean([len(scores) for scores in scores_by_frame])) if scores_by_frame else 0.0,
            "detected_labels_sample": labels_by_frame[:3],
            "gdino_box_source": gdino_box_source,
            "gdino_sam_source": gdino_sam_source,
            "sam_box_source": sam_box_source,
            "grabcut_box_source": grabcut_box_source,
            "temporal_rgb_source": temporal_source,
        },
        "metrics_vs_webuot_bbox_gt": _metrics_table(masks, gt),
        "aqua_stats": aqua_stats,
        "aqua_visualization": {
            "dynamic_threshold": float(args.dynamic_threshold),
            "smooth_dynamic_threshold": float(args.smooth_dynamic_threshold),
            "smooth_blur_kernel": int(args.smooth_blur_kernel),
        },
        "notes": [
            "WebUOT GT is a tracked-target bounding-box mask, not full fish instance segmentation.",
            "SAM+GT-box and GrabCut+GT-box use GT boxes and are oracle-ish/prompted baselines.",
            "Aqua dynamic is shown from query-grid head probabilities; Aqua static keep visualizes static_confidence >= threshold.",
        ],
        "outputs": {
            "comparison_sheet": str(sheet_path.resolve()),
            "comparison_video": video_path,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved sheet: {sheet_path}")
    print(f"Saved summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
