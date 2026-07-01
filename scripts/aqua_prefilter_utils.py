#!/usr/bin/env python3
"""Utility functions for Aqua-D4RT RGB prefilter baselines."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def binary_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, Any]:
    """Compute binary mask metrics for boolean-like arrays."""

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    precision = tp / float(max(1, tp + fp))
    recall = tp / float(max(1, tp + fn))
    f1 = 2.0 * precision * recall / float(max(1e-12, precision + recall))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pred_positive_rate": float(pred.mean()) if pred.size else 0.0,
        "gt_positive_rate": float(gt.mean()) if gt.size else 0.0,
    }


def query_labels_from_mask(mask_t_hw: np.ndarray, coord_txy: np.ndarray) -> np.ndarray:
    """Sample a dense T,H,W mask at D4RT grid query coordinates."""

    mask = mask_t_hw.astype(bool)
    coords = coord_txy.astype(np.int64)
    t = np.clip(coords[:, 0], 0, mask.shape[0] - 1)
    x = np.clip(coords[:, 1], 0, mask.shape[2] - 1)
    y = np.clip(coords[:, 2], 0, mask.shape[1] - 1)
    return mask[t, y, x].astype(bool)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if int(min_area) <= 1:
        return mask.astype(bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, dtype=np.bool_)
    for label in range(1, int(num_labels)):
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area):
            out |= labels == label
    return out


def temporal_rgb_pseudo_mask(
    video_rgb: np.ndarray,
    *,
    percentile: float = 92.0,
    min_threshold: float = 18.0,
    blur_kernel: int = 5,
    morph_kernel: int = 5,
    dilate_iterations: int = 1,
    min_component_area: int = 20,
    max_mask_fraction: float = 0.45,
) -> np.ndarray:
    """Generate a non-oracle transient pseudo-mask from RGB temporal change.

    This is intentionally a lightweight baseline, not a learned segmenter. It
    estimates a per-pixel temporal median background, combines median residuals
    with adjacent-frame differences, then keeps only high-change components.
    """

    if video_rgb.ndim != 4 or video_rgb.shape[-1] != 3:
        raise ValueError(f"video_rgb must have shape [T,H,W,3], got {video_rgb.shape}")
    if video_rgb.shape[0] == 0:
        return np.zeros(video_rgb.shape[:3], dtype=np.bool_)

    video_u8 = np.clip(video_rgb, 0, 255).astype(np.uint8, copy=False)
    gray_frames: list[np.ndarray] = []
    blur_k = int(max(1, blur_kernel))
    if blur_k % 2 == 0:
        blur_k += 1
    for frame in video_u8:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if blur_k > 1:
            gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        gray_frames.append(gray)
    gray = np.stack(gray_frames, axis=0)

    median_bg = np.median(gray, axis=0).astype(np.float32)
    residual = np.abs(gray - median_bg[None, :, :])
    adjacent = np.zeros_like(residual, dtype=np.float32)
    if gray.shape[0] > 1:
        adjacent[1:] = np.maximum(adjacent[1:], np.abs(gray[1:] - gray[:-1]))
        adjacent[:-1] = np.maximum(adjacent[:-1], np.abs(gray[:-1] - gray[1:]))
    score = 0.65 * residual + 0.35 * adjacent

    out = np.zeros(score.shape, dtype=np.bool_)
    morph_k = int(max(1, morph_kernel))
    kernel = np.ones((morph_k, morph_k), dtype=np.uint8)
    max_fraction = float(np.clip(max_mask_fraction, 0.0, 1.0))
    for idx in range(score.shape[0]):
        frame_score = score[idx]
        threshold = max(float(min_threshold), float(np.percentile(frame_score, float(percentile))))
        mask = frame_score >= threshold
        if max_fraction > 0.0 and float(mask.mean()) > max_fraction:
            threshold = max(threshold, float(np.percentile(frame_score, 100.0 * (1.0 - max_fraction))))
            mask = frame_score >= threshold
        mask_u8 = mask.astype(np.uint8)
        if morph_k > 1:
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        if int(dilate_iterations) > 0:
            mask_u8 = cv2.dilate(mask_u8, kernel, iterations=int(dilate_iterations))
        out[idx] = _remove_small_components(mask_u8 > 0, min_area=int(min_component_area))
    return out


def inpaint_video_with_mask(video_rgb: np.ndarray, mask_t_hw: np.ndarray, radius: float = 3.0) -> np.ndarray:
    """Inpaint masked pixels in an RGB video with OpenCV Telea inpainting."""

    if video_rgb.shape[:3] != mask_t_hw.shape:
        raise ValueError(f"video/mask shape mismatch: {video_rgb.shape} vs {mask_t_hw.shape}")
    out_frames: list[np.ndarray] = []
    for frame_rgb, frame_mask in zip(video_rgb, mask_t_hw.astype(bool)):
        if not bool(frame_mask.any()):
            out_frames.append(frame_rgb.copy())
            continue
        mask_u8 = (frame_mask.astype(np.uint8) * 255)
        out_frames.append(cv2.inpaint(frame_rgb, mask_u8, float(radius), cv2.INPAINT_TELEA))
    return np.stack(out_frames, axis=0)


def soft_temporal_fill_video(
    video_rgb: np.ndarray,
    mask_t_hw: np.ndarray,
    *,
    static_confidence: np.ndarray | None = None,
    retain_mask: np.ndarray | None = None,
    retain_weight: np.ndarray | None = None,
    min_alpha: float = 0.35,
    max_alpha: float = 0.92,
    temporal_radius: int = 2,
    blur_kernel: int = 5,
) -> np.ndarray:
    """Softly replace rejected pixels with a local temporal median.

    This keeps the image dense for SfM while avoiding the hard edges introduced
    by binary masking or single-frame inpainting. Pixels marked by
    ``retain_mask`` are copied back from the original video.
    """

    if video_rgb.shape[:3] != mask_t_hw.shape:
        raise ValueError(f"video/mask shape mismatch: {video_rgb.shape} vs {mask_t_hw.shape}")
    if static_confidence is not None and static_confidence.shape != mask_t_hw.shape:
        raise ValueError(f"static confidence shape mismatch: {static_confidence.shape} vs {mask_t_hw.shape}")
    if retain_mask is not None and retain_mask.shape != mask_t_hw.shape:
        raise ValueError(f"retain mask shape mismatch: {retain_mask.shape} vs {mask_t_hw.shape}")
    if retain_weight is not None and retain_weight.shape != mask_t_hw.shape:
        raise ValueError(f"retain weight shape mismatch: {retain_weight.shape} vs {mask_t_hw.shape}")

    video = np.clip(video_rgb, 0, 255).astype(np.uint8, copy=False)
    rejected = mask_t_hw.astype(bool)
    keep_original = retain_mask.astype(bool) if retain_mask is not None else np.zeros(rejected.shape, dtype=bool)
    if retain_weight is None:
        original_weight = np.zeros(rejected.shape, dtype=np.float32)
    else:
        original_weight = np.clip(retain_weight.astype(np.float32), 0.0, 1.0)
    if bool(keep_original.any()):
        original_weight[keep_original] = 1.0
    out = video.astype(np.float32).copy()
    radius = max(0, int(temporal_radius))
    blur_k = max(1, int(blur_kernel))
    if blur_k % 2 == 0:
        blur_k += 1

    for frame_idx in range(video.shape[0]):
        frame_mask = rejected[frame_idx]
        if not bool(frame_mask.any()):
            continue
        start = max(0, frame_idx - radius)
        stop = min(video.shape[0], frame_idx + radius + 1)
        neighbors = [idx for idx in range(start, stop) if idx != frame_idx]
        if not neighbors:
            neighbors = [frame_idx]
        fill = np.median(video[neighbors].astype(np.float32), axis=0)
        if blur_k > 1:
            fill = cv2.GaussianBlur(fill.astype(np.uint8), (blur_k, blur_k), 0).astype(np.float32)

        if static_confidence is None:
            alpha = np.full(frame_mask.shape, float(max_alpha), dtype=np.float32)
        else:
            conf = np.clip(static_confidence[frame_idx].astype(np.float32), 0.0, 1.0)
            alpha = float(min_alpha) + (1.0 - conf) * (float(max_alpha) - float(min_alpha))
            alpha = np.clip(alpha, min(float(min_alpha), float(max_alpha)), max(float(min_alpha), float(max_alpha)))
        # Pose-aware retention is represented as an original-image weight. A
        # weight of 1 keeps the source pixel; 0 uses the usual temporal fill.
        alpha = alpha * (1.0 - original_weight[frame_idx])
        alpha3 = alpha[:, :, None]
        current = out[frame_idx]
        blended = current * (1.0 - alpha3) + fill * alpha3
        current[frame_mask] = blended[frame_mask]
        out[frame_idx] = current

    return np.clip(out, 0, 255).astype(np.uint8)
