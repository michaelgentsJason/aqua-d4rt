#!/usr/bin/env python3
"""Evaluate non-oracle detector(+SAM) prefilter masks on WebUOT labels."""

from __future__ import annotations

import argparse
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

from aqua_prefilter_utils import binary_mask_metrics  # noqa: E402
from eval_aqua_box_prefilter_masks import _SamBoxPredictor  # noqa: E402
from eval_aqua_transient_heads import _load_rgb  # noqa: E402


def _read_manifest_list(path: str | Path) -> list[str]:
    items: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            items.append(value)
    return items


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


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


def _resize_mask(mask: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = image_hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def _load_clip(
    manifest_path: Path,
    image_hw: tuple[int, int],
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_paths = [str(path) for path in manifest["frames"]]
    if max_frames > 0:
        frame_paths = frame_paths[: int(max_frames)]
    frames = [_load_rgb(path, image_hw=image_hw) for path in frame_paths]
    video = np.stack(frames, axis=0)
    labels = np.load(str(manifest["labels_npz"]))
    dynamic = labels["dynamic_object_mask"][: len(frames)]
    particle = labels["particle_mask"][: len(frames)]
    dynamic = np.stack([_resize_mask(dynamic[t], image_hw) for t in range(dynamic.shape[0])], axis=0)
    particle = np.stack([_resize_mask(particle[t], image_hw) for t in range(particle.shape[0])], axis=0)
    return video, dynamic, particle, manifest


def _mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return intersection / float(max(1, union))


def _clip_boxes_xyxy(
    boxes: np.ndarray,
    image_hw: tuple[int, int],
    min_box_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32), np.zeros((0,), dtype=np.int64)
    h, w = image_hw
    out = boxes.astype(np.float32, copy=True).reshape(-1, 4)
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0.0, float(w))
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0.0, float(h))
    x0 = np.minimum(out[:, 0], out[:, 2])
    y0 = np.minimum(out[:, 1], out[:, 3])
    x1 = np.maximum(out[:, 0], out[:, 2])
    y1 = np.maximum(out[:, 1], out[:, 3])
    out = np.stack([x0, y0, x1, y1], axis=1)
    keep = ((out[:, 2] - out[:, 0]) >= float(min_box_size)) & ((out[:, 3] - out[:, 1]) >= float(min_box_size))
    keep_idx = np.nonzero(keep)[0].astype(np.int64)
    return out[keep_idx].astype(np.float32), keep_idx


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    order = np.argsort(-scores.astype(np.float32))
    keep: list[int] = []
    threshold = float(np.clip(iou_threshold, 0.0, 1.0))
    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(x0[idx], x0[rest])
        yy0 = np.maximum(y0[idx], y0[rest])
        xx1 = np.minimum(x1[idx], x1[rest])
        yy1 = np.minimum(y1[idx], y1[rest])
        inter = np.maximum(0.0, xx1 - xx0) * np.maximum(0.0, yy1 - yy0)
        union = areas[idx] + areas[rest] - inter
        iou = inter / np.maximum(1e-6, union)
        order = rest[iou <= threshold]
    return np.asarray(keep, dtype=np.int64)


def _dilate_and_clip_box(box: np.ndarray, image_hw: tuple[int, int], dilation_px: int) -> list[float] | None:
    h, w = image_hw
    x0, y0, x1, y1 = [float(v) for v in box]
    x0 = max(0.0, x0 - float(dilation_px))
    y0 = max(0.0, y0 - float(dilation_px))
    x1 = min(float(w), x1 + float(dilation_px))
    y1 = min(float(h), y1 + float(dilation_px))
    if x1 <= x0 + 2.0 or y1 <= y0 + 2.0:
        return None
    return [x0, y0, x1, y1]


def _boxes_to_mask(
    boxes_by_frame: list[np.ndarray],
    image_hw: tuple[int, int],
    dilation_px: int,
) -> np.ndarray:
    h, w = image_hw
    out = np.zeros((len(boxes_by_frame), h, w), dtype=np.bool_)
    for ti, boxes in enumerate(boxes_by_frame):
        for box in boxes:
            clipped = _dilate_and_clip_box(box, image_hw, dilation_px=dilation_px)
            if clipped is None:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in clipped]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 > x0 and y1 > y0:
                out[ti, y0:y1, x0:x1] = True
    return out


class _GroundingDinoDetector:
    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        box_threshold: float,
        text_threshold: float,
        nms_iou: float,
        max_boxes: int,
        min_box_size: float,
        local_files_only: bool,
    ) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        ).to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.nms_iou = float(nms_iou)
        self.max_boxes = int(max_boxes)
        self.min_box_size = float(min_box_size)

    @torch.no_grad()
    def detect_frame(self, frame_rgb: np.ndarray, prompt: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        h, w = frame_rgb.shape[:2]
        inputs = self.processor(images=frame_rgb.astype(np.uint8), text=str(prompt), return_tensors="pt")
        inputs = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=float(self.box_threshold),
            text_threshold=float(self.text_threshold),
            target_sizes=[(int(h), int(w))],
        )[0]
        raw_boxes = results["boxes"].detach().cpu().numpy().astype(np.float32)
        raw_scores = results["scores"].detach().cpu().numpy().astype(np.float32)
        raw_labels = [str(value) for value in results.get("text_labels", results.get("labels", []))]
        boxes, keep_size = _clip_boxes_xyxy(raw_boxes, (h, w), min_box_size=float(self.min_box_size))
        if boxes.shape[0] == 0:
            return boxes, np.zeros((0,), dtype=np.float32), []
        scores = raw_scores[keep_size]
        labels = [raw_labels[idx] if idx < len(raw_labels) else "" for idx in keep_size]
        keep_nms = _nms_xyxy(boxes, scores, iou_threshold=float(self.nms_iou))
        if self.max_boxes > 0:
            keep_nms = keep_nms[: self.max_boxes]
        return boxes[keep_nms], scores[keep_nms], [labels[int(idx)] for idx in keep_nms]


def _detect_video(
    video: np.ndarray,
    *,
    detector: _GroundingDinoDetector,
    prompt: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[str]]]:
    boxes_by_frame: list[np.ndarray] = []
    scores_by_frame: list[np.ndarray] = []
    labels_by_frame: list[list[str]] = []
    for frame in video:
        boxes, scores, labels = detector.detect_frame(frame.astype(np.uint8), prompt=str(prompt))
        boxes_by_frame.append(boxes)
        scores_by_frame.append(scores)
        labels_by_frame.append(labels)
    return boxes_by_frame, scores_by_frame, labels_by_frame


def _sam_from_detector_boxes(
    video: np.ndarray,
    boxes_by_frame: list[np.ndarray],
    *,
    predictor: _SamBoxPredictor,
    dilation_px: int,
    fallback_to_box: bool,
) -> np.ndarray:
    t, h, w, _ = video.shape
    image_hw = (h, w)
    box_mask = _boxes_to_mask(boxes_by_frame, image_hw, dilation_px=dilation_px)
    out = np.zeros((t, h, w), dtype=np.bool_)
    for ti, boxes in enumerate(boxes_by_frame):
        frame_boxes: list[list[float]] = []
        for box in boxes:
            clipped = _dilate_and_clip_box(box, image_hw, dilation_px=dilation_px)
            if clipped is not None:
                frame_boxes.append(clipped)
        if not frame_boxes:
            continue
        pred = predictor.predict_frame(video[ti].astype(np.uint8), frame_boxes)
        if bool(pred.any()) or not bool(fallback_to_box):
            out[ti] = pred
        else:
            out[ti] = box_mask[ti]
    return out


def _summarize_detections(scores_by_frame: list[np.ndarray]) -> dict[str, Any]:
    counts = np.asarray([len(scores) for scores in scores_by_frame], dtype=np.float32)
    all_scores = np.concatenate([scores for scores in scores_by_frame if len(scores) > 0], axis=0) if any(len(scores) > 0 for scores in scores_by_frame) else np.zeros((0,), dtype=np.float32)
    return {
        "total_boxes": int(counts.sum()),
        "mean_boxes_per_frame": float(counts.mean()) if counts.size else 0.0,
        "max_boxes_per_frame": int(counts.max()) if counts.size else 0,
        "frames_with_box_fraction": float((counts > 0).mean()) if counts.size else 0.0,
        "score_mean": float(all_scores.mean()) if all_scores.size else 0.0,
        "score_median": float(np.median(all_scores)) if all_scores.size else 0.0,
        "score_max": float(all_scores.max()) if all_scores.size else 0.0,
    }


def _write_visual(
    output_dir: Path,
    name: str,
    video: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    boxes_by_frame: list[np.ndarray],
    method: str,
) -> str | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if video.shape[0] == 0:
        return None
    picks = np.linspace(0, video.shape[0] - 1, num=min(6, video.shape[0])).round().astype(int)
    rows: list[np.ndarray] = []
    for idx in picks:
        frame = cv2.cvtColor(video[int(idx)], cv2.COLOR_RGB2BGR)
        box_view = frame.copy()
        for box in boxes_by_frame[int(idx)]:
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            cv2.rectangle(box_view, (x0, y0), (x1, y1), (40, 220, 120), 1, cv2.LINE_AA)
        gt = frame.copy()
        gt_overlay = gt.copy()
        gt_overlay[gt_mask[int(idx)].astype(bool)] = (40, 80, 255)
        gt = cv2.addWeighted(gt_overlay, 0.35, gt, 0.65, 0.0)
        pred = frame.copy()
        pred_overlay = pred.copy()
        pred_overlay[pred_mask[int(idx)].astype(bool)] = (40, 220, 120)
        pred = cv2.addWeighted(pred_overlay, 0.35, pred, 0.65, 0.0)
        bar = np.full((24, frame.shape[1] * 3, 3), 24, dtype=np.uint8)
        cv2.putText(
            bar,
            f"frame {idx:03d} | detector boxes / WebUOT bbox GT / {method}",
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        rows.append(np.concatenate([bar, np.concatenate([box_view, gt, pred], axis=1)], axis=0))
    sheet = np.concatenate(rows, axis=0)
    path = output_dir / f"{name}_{method}_mask_eval.png"
    cv2.imwrite(str(path), sheet)
    return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=None)
    parser.add_argument("--manifest-list", action="append", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default="detector_sam", choices=("detector_box", "detector_sam"))
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--prompt", default="underwater fish.")
    parser.add_argument("--detector-model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--max-boxes-per-frame", type=int, default=8)
    parser.add_argument("--min-box-size", type=float, default=4.0)
    parser.add_argument("--box-dilation-px", type=int, default=0)
    parser.add_argument("--sam-model-id", default="facebook/sam-vit-base")
    parser.add_argument("--sam-fallback-to-box", action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument(
        "--save-mask-npz",
        action="store_true",
        help="Cache each clip prediction as output_dir/masks/<clip>.npz with key pred_mask for downstream reuse.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    manifests = _resolve_manifests(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    if bool(args.save_mask_npz):
        mask_dir.mkdir(parents=True, exist_ok=True)
    image_hw = (int(args.image_height), int(args.image_width))
    local_files_only = not bool(args.allow_download)

    detector = _GroundingDinoDetector(
        model_id=str(args.detector_model_id),
        device=str(args.device),
        box_threshold=float(args.box_threshold),
        text_threshold=float(args.text_threshold),
        nms_iou=float(args.nms_iou),
        max_boxes=int(args.max_boxes_per_frame),
        min_box_size=float(args.min_box_size),
        local_files_only=local_files_only,
    )
    sam_predictor = (
        _SamBoxPredictor(model_id=str(args.sam_model_id), device=str(args.device))
        if str(args.method) == "detector_sam"
        else None
    )

    per_clip: list[dict[str, Any]] = []
    all_pred: list[np.ndarray] = []
    all_dyn: list[np.ndarray] = []
    all_particle: list[np.ndarray] = []
    all_transient: list[np.ndarray] = []
    all_detection_counts: list[float] = []
    all_detection_scores: list[np.ndarray] = []
    for idx, manifest_path in enumerate(manifests):
        print(f"[{idx + 1}/{len(manifests)}] {manifest_path}", flush=True)
        video, dynamic, particle, manifest = _load_clip(
            manifest_path,
            image_hw=image_hw,
            max_frames=int(args.max_frames),
        )
        boxes_by_frame, scores_by_frame, labels_by_frame = _detect_video(
            video,
            detector=detector,
            prompt=str(args.prompt),
        )
        if str(args.method) == "detector_box":
            pred = _boxes_to_mask(boxes_by_frame, image_hw=image_hw, dilation_px=int(args.box_dilation_px))
            source = f"groundingdino_box:{args.detector_model_id}"
        else:
            assert sam_predictor is not None
            pred = _sam_from_detector_boxes(
                video,
                boxes_by_frame,
                predictor=sam_predictor,
                dilation_px=int(args.box_dilation_px),
                fallback_to_box=bool(args.sam_fallback_to_box),
            )
            source = f"groundingdino_sam:{args.detector_model_id}+{args.sam_model_id}"
        transient = dynamic | particle
        detection = _summarize_detections(scores_by_frame)
        name = str(manifest.get("name", manifest_path.parent.name or manifest_path.stem))
        mask_npz_path: str | None = None
        if bool(args.save_mask_npz):
            npz_path = mask_dir / f"{_safe_stem(name)}.npz"
            np.savez_compressed(
                npz_path,
                pred_mask=pred.astype(np.bool_),
                dynamic_object_mask=dynamic.astype(np.bool_),
                particle_mask=particle.astype(np.bool_),
                transient_mask=transient.astype(np.bool_),
                clip=np.asarray(name),
                manifest=np.asarray(str(manifest_path.resolve())),
                method=np.asarray(str(args.method)),
                source=np.asarray(source),
                image_hw=np.asarray([int(video.shape[1]), int(video.shape[2])], dtype=np.int32),
                num_frames=np.asarray(int(video.shape[0]), dtype=np.int32),
            )
            mask_npz_path = str(npz_path.resolve())
        visual_path = (
            _write_visual(output_dir / "visuals", name, video, transient, pred, boxes_by_frame, method=str(args.method))
            if bool(args.save_visuals)
            else None
        )
        item = {
            "manifest": str(manifest_path.resolve()),
            "clip": name,
            "dataset": manifest.get("dataset"),
            "language": manifest.get("language"),
            "num_frames": int(video.shape[0]),
            "image_hw": [int(video.shape[1]), int(video.shape[2])],
            "method": str(args.method),
            "pseudo_source": source,
            "prompt": str(args.prompt),
            "coverage": {
                "dynamic_object": float(dynamic.mean()),
                "particle": float(particle.mean()),
                "transient": float(transient.mean()),
                "pred": float(pred.mean()),
            },
            "detection": detection,
            "detected_labels_sample": [labels for labels in labels_by_frame[: min(3, len(labels_by_frame))]],
            "metrics_vs_dynamic": binary_mask_metrics(pred, dynamic),
            "metrics_vs_particle": binary_mask_metrics(pred, particle),
            "metrics_vs_transient": binary_mask_metrics(pred, transient),
            "iou_vs_transient": float(_mask_iou(pred, transient)),
            "visual": visual_path,
            "mask_npz": mask_npz_path,
        }
        per_clip.append(item)
        all_pred.append(pred.reshape(-1))
        all_dyn.append(dynamic.reshape(-1))
        all_particle.append(particle.reshape(-1))
        all_transient.append(transient.reshape(-1))
        all_detection_counts.extend(float(len(scores)) for scores in scores_by_frame)
        all_detection_scores.extend(scores_by_frame)

    pred_all = np.concatenate(all_pred, axis=0).astype(bool)
    dyn_all = np.concatenate(all_dyn, axis=0).astype(bool)
    particle_all = np.concatenate(all_particle, axis=0).astype(bool)
    transient_all = np.concatenate(all_transient, axis=0).astype(bool)
    score_all = (
        np.concatenate([scores for scores in all_detection_scores if len(scores) > 0], axis=0)
        if any(len(scores) > 0 for scores in all_detection_scores)
        else np.zeros((0,), dtype=np.float32)
    )
    count_all = np.asarray(all_detection_counts, dtype=np.float32)
    wall_seconds = float(time.perf_counter() - start_time)
    total_frames = int(sum(int(item["num_frames"]) for item in per_clip))
    peak_vram_gb = (
        float(torch.cuda.max_memory_allocated() / (1024.0**3))
        if args.device == "cuda" and torch.cuda.is_available()
        else 0.0
    )
    aggregate = {
        "inputs": {
            "manifests": [str(path.resolve()) for path in manifests],
            "num_clips": len(manifests),
            "image_hw": [int(image_hw[0]), int(image_hw[1])],
            "max_frames": int(args.max_frames),
            "method": str(args.method),
            "prompt": str(args.prompt),
            "detector_model_id": str(args.detector_model_id),
            "box_threshold": float(args.box_threshold),
            "text_threshold": float(args.text_threshold),
            "nms_iou": float(args.nms_iou),
            "max_boxes_per_frame": int(args.max_boxes_per_frame),
            "min_box_size": float(args.min_box_size),
            "box_dilation_px": int(args.box_dilation_px),
            "sam_model_id": str(args.sam_model_id) if args.method == "detector_sam" else None,
            "sam_fallback_to_box": bool(args.sam_fallback_to_box),
            "local_files_only": bool(local_files_only),
            "save_mask_npz": bool(args.save_mask_npz),
            "mask_dir": str(mask_dir.resolve()) if bool(args.save_mask_npz) else None,
        },
        "coverage": {
            "dynamic_object": float(dyn_all.mean()),
            "particle": float(particle_all.mean()),
            "transient": float(transient_all.mean()),
            "pred": float(pred_all.mean()),
        },
        "detection": {
            "total_boxes": int(count_all.sum()) if count_all.size else 0,
            "mean_boxes_per_frame": float(count_all.mean()) if count_all.size else 0.0,
            "max_boxes_per_frame": int(count_all.max()) if count_all.size else 0,
            "frames_with_box_fraction": float((count_all > 0).mean()) if count_all.size else 0.0,
            "score_mean": float(score_all.mean()) if score_all.size else 0.0,
            "score_median": float(np.median(score_all)) if score_all.size else 0.0,
            "score_max": float(score_all.max()) if score_all.size else 0.0,
        },
        "runtime": {
            "wall_seconds": wall_seconds,
            "clips_per_second": float(len(manifests)) / float(max(wall_seconds, 1e-9)),
            "frames_per_second": float(total_frames) / float(max(wall_seconds, 1e-9)),
            "peak_vram_gb": peak_vram_gb,
        },
        "metrics_vs_dynamic": binary_mask_metrics(pred_all, dyn_all),
        "metrics_vs_particle": binary_mask_metrics(pred_all, particle_all),
        "metrics_vs_transient": binary_mask_metrics(pred_all, transient_all),
        "iou_vs_transient": float(_mask_iou(pred_all, transient_all)),
        "per_clip": per_clip,
        "notes": [
            "This is a non-oracle image-only detector baseline: detector boxes are predicted from RGB frames and text prompt.",
            "detector_sam uses predicted detector boxes as SAM prompts; no WebUOT boxes are used at test time.",
            "When --save-mask-npz is set, pred_mask is saved as a transient mask cache for static-map and ORB proxy baselines.",
            "For WebUOT, GT is a tracked-target bounding-box mask, not full fish instance segmentation.",
            "This evaluates prefilter masks only; it does not run D4RT or downstream SfM.",
        ],
    }
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    with (output_dir / "per_clip_metrics.jsonl").open("w", encoding="utf-8") as f:
        for item in per_clip:
            f.write(json.dumps(item) + "\n")
    metrics = aggregate["metrics_vs_transient"]
    print(f"Saved aggregate: {output_dir / 'aggregate_metrics.json'}")
    print(
        f"{args.method} vs transient: "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} iou={aggregate['iou_vs_transient']:.4f} "
        f"pred_cov={aggregate['coverage']['pred']:.4f} boxes/frame={aggregate['detection']['mean_boxes_per_frame']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
