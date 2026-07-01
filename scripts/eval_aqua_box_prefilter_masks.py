#!/usr/bin/env python3
"""Evaluate box-prompted prefilter masks against WebUOT transient labels."""

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

from aqua_prefilter_utils import binary_mask_metrics  # noqa: E402
from eval_aqua_transient_heads import _load_rgb  # noqa: E402


def _read_manifest_list(path: str | Path) -> list[str]:
    items: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            items.append(value)
    return items


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


def _resize_boxes_xyxy(boxes: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    scale_x = float(dst_w) / float(max(1, src_w))
    scale_y = float(dst_h) / float(max(1, src_h))
    out = boxes.astype(np.float32).copy()
    out[..., [0, 2]] *= scale_x
    out[..., [1, 3]] *= scale_y
    out[..., [0, 2]] = np.clip(out[..., [0, 2]], 0, dst_w)
    out[..., [1, 3]] = np.clip(out[..., [1, 3]], 0, dst_h)
    return out


def _load_clip(
    manifest_path: Path,
    image_hw: tuple[int, int],
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
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
    boxes = labels["bbox_xyxy"][: len(frames)].astype(np.float32)
    valid = labels["bbox_valid"][: len(frames)].astype(bool)
    src_hw = (int(manifest.get("height", image_hw[0])), int(manifest.get("width", image_hw[1])))
    boxes = _resize_boxes_xyxy(boxes, src_hw=src_hw, dst_hw=image_hw)
    return video, dynamic, particle, boxes, valid, manifest


def _mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return intersection / float(max(1, union))


def _bbox_mask(boxes: np.ndarray, valid: np.ndarray, image_hw: tuple[int, int], dilation_px: int) -> np.ndarray:
    t, _, _ = boxes.shape
    h, w = image_hw
    out = np.zeros((t, h, w), dtype=np.bool_)
    for ti in range(t):
        for box, is_valid in zip(boxes[ti], valid[ti]):
            if not bool(is_valid):
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            x0 = max(0, x0 - int(dilation_px))
            y0 = max(0, y0 - int(dilation_px))
            x1 = min(w, x1 + int(dilation_px))
            y1 = min(h, y1 + int(dilation_px))
            if x1 > x0 and y1 > y0:
                out[ti, y0:y1, x0:x1] = True
    return out


def _grabcut_box_mask(
    video: np.ndarray,
    boxes: np.ndarray,
    valid: np.ndarray,
    *,
    dilation_px: int,
    iterations: int,
    fallback_to_box: bool,
) -> np.ndarray:
    t, h, w, _ = video.shape
    out = np.zeros((t, h, w), dtype=np.bool_)
    box_fallback = _bbox_mask(boxes, valid, (h, w), dilation_px=dilation_px)
    for ti in range(t):
        frame_bgr = cv2.cvtColor(video[ti].astype(np.uint8), cv2.COLOR_RGB2BGR)
        for box, is_valid in zip(boxes[ti], valid[ti]):
            if not bool(is_valid):
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            x0 = max(0, x0 - int(dilation_px))
            y0 = max(0, y0 - int(dilation_px))
            x1 = min(w, x1 + int(dilation_px))
            y1 = min(h, y1 + int(dilation_px))
            if x1 <= x0 + 2 or y1 <= y0 + 2:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(
                    frame_bgr,
                    mask,
                    (x0, y0, x1 - x0, y1 - y0),
                    bgd_model,
                    fgd_model,
                    int(iterations),
                    cv2.GC_INIT_WITH_RECT,
                )
                fg = np.logical_or(mask == cv2.GC_FGD, mask == cv2.GC_PR_FGD)
                fg &= box_fallback[ti]
                if bool(fg.any()):
                    out[ti] |= fg
                elif bool(fallback_to_box):
                    out[ti] |= box_fallback[ti]
            except cv2.error:
                if bool(fallback_to_box):
                    out[ti] |= box_fallback[ti]
    return out


class _SamBoxPredictor:
    def __init__(self, model_id: str, device: str) -> None:
        from transformers import SamModel, SamProcessor

        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device = torch.device(device)

    @torch.no_grad()
    def predict_frame(self, frame_rgb: np.ndarray, boxes_xyxy: list[list[float]]) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        if not boxes_xyxy:
            return np.zeros((h, w), dtype=np.bool_)
        input_boxes = [[boxes_xyxy]]
        inputs = self.processor(frame_rgb, input_boxes=input_boxes, return_tensors="pt")
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        scores = outputs.iou_scores.detach().cpu()
        out = np.zeros((h, w), dtype=np.bool_)
        for idx in range(masks.shape[0]):
            mask_set = masks[idx]
            score_set = scores[0, idx]
            best_idx = int(torch.argmax(score_set).item())
            out |= mask_set[best_idx].numpy().astype(bool)
        return out


def _sam_box_mask(
    video: np.ndarray,
    boxes: np.ndarray,
    valid: np.ndarray,
    *,
    predictor: _SamBoxPredictor,
    dilation_px: int,
) -> np.ndarray:
    t, h, w, _ = video.shape
    out = np.zeros((t, h, w), dtype=np.bool_)
    for ti in range(t):
        frame_boxes: list[list[float]] = []
        for box, is_valid in zip(boxes[ti], valid[ti]):
            if not bool(is_valid):
                continue
            x0, y0, x1, y1 = [float(v) for v in box]
            x0 = max(0.0, x0 - float(dilation_px))
            y0 = max(0.0, y0 - float(dilation_px))
            x1 = min(float(w), x1 + float(dilation_px))
            y1 = min(float(h), y1 + float(dilation_px))
            if x1 > x0 + 2 and y1 > y0 + 2:
                frame_boxes.append([x0, y0, x1, y1])
        out[ti] = predictor.predict_frame(video[ti].astype(np.uint8), frame_boxes)
    return out


def _write_visual(
    output_dir: Path,
    name: str,
    video: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    method: str,
) -> str | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if video.shape[0] == 0:
        return None
    picks = np.linspace(0, video.shape[0] - 1, num=min(6, video.shape[0])).round().astype(int)
    rows: list[np.ndarray] = []
    for idx in picks:
        frame = cv2.cvtColor(video[int(idx)], cv2.COLOR_RGB2BGR)
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
            f"frame {idx:03d} | input / WebUOT bbox GT / {method}",
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        rows.append(np.concatenate([bar, np.concatenate([frame, gt, pred], axis=1)], axis=0))
    sheet = np.concatenate(rows, axis=0)
    path = output_dir / f"{name}_{method}_mask_eval.png"
    cv2.imwrite(str(path), sheet)
    return str(path.resolve())


def _predict_mask(
    *,
    method: str,
    video: np.ndarray,
    boxes: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
    sam_predictor: _SamBoxPredictor | None = None,
) -> tuple[np.ndarray, str]:
    image_hw = (int(video.shape[1]), int(video.shape[2]))
    if method == "bbox":
        return _bbox_mask(boxes, valid, image_hw, dilation_px=int(args.box_dilation_px)), "bbox_prompt_oracle_mask"
    if method == "grabcut":
        return (
            _grabcut_box_mask(
                video,
                boxes,
                valid,
                dilation_px=int(args.box_dilation_px),
                iterations=int(args.grabcut_iterations),
                fallback_to_box=bool(args.grabcut_fallback_to_box),
            ),
            "grabcut_box_prompt",
        )
    if method == "sam":
        predictor = sam_predictor or _SamBoxPredictor(model_id=str(args.sam_model_id), device=str(args.device))
        return (
            _sam_box_mask(
                video,
                boxes,
                valid,
                predictor=predictor,
                dilation_px=int(args.box_dilation_px),
            ),
            f"sam_box_prompt:{args.sam_model_id}",
        )
    raise ValueError(f"Unsupported method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=None)
    parser.add_argument("--manifest-list", action="append", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default="grabcut", choices=("bbox", "grabcut", "sam"))
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--box-dilation-px", type=int, default=0)
    parser.add_argument("--grabcut-iterations", type=int, default=3)
    parser.add_argument("--grabcut-fallback-to-box", action="store_true")
    parser.add_argument("--sam-model-id", default="facebook/sam-vit-base")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--save-visuals", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.method == "sam" and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    manifests = _resolve_manifests(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_hw = (int(args.image_height), int(args.image_width))
    sam_predictor = (
        _SamBoxPredictor(model_id=str(args.sam_model_id), device=str(args.device))
        if str(args.method) == "sam"
        else None
    )

    per_clip: list[dict[str, Any]] = []
    all_pred: list[np.ndarray] = []
    all_dyn: list[np.ndarray] = []
    all_particle: list[np.ndarray] = []
    all_transient: list[np.ndarray] = []
    for idx, manifest_path in enumerate(manifests):
        print(f"[{idx + 1}/{len(manifests)}] {manifest_path}", flush=True)
        video, dynamic, particle, boxes, valid, manifest = _load_clip(
            manifest_path,
            image_hw=image_hw,
            max_frames=int(args.max_frames),
        )
        pred, source = _predict_mask(
            method=str(args.method),
            video=video,
            boxes=boxes,
            valid=valid,
            args=args,
            sam_predictor=sam_predictor,
        )
        transient = dynamic | particle
        name = manifest_path.parent.name or manifest_path.stem
        visual_path = (
            _write_visual(output_dir / "visuals", name, video, transient, pred, method=str(args.method))
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
            "coverage": {
                "dynamic_object": float(dynamic.mean()),
                "particle": float(particle.mean()),
                "transient": float(transient.mean()),
                "pred": float(pred.mean()),
            },
            "metrics_vs_dynamic": binary_mask_metrics(pred, dynamic),
            "metrics_vs_particle": binary_mask_metrics(pred, particle),
            "metrics_vs_transient": binary_mask_metrics(pred, transient),
            "iou_vs_transient": float(_mask_iou(pred, transient)),
            "visual": visual_path,
        }
        per_clip.append(item)
        all_pred.append(pred.reshape(-1))
        all_dyn.append(dynamic.reshape(-1))
        all_particle.append(particle.reshape(-1))
        all_transient.append(transient.reshape(-1))

    pred_all = np.concatenate(all_pred, axis=0).astype(bool)
    dyn_all = np.concatenate(all_dyn, axis=0).astype(bool)
    particle_all = np.concatenate(all_particle, axis=0).astype(bool)
    transient_all = np.concatenate(all_transient, axis=0).astype(bool)
    aggregate = {
        "inputs": {
            "manifests": [str(path.resolve()) for path in manifests],
            "num_clips": len(manifests),
            "image_hw": [int(image_hw[0]), int(image_hw[1])],
            "max_frames": int(args.max_frames),
            "method": str(args.method),
            "box_dilation_px": int(args.box_dilation_px),
            "grabcut_iterations": int(args.grabcut_iterations),
            "grabcut_fallback_to_box": bool(args.grabcut_fallback_to_box),
            "sam_model_id": str(args.sam_model_id) if args.method == "sam" else None,
        },
        "coverage": {
            "dynamic_object": float(dyn_all.mean()),
            "particle": float(particle_all.mean()),
            "transient": float(transient_all.mean()),
            "pred": float(pred_all.mean()),
        },
        "metrics_vs_dynamic": binary_mask_metrics(pred_all, dyn_all),
        "metrics_vs_particle": binary_mask_metrics(pred_all, particle_all),
        "metrics_vs_transient": binary_mask_metrics(pred_all, transient_all),
        "iou_vs_transient": float(_mask_iou(pred_all, transient_all)),
        "per_clip": per_clip,
        "notes": [
            "This evaluates prefilter masks only; it does not run D4RT.",
            "For WebUOT, GT is a tracked-target bounding-box mask, not full fish instance segmentation.",
            "bbox mode uses WebUOT labels directly and is an oracle/upper-bound style baseline.",
            "grabcut/sam modes are box-prompted baselines; they require boxes at test time.",
        ],
    }
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    with (output_dir / "per_clip_metrics.jsonl").open("w", encoding="utf-8") as f:
        for item in per_clip:
            f.write(json.dumps(item) + "\n")
    metrics = aggregate["metrics_vs_transient"]
    print(f"Saved aggregate: {output_dir / 'aggregate_metrics.json'}")
    print(
        f"{args.method} prefilter vs transient: "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} iou={aggregate['iou_vs_transient']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
