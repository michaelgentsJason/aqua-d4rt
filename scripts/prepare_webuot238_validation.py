#!/usr/bin/env python3
"""Prepare WebUOT-238-Test clips as real fish validation data for Aqua-D4RT."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aqua_prefilter_utils import binary_mask_metrics, temporal_rgb_pseudo_mask  # noqa: E402


META_FILENAMES = ("README.md", "samples.json", "frames.json", "metadata.json", "fiftyone.yml")


def _oid(value: dict[str, Any]) -> str:
    oid = value.get("$oid")
    return str(oid) if oid is not None else str(value)


def _clip_stem(filepath: str) -> str:
    return Path(filepath).stem


def _normalize_clip_token(token: str) -> str:
    value = str(token).strip()
    if not value:
        return ""
    if value.isdigit():
        return f"WebUOT-1M_Test_{int(value):06d}"
    if re.fullmatch(r"\d{6}", value):
        return f"WebUOT-1M_Test_{value}"
    return Path(value).stem


def _download_file(repo_id: str, filename: str, output_dir: Path) -> Path:
    return Path(hf_hub_download(repo_id, filename, repo_type="dataset", local_dir=output_dir))


def _ensure_metadata(repo_id: str, metadata_dir: Path) -> dict[str, Path]:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for filename in META_FILENAMES:
        path = metadata_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            path = _download_file(repo_id, filename, metadata_dir)
        out[filename] = path
    return out


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_webuot_metadata(metadata_dir: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    samples = _load_json(metadata_dir / "samples.json")["samples"]
    frame_docs = _load_json(metadata_dir / "frames.json")["frames"]
    frames_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frame_docs:
        sid = _oid(frame["_sample_id"])
        frames_by_sample[sid].append(frame)
    for frames in frames_by_sample.values():
        frames.sort(key=lambda item: int(item.get("frame_number", 0)))
    return samples, frames_by_sample


def _parse_keywords(value: str) -> list[str]:
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def _select_samples(args: argparse.Namespace, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requested = {_normalize_clip_token(item) for item in str(args.clip_ids).split(",") if item.strip()}
    by_stem = {_clip_stem(str(sample["filepath"])): sample for sample in samples}
    if requested:
        missing = sorted(item for item in requested if item and item not in by_stem)
        if missing:
            raise RuntimeError(f"Requested clip IDs not found in WebUOT metadata: {missing}")
        return [by_stem[item] for item in sorted(requested)]

    keywords = _parse_keywords(args.keywords)
    max_bytes = float(args.max_video_mb) * 1024.0 * 1024.0
    candidates: list[dict[str, Any]] = []
    for sample in samples:
        language = str(sample.get("language", "")).lower()
        if keywords and not any(keyword in language for keyword in keywords):
            continue
        size_bytes = float(sample.get("metadata", {}).get("size_bytes", 0.0))
        if float(args.max_video_mb) > 0 and size_bytes > max_bytes:
            continue
        candidates.append(sample)
    candidates.sort(
        key=lambda sample: (
            float(sample.get("metadata", {}).get("size_bytes", 0.0)),
            int(sample.get("metadata", {}).get("total_frame_count", 0)),
            str(sample.get("filepath", "")),
        )
    )
    limit = max(1, int(args.limit))
    selected = candidates[:limit]
    if not selected:
        raise RuntimeError("No WebUOT samples matched the selection criteria.")
    return selected


def _frame_detections(frame_doc: dict[str, Any]) -> list[dict[str, Any]]:
    gt = frame_doc.get("gt")
    if not isinstance(gt, dict):
        return []
    detections = gt.get("detections", [])
    return detections if isinstance(detections, list) else []


def _bbox_score(frame_doc: dict[str, Any]) -> float:
    score = 0.0
    for det in _frame_detections(frame_doc):
        bbox = det.get("bounding_box", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        visibility = float(det.get("visibility", 1.0))
        score += max(0.0, float(bbox[2])) * max(0.0, float(bbox[3])) * max(0.0, visibility)
    return float(score)


def _select_frame_docs(
    frame_docs: list[dict[str, Any]],
    *,
    num_frames: int,
    source_stride: int,
    mode: str,
) -> tuple[list[dict[str, Any]], int, int]:
    if not frame_docs:
        raise RuntimeError("Sample has no frame annotations.")
    target = max(1, int(num_frames))
    if len(frame_docs) <= target:
        return frame_docs, 0, 1

    stride = max(1, int(source_stride))
    max_stride = max(1, (len(frame_docs) - 1) // max(1, target - 1))
    stride = min(stride, max_stride)
    span = (target - 1) * stride + 1
    if mode == "uniform":
        indices = np.linspace(0, len(frame_docs) - 1, num=target).round().astype(int).tolist()
        return [frame_docs[idx] for idx in indices], int(indices[0]), 0

    max_start = max(0, len(frame_docs) - span)
    best_start = 0
    best_score = -1.0
    for start in range(max_start + 1):
        indices = [start + i * stride for i in range(target)]
        score = sum(_bbox_score(frame_docs[idx]) for idx in indices)
        if score > best_score:
            best_score = float(score)
            best_start = int(start)
    indices = [best_start + i * stride for i in range(target)]
    return [frame_docs[idx] for idx in indices], best_start, stride


def _read_video_frame(cap: cv2.VideoCapture, frame_number_one_based: int) -> np.ndarray:
    frame_index = max(0, int(frame_number_one_based) - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to decode video frame {frame_number_one_based}")
    return frame


def _bbox_to_pixels(bbox_norm: list[float], width: int, height: int, dilation_px: int) -> tuple[int, int, int, int]:
    x, y, w, h = [float(v) for v in bbox_norm]
    x0 = int(np.floor(x * width)) - int(dilation_px)
    y0 = int(np.floor(y * height)) - int(dilation_px)
    x1 = int(np.ceil((x + w) * width)) + int(dilation_px)
    y1 = int(np.ceil((y + h) * height)) + int(dilation_px)
    x0 = int(np.clip(x0, 0, max(0, width - 1)))
    y0 = int(np.clip(y0, 0, max(0, height - 1)))
    x1 = int(np.clip(x1, x0 + 1, width))
    y1 = int(np.clip(y1, y0 + 1, height))
    return x0, y0, x1, y1


def _write_preview(
    frames_bgr: list[np.ndarray],
    dynamic_masks: np.ndarray,
    bbox_xyxy: np.ndarray,
    output_path: Path,
    fps: float,
) -> bool:
    if not frames_bgr:
        return False
    height, width = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        return False
    try:
        for idx, frame in enumerate(frames_bgr):
            vis = frame.copy()
            mask = dynamic_masks[idx].astype(bool)
            overlay = vis.copy()
            overlay[mask] = (40, 80, 255)
            vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0.0)
            for bbox in bbox_xyxy[idx]:
                x0, y0, x1, y1 = [int(v) for v in bbox]
                if x1 > x0 and y1 > y0:
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (30, 220, 255), 1)
            writer.write(vis)
    finally:
        writer.release()
    return output_path.exists() and output_path.stat().st_size > 0


def _write_contact_sheet(
    frames_bgr: list[np.ndarray],
    dynamic_masks: np.ndarray,
    pseudo_masks: np.ndarray | None,
    output_path: Path,
    max_frames: int = 8,
) -> None:
    if not frames_bgr:
        return
    picks = np.linspace(0, len(frames_bgr) - 1, num=min(max_frames, len(frames_bgr))).round().astype(int)
    rows: list[np.ndarray] = []
    for idx in picks:
        frame = frames_bgr[int(idx)]
        gt = frame.copy()
        gt_overlay = gt.copy()
        gt_overlay[dynamic_masks[int(idx)].astype(bool)] = (40, 80, 255)
        gt = cv2.addWeighted(gt_overlay, 0.35, gt, 0.65, 0.0)
        if pseudo_masks is None:
            pseudo = np.zeros_like(frame)
        else:
            pseudo = frame.copy()
            pseudo_overlay = pseudo.copy()
            pseudo_overlay[pseudo_masks[int(idx)].astype(bool)] = (40, 220, 120)
            pseudo = cv2.addWeighted(pseudo_overlay, 0.35, pseudo, 0.65, 0.0)
        label_bar = np.full((24, frame.shape[1] * 3, 3), 24, dtype=np.uint8)
        cv2.putText(label_bar, f"frame {idx:03d} | input / WebUOT bbox / RGB pseudo", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
        row = np.concatenate([frame, gt, pseudo], axis=1)
        rows.append(np.concatenate([label_bar, row], axis=0))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(output_path), sheet)


def _prepare_clip(
    sample: dict[str, Any],
    frame_docs: list[dict[str, Any]],
    *,
    repo_id: str,
    output_root: Path,
    raw_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_rel = str(sample["filepath"])
    clip_name = _clip_stem(source_rel)
    video_path = _download_file(repo_id, source_rel, raw_dir)
    clip_dir = output_root / clip_name
    frames_dir = clip_dir / "frames"
    labels_dir = clip_dir / "labels"
    visuals_dir = clip_dir / "visuals"
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    selected_docs, start_index, effective_stride = _select_frame_docs(
        frame_docs,
        num_frames=int(args.num_frames),
        source_stride=int(args.source_stride),
        mode=str(args.window_mode),
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open downloaded video: {video_path}")

    output_w = int(args.output_width)
    output_h = int(args.output_height)
    frames_bgr: list[np.ndarray] = []
    frame_paths: list[Path] = []
    dynamic_masks: list[np.ndarray] = []
    bbox_xyxy_per_frame: list[list[list[int]]] = []
    bbox_xywh_norm_per_frame: list[list[list[float]]] = []
    labels_per_frame: list[list[str]] = []
    visibility_per_frame: list[list[float]] = []
    source_frame_numbers: list[int] = []
    try:
        for out_idx, frame_doc in enumerate(selected_docs):
            frame_number = int(frame_doc.get("frame_number", out_idx + 1))
            source_frame_numbers.append(frame_number)
            frame = _read_video_frame(cap, frame_number)
            if output_w > 0 and output_h > 0:
                frame = cv2.resize(frame, (output_w, output_h), interpolation=cv2.INTER_AREA)
            height, width = frame.shape[:2]
            mask = np.zeros((height, width), dtype=np.bool_)
            frame_bboxes: list[list[int]] = []
            frame_bboxes_norm: list[list[float]] = []
            frame_labels: list[str] = []
            frame_visibility: list[float] = []
            for det in _frame_detections(frame_doc):
                bbox = det.get("bounding_box", [])
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                visibility = float(det.get("visibility", 1.0))
                if visibility <= 0:
                    continue
                x0, y0, x1, y1 = _bbox_to_pixels([float(v) for v in bbox], width, height, int(args.bbox_dilation_px))
                mask[y0:y1, x0:x1] = True
                frame_bboxes.append([x0, y0, x1, y1])
                frame_bboxes_norm.append([float(v) for v in bbox])
                frame_labels.append(str(det.get("label", "")))
                frame_visibility.append(float(visibility))
            dst = frames_dir / f"frame_{out_idx:06d}.png"
            cv2.imwrite(str(dst), frame)
            frame_paths.append(dst)
            frames_bgr.append(frame)
            dynamic_masks.append(mask)
            bbox_xyxy_per_frame.append(frame_bboxes)
            bbox_xywh_norm_per_frame.append(frame_bboxes_norm)
            labels_per_frame.append(frame_labels)
            visibility_per_frame.append(frame_visibility)
    finally:
        cap.release()

    dynamic = np.stack(dynamic_masks, axis=0).astype(np.bool_)
    particle = np.zeros_like(dynamic, dtype=np.bool_)
    max_det = max([len(item) for item in bbox_xyxy_per_frame] + [1])
    bbox_xyxy = np.zeros((len(frames_bgr), max_det, 4), dtype=np.float32)
    bbox_xywh_norm = np.zeros((len(frames_bgr), max_det, 4), dtype=np.float32)
    bbox_valid = np.zeros((len(frames_bgr), max_det), dtype=np.bool_)
    visibility = np.zeros((len(frames_bgr), max_det), dtype=np.float32)
    for t_idx, (boxes, boxes_norm, vis_values) in enumerate(zip(bbox_xyxy_per_frame, bbox_xywh_norm_per_frame, visibility_per_frame)):
        for det_idx, box in enumerate(boxes):
            bbox_xyxy[t_idx, det_idx] = np.asarray(box, dtype=np.float32)
            bbox_xywh_norm[t_idx, det_idx] = np.asarray(boxes_norm[det_idx], dtype=np.float32)
            bbox_valid[t_idx, det_idx] = True
            visibility[t_idx, det_idx] = float(vis_values[det_idx])

    labels_npz = labels_dir / "webuot_bbox_masks.npz"
    np.savez_compressed(
        labels_npz,
        dynamic_object_mask=dynamic,
        particle_mask=particle,
        transient_valid=np.ones(dynamic.shape, dtype=np.bool_),
        bbox_xyxy=bbox_xyxy,
        bbox_xywh_norm=bbox_xywh_norm,
        bbox_valid=bbox_valid,
        visibility=visibility,
        source_frame_numbers=np.asarray(source_frame_numbers, dtype=np.int32),
    )

    pseudo_npz: Path | None = None
    pseudo_metrics: dict[str, Any] | None = None
    pseudo_masks: np.ndarray | None = None
    if bool(args.make_temporal_pseudo_masks):
        video_rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr], axis=0)
        pseudo_masks = temporal_rgb_pseudo_mask(
            video_rgb,
            percentile=float(args.temporal_percentile),
            min_threshold=float(args.temporal_min_threshold),
            blur_kernel=int(args.temporal_blur_kernel),
            morph_kernel=int(args.temporal_morph_kernel),
            dilate_iterations=int(args.temporal_dilate_iterations),
            min_component_area=int(args.temporal_min_component_area),
            max_mask_fraction=float(args.temporal_max_mask_fraction),
        )
        pseudo_npz = labels_dir / "temporal_rgb_pseudo_masks.npz"
        np.savez_compressed(
            pseudo_npz,
            pseudo_dynamic_mask=pseudo_masks.astype(np.bool_),
            source="temporal_rgb_median_residual",
        )
        pseudo_metrics = binary_mask_metrics(pseudo_masks, dynamic)

    preview_path = visuals_dir / "bbox_preview.mp4"
    preview_ok = _write_preview(frames_bgr, dynamic, bbox_xyxy.astype(np.int32), preview_path, fps=float(args.preview_fps))
    contact_path = visuals_dir / "contact_sheet.png"
    _write_contact_sheet(frames_bgr, dynamic, pseudo_masks, contact_path)

    first = frames_bgr[0]
    manifest = {
        "name": clip_name,
        "dataset": "WebUOT-238-Test",
        "source_kind": "huggingface_video",
        "repo_id": repo_id,
        "source_path": str(video_path.resolve()),
        "source_relative_path": source_rel,
        "frames_dir": str(frames_dir.resolve()),
        "frames": [str(path.resolve()) for path in frame_paths],
        "labels_npz": str(labels_npz.resolve()),
        "pseudo_mask_npz": str(pseudo_npz.resolve()) if pseudo_npz is not None else None,
        "num_frames": len(frame_paths),
        "height": int(first.shape[0]),
        "width": int(first.shape[1]),
        "fps": float(sample.get("metadata", {}).get("frame_rate", 0.0)),
        "source_frame_numbers": source_frame_numbers,
        "frame_selection": {
            "mode": str(args.window_mode),
            "start_index_in_fiftyone_frames": int(start_index),
            "effective_source_stride": int(effective_stride),
        },
        "language": str(sample.get("language", "")),
        "metadata": sample.get("metadata", {}),
        "mask_notes": {
            "dynamic_object_mask": "WebUOT tracking bounding boxes rasterized as a partial target mask.",
            "particle_mask": "All false; WebUOT has no marine-snow labels.",
            "coverage_warning": "Only the tracked target is labeled, so other fish may be unlabeled.",
        },
        "preview_mp4": str(preview_path.resolve()) if preview_ok else None,
        "contact_sheet": str(contact_path.resolve()) if contact_path.exists() else None,
        "temporal_rgb_pseudo_metrics_vs_bbox": pseudo_metrics,
    }
    manifest_path = clip_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (clip_dir / "frames.txt").write_text("\n".join(manifest["frames"]) + "\n", encoding="utf-8")
    return {
        "clip": clip_name,
        "manifest": str(manifest_path.resolve()),
        "language": manifest["language"],
        "num_frames": int(len(frame_paths)),
        "source_frame_numbers": source_frame_numbers,
        "dynamic_bbox_coverage": float(dynamic.mean()),
        "pseudo_metrics_vs_bbox": pseudo_metrics,
        "preview_mp4": manifest["preview_mp4"],
        "contact_sheet": manifest["contact_sheet"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Voxel51/WebUOT-238-Test")
    parser.add_argument("--metadata-dir", default="data/real_underwater/webuot238_hf_meta")
    parser.add_argument("--output-root", default="data/real_underwater/webuot238_sample")
    parser.add_argument("--raw-dir", default="")
    parser.add_argument("--clip-ids", default="", help="Comma-separated IDs like 000022,000025. Empty selects by keywords.")
    parser.add_argument("--keywords", default="fish,shark", help="Comma-separated language keywords used when --clip-ids is empty.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-video-mb", type=float, default=25.0)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--source-stride", type=int, default=2)
    parser.add_argument("--window-mode", default="max_coverage", choices=("max_coverage", "uniform"))
    parser.add_argument("--output-width", type=int, default=256)
    parser.add_argument("--output-height", type=int, default=256)
    parser.add_argument("--bbox-dilation-px", type=int, default=2)
    parser.add_argument("--preview-fps", type=float, default=10.0)
    parser.add_argument("--make-temporal-pseudo-masks", action="store_true", default=True)
    parser.add_argument("--no-temporal-pseudo-masks", action="store_false", dest="make_temporal_pseudo_masks")
    parser.add_argument("--temporal-percentile", type=float, default=92.0)
    parser.add_argument("--temporal-min-threshold", type=float, default=18.0)
    parser.add_argument("--temporal-blur-kernel", type=int, default=5)
    parser.add_argument("--temporal-morph-kernel", type=int, default=5)
    parser.add_argument("--temporal-dilate-iterations", type=int, default=1)
    parser.add_argument("--temporal-min-component-area", type=int, default=20)
    parser.add_argument("--temporal-max-mask-fraction", type=float, default=0.45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_dir = Path(args.metadata_dir)
    output_root = Path(args.output_root)
    raw_dir = Path(args.raw_dir) if str(args.raw_dir).strip() else output_root / "_hf_raw"
    output_root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    meta_paths = _ensure_metadata(str(args.repo_id), metadata_dir)
    samples, frames_by_sample = _load_webuot_metadata(metadata_dir)
    selected = _select_samples(args, samples)
    summaries: list[dict[str, Any]] = []
    for idx, sample in enumerate(selected, start=1):
        sid = _oid(sample["_id"])
        clip = _clip_stem(str(sample["filepath"]))
        print(f"[{idx}/{len(selected)}] preparing {clip}: {sample.get('language', '')}", flush=True)
        summaries.append(
            _prepare_clip(
                sample,
                frames_by_sample.get(sid, []),
                repo_id=str(args.repo_id),
                output_root=output_root,
                raw_dir=raw_dir,
                args=args,
            )
        )

    manifests = [item["manifest"] for item in summaries]
    manifest_list_path = output_root / "manifests.txt"
    manifest_list_path.write_text("\n".join(manifests) + "\n", encoding="utf-8")
    dataset_manifest = {
        "dataset": "WebUOT-238-Test",
        "repo_id": str(args.repo_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata_files": {name: str(path.resolve()) for name, path in meta_paths.items()},
        "output_root": str(output_root.resolve()),
        "raw_dir": str(raw_dir.resolve()),
        "selection": {
            "clip_ids": str(args.clip_ids),
            "keywords": str(args.keywords),
            "limit": int(args.limit),
            "max_video_mb": float(args.max_video_mb),
        },
        "manifest_list": str(manifest_list_path.resolve()),
        "num_clips": len(summaries),
        "clips": summaries,
        "notes": [
            "WebUOT labels are tracking bounding boxes, not full fish instance masks.",
            "Use these clips for real qualitative/domain-gap validation and partial-label metrics.",
            "Temporal RGB pseudo masks are non-oracle baseline masks generated without WebUOT labels.",
        ],
    }
    dataset_manifest_path = output_root / "dataset_manifest.json"
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")
    print(f"Manifest list: {manifest_list_path}")
    print(f"Dataset manifest: {dataset_manifest_path}")
    for item in summaries:
        pseudo = item.get("pseudo_metrics_vs_bbox") or {}
        print(
            f"{item['clip']}: bbox_coverage={item['dynamic_bbox_coverage']:.4f} "
            f"pseudo_f1={pseudo.get('f1')} manifest={item['manifest']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
