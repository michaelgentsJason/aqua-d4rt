#!/usr/bin/env python3
"""Create a compact, paper-facing Aqua-D4RT mapping-claim figure.

The output is intentionally simpler than the debug sheet: Input, raw D4RT
query candidates, Aqua static query candidates, and a zoom/numbers panel.
It visualizes query-map cleanliness, not RGB image restoration.
"""

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

from eval_aqua_transient_heads import _grid_queries, _load_clip, _load_model, _sigmoid  # noqa: E402
from infer_track_3d import _resolve_device  # noqa: E402
from src.core import load_yaml_config, seed_everything  # noqa: E402
from src.eval.tasks import _encode_model_memory, _run_model_for_queries  # noqa: E402


def _safe_rate(num: int | float, den: int | float) -> float:
    return float(num) / float(max(float(den), 1.0))


def _query_metrics(keep: np.ndarray, labels_transient: np.ndarray) -> dict[str, Any]:
    keep_b = keep.astype(bool)
    transient = labels_transient.astype(bool)
    static = ~transient
    kept = int(keep_b.sum())
    kept_transient = int(np.logical_and(keep_b, transient).sum())
    kept_static = int(np.logical_and(keep_b, static).sum())
    total_static = int(static.sum())
    total_transient = int(transient.sum())
    return {
        "kept": kept,
        "total": int(keep_b.size),
        "kept_rate": _safe_rate(kept, keep_b.size),
        "contamination": _safe_rate(kept_transient, kept),
        "static_retention": _safe_rate(kept_static, total_static),
        "transient_rejection": 1.0 - _safe_rate(kept_transient, total_transient),
    }


def _run_aqua(
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
    confidence_probs = _sigmoid(pred["confidence"].numpy()).astype(np.float32)
    dynamic_probs = _sigmoid(pred["dynamic_object_logit"].numpy()).astype(np.float32)
    particle_probs = _sigmoid(pred["particle_logit"].numpy()).astype(np.float32)
    if "static_confidence" in pred:
        static_probs = pred["static_confidence"].numpy().astype(np.float32)
    else:
        static_probs = confidence_probs * (1.0 - dynamic_probs) * (1.0 - particle_probs)
    return {
        "coord_txy": coord_txy,
        "static_probs": static_probs,
        "transient_probs": np.maximum(dynamic_probs, particle_probs).astype(np.float32),
    }


def _fit_panel(image_bgr: np.ndarray, panel_hw: tuple[int, int]) -> np.ndarray:
    ph, pw = panel_hw
    h, w = image_bgr.shape[:2]
    scale = min(float(pw) / float(max(1, w)), float(ph) / float(max(1, h)))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((ph, pw, 3), 248, dtype=np.uint8)
    y0 = (ph - nh) // 2
    x0 = (pw - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _title(panel_bgr: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    h, w = panel_bgr.shape[:2]
    bar_h = 42 if subtitle else 30
    bar = np.full((bar_h, w, 3), 255, dtype=np.uint8)
    cv2.putText(bar, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (22, 22, 22), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(bar, subtitle, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (82, 82, 82), 1, cv2.LINE_AA)
    out = np.concatenate([bar, panel_bgr], axis=0)
    cv2.rectangle(out, (0, 0), (w - 1, h + bar_h - 1), (210, 210, 210), 1, cv2.LINE_AA)
    return out


def _draw_queries(
    frame_rgb: np.ndarray,
    coord_txy: np.ndarray,
    labels_transient: np.ndarray,
    keep: np.ndarray,
    frame_idx: int,
    *,
    draw_rejected: bool,
    point_radius: int,
) -> np.ndarray:
    out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    frame_mask = coord_txy[:, 0] == int(frame_idx)
    xy = coord_txy[frame_mask][:, 1:3]
    labels = labels_transient[frame_mask].astype(bool)
    keep_f = keep[frame_mask].astype(bool)
    if draw_rejected:
        for x, y in xy[~keep_f]:
            cv2.circle(out, (int(x), int(y)), max(1, point_radius - 1), (120, 120, 120), -1, cv2.LINE_AA)
    for (x, y), is_transient in zip(xy[keep_f], labels[keep_f]):
        color = (35, 35, 230) if bool(is_transient) else (55, 205, 105)
        cv2.circle(out, (int(x), int(y)), int(point_radius), color, -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), int(point_radius), (20, 20, 20), 1, cv2.LINE_AA)
    return out


def _best_zoom_box(transient_mask: np.ndarray, width: int, height: int, box_size: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(transient_mask.astype(bool))
    size = int(min(max(64, box_size), width, height))
    if xs.size == 0:
        cx, cy = width // 2, height // 2
    else:
        cx, cy = int(np.median(xs)), int(np.median(ys))
    x0 = int(np.clip(cx - size // 2, 0, max(0, width - size)))
    y0 = int(np.clip(cy - size // 2, 0, max(0, height - size)))
    return x0, y0, x0 + size, y0 + size


def _crop_zoom(image_bgr: np.ndarray, box: tuple[int, int, int, int], panel_hw: tuple[int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = image_bgr[y0:y1, x0:x1]
    return _fit_panel(crop, panel_hw)


def _numbers_panel(
    *,
    panel_hw: tuple[int, int],
    raw_metrics: dict[str, Any],
    aqua_metrics: dict[str, Any],
    dataset_note: str,
) -> np.ndarray:
    h, w = panel_hw
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    raw_c = 100.0 * float(raw_metrics["contamination"])
    aqua_c = 100.0 * float(aqua_metrics["contamination"])
    retention = 100.0 * float(aqua_metrics["static_retention"])
    rejection = 100.0 * float(aqua_metrics["transient_rejection"])
    lines = [
        ("Query-map contamination", 0.58, (30, 30, 30)),
        (f"{raw_c:.2f}% -> {aqua_c:.2f}%", 0.92, (15, 105, 85)),
        ("Static retention", 0.58, (30, 30, 30)),
        (f"{retention:.2f}%", 0.92, (25, 85, 180)),
        ("Transient rejection", 0.58, (30, 30, 30)),
        (f"{rejection:.2f}%", 0.72, (140, 70, 25)),
        ("green=static kept, red=transient kept", 0.36, (85, 85, 85)),
        ("gray=rejected query", 0.36, (85, 85, 85)),
    ]
    y = 34
    for text, scale, color in lines:
        cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if scale >= 0.9 else 1, cv2.LINE_AA)
        y += 42 if scale >= 0.8 else 28
    if dataset_note:
        wrapped = [dataset_note[i : i + 44] for i in range(0, len(dataset_note), 44)]
        y = max(y + 4, h - 54)
        for line in wrapped[:2]:
            cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (95, 95, 95), 1, cv2.LINE_AA)
            y += 18
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", default="output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--frame-id", type=int, default=20)
    parser.add_argument("--grid-stride", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--static-threshold", type=float, default=0.55)
    parser.add_argument("--panel-size", type=int, default=320)
    parser.add_argument("--zoom-size", type=int, default=112)
    parser.add_argument("--point-radius", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    cfg = load_yaml_config(args.model_config)
    image_hw = tuple(int(v) for v in cfg["model"]["input"].get("image_size", [256, 256]))
    device = _resolve_device(str(args.device))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    video, dynamic_mask, particle_mask, manifest = _load_clip(
        manifest_path,
        image_hw=image_hw,
        max_frames=int(args.max_frames),
    )
    frame_idx = int(np.clip(int(args.frame_id), 0, max(0, video.shape[0] - 1)))
    transient_gt = (dynamic_mask | particle_mask).astype(bool)
    aqua = _run_aqua(
        video=video,
        manifest=manifest,
        model_config=Path(args.model_config),
        ckpt_path=Path(args.ckpt_path),
        device=device,
        grid_stride=int(args.grid_stride),
        query_chunk_size=int(args.query_chunk_size),
    )
    coord_txy = aqua["coord_txy"]
    labels_transient = transient_gt[coord_txy[:, 0], coord_txy[:, 2], coord_txy[:, 1]]
    raw_keep = np.ones_like(labels_transient, dtype=bool)
    aqua_keep = aqua["static_probs"] >= float(args.static_threshold)
    raw_metrics = _query_metrics(raw_keep, labels_transient)
    aqua_metrics = _query_metrics(aqua_keep, labels_transient)

    panel_hw = (int(args.panel_size), int(args.panel_size))
    input_panel = _title(
        _fit_panel(cv2.cvtColor(video[frame_idx], cv2.COLOR_RGB2BGR), panel_hw),
        "Input",
        "dynamic underwater frame",
    )
    raw_query = _draw_queries(
        video[frame_idx],
        coord_txy,
        labels_transient,
        raw_keep,
        frame_idx,
        draw_rejected=False,
        point_radius=int(args.point_radius),
    )
    raw_panel = _title(
        _fit_panel(raw_query, panel_hw),
        "Raw D4RT Query Map",
        f"{100.0 * raw_metrics['contamination']:.2f}% transient contamination",
    )
    aqua_query = _draw_queries(
        video[frame_idx],
        coord_txy,
        labels_transient,
        aqua_keep,
        frame_idx,
        draw_rejected=True,
        point_radius=int(args.point_radius),
    )
    aqua_panel = _title(
        _fit_panel(aqua_query, panel_hw),
        "Aqua Static Query Map",
        f"{100.0 * aqua_metrics['contamination']:.2f}% contamination, {100.0 * aqua_metrics['static_retention']:.2f}% retention",
    )

    h, w = video.shape[1:3]
    zoom_box = _best_zoom_box(transient_gt[frame_idx], width=w, height=h, box_size=int(args.zoom_size))
    raw_boxed = raw_query.copy()
    aqua_boxed = aqua_query.copy()
    x0, y0, x1, y1 = zoom_box
    cv2.rectangle(raw_boxed, (x0, y0), (x1, y1), (20, 20, 20), 2, cv2.LINE_AA)
    cv2.rectangle(aqua_boxed, (x0, y0), (x1, y1), (20, 20, 20), 2, cv2.LINE_AA)
    zoom_raw = _crop_zoom(raw_boxed, zoom_box, (panel_hw[0] // 2, panel_hw[1]))
    zoom_aqua = _crop_zoom(aqua_boxed, zoom_box, (panel_hw[0] // 2, panel_hw[1]))
    zoom_pair = np.concatenate([zoom_raw, zoom_aqua], axis=0)
    dataset_note = ""
    if "webuot" in str(manifest_path).lower() or "webuot" in str(manifest.get("dataset", "")).lower():
        dataset_note = "WebUOT GT is a tracked-target bbox mask."
    numbers = _numbers_panel(
        panel_hw=panel_hw,
        raw_metrics=raw_metrics,
        aqua_metrics=aqua_metrics,
        dataset_note=dataset_note,
    )
    zoom_numbers = np.concatenate([zoom_pair, numbers], axis=1)
    zoom_panel = _title(zoom_numbers, "Zoom + Clip Numbers", "same queries, cleaner static support")

    spacer = np.full((input_panel.shape[0], 10, 3), 255, dtype=np.uint8)
    sheet = np.concatenate([input_panel, spacer, raw_panel, spacer, aqua_panel, spacer, zoom_panel], axis=1)
    png_path = output_dir / "compact_mapping_claim.png"
    pdf_path = output_dir / "compact_mapping_claim.pdf"
    cv2.imwrite(str(png_path), sheet)
    try:
        import matplotlib.pyplot as plt

        fig_w = sheet.shape[1] / 180.0
        fig_h = sheet.shape[0] / 180.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    except Exception as exc:
        print(f"Warning: failed to save PDF ({exc})")
        pdf_path = None

    summary = {
        "manifest": str(manifest_path.resolve()),
        "clip_name": str(manifest.get("name", manifest_path.parent.name)),
        "dataset": str(manifest.get("dataset", "unknown")),
        "frame_id": frame_idx,
        "static_threshold": float(args.static_threshold),
        "grid_stride": int(args.grid_stride),
        "query_metrics": {"raw": raw_metrics, "aqua_static": aqua_metrics},
        "zoom_box_xyxy": [int(v) for v in zoom_box],
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()) if pdf_path is not None else None,
        },
        "notes": [
            "This figure shows query-level static-map cleaning, not RGB image restoration.",
            "Aqua keeps queries with static_confidence >= threshold.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {png_path}")
    if pdf_path is not None:
        print(f"Saved: {pdf_path}")
    print(f"Saved: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
