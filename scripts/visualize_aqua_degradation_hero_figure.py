#!/usr/bin/env python3
"""Create a paper-facing Aqua-D4RT degraded-underwater hero figure.

The figure is designed for the main paper, not as a debug sheet.  It highlights
the reviewer-relevant story:

1. raw D4RT/ORB candidates are contaminated by dynamic underwater transients;
2. a strong detector-box prefilter is clean but too destructive for geometry;
3. Aqua keeps a D4RT-native static query/front-end map with a usable retention
   point.

All numeric captions are read from the R118 metric JSON files, while the visual
panels are generated from the selected manifest, Aqua checkpoint, and cached
GroundingDINO mask.
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
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_aqua_downstream_slam_proxy import (  # noqa: E402
    _detect_features,
    _load_external_pred_mask,
    _make_detector,
    _sample_mask,
)
from eval_aqua_transient_heads import _grid_queries, _load_clip, _load_model, _sigmoid  # noqa: E402
from infer_track_3d import _resolve_device  # noqa: E402
from src.core import load_yaml_config, seed_everything  # noqa: E402
from src.eval.tasks import _encode_model_memory, _run_model_for_queries  # noqa: E402


FONT_REGULAR = Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")
_FONT_CACHE: dict[tuple[int, bool], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    key = (int(size), bool(bold))
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    font_path = FONT_BOLD if bold else FONT_REGULAR
    try:
        font = ImageFont.truetype(str(font_path), int(size))
    except Exception:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _put_text(
    image_bgr: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    size: int,
    color: tuple[int, int, int],
    bold: bool = False,
) -> np.ndarray:
    """Draw antialiased TrueType text on an OpenCV BGR image."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    b, g, r = color
    draw.text(tuple(int(v) for v in xy), str(text), font=_font(size, bold=bold), fill=(int(r), int(g), int(b)))
    image_bgr[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return image_bgr


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


def _load_metric_item(metrics_path: Path, manifest_path: Path) -> dict[str, Any]:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest_abs = str(manifest_path.resolve())
    for item in data:
        if str(Path(item["manifest"]).resolve()) == manifest_abs:
            return item
    raise KeyError(f"Manifest {manifest_abs} not found in {metrics_path}")


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
    dynamic_probs = _sigmoid(pred["dynamic_object_logit"].numpy()).astype(np.float32)
    particle_probs = _sigmoid(pred["particle_logit"].numpy()).astype(np.float32)
    confidence_probs = _sigmoid(pred["confidence"].numpy()).astype(np.float32)
    if "static_confidence" in pred:
        static_probs = pred["static_confidence"].numpy().astype(np.float32)
    else:
        static_probs = confidence_probs * (1.0 - dynamic_probs) * (1.0 - particle_probs)
    return {
        "coord_txy": coord_txy,
        "dynamic_probs": dynamic_probs,
        "particle_probs": particle_probs,
        "static_probs": static_probs,
        "transient_probs": np.maximum(dynamic_probs, particle_probs).astype(np.float32),
    }


def _dense_grid_map(
    coord_txy: np.ndarray,
    values: np.ndarray,
    frame_idx: int,
    height: int,
    width: int,
    stride: int,
    *,
    smooth: bool,
) -> np.ndarray:
    mask = coord_txy[:, 0] == int(frame_idx)
    out = np.zeros((height, width), dtype=np.float32)
    if not np.any(mask):
        return out
    xy = coord_txy[mask][:, 1:3]
    vals = values[mask].astype(np.float32)
    if smooth:
        xs = np.unique(xy[:, 0])
        ys = np.unique(xy[:, 1])
        coarse = np.zeros((len(ys), len(xs)), dtype=np.float32)
        x_to_i = {int(x): i for i, x in enumerate(xs)}
        y_to_i = {int(y): i for i, y in enumerate(ys)}
        for (x, y), value in zip(xy, vals):
            coarse[y_to_i[int(y)], x_to_i[int(x)]] = float(value)
        dense = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
        dense = cv2.GaussianBlur(dense, (7, 7), 0)
        return np.clip(dense, 0.0, 1.0).astype(np.float32)
    out[xy[:, 1], xy[:, 0]] = vals
    kernel = np.ones((max(3, int(stride)), max(3, int(stride))), np.uint8)
    return cv2.dilate(out, kernel).astype(np.float32)


def _fit_panel(image_bgr: np.ndarray, panel_hw: tuple[int, int], bg: int = 250) -> np.ndarray:
    ph, pw = panel_hw
    h, w = image_bgr.shape[:2]
    scale = min(float(pw) / float(max(1, w)), float(ph) / float(max(1, h)))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=interp)
    canvas = np.full((ph, pw, 3), int(bg), dtype=np.uint8)
    y0 = (ph - nh) // 2
    x0 = (pw - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _panel(title: str, subtitle: str, body_bgr: np.ndarray, *, top_h: int = 58) -> np.ndarray:
    h, w = body_bgr.shape[:2]
    top = np.full((top_h, w, 3), 255, dtype=np.uint8)
    _put_text(top, title, (12, 6), size=22, color=(18, 18, 18), bold=True)
    if subtitle:
        _put_text(top, subtitle, (12, 35), size=15, color=(76, 76, 76), bold=False)
    out = np.concatenate([top, body_bgr], axis=0)
    cv2.rectangle(out, (0, 0), (w - 1, h + top_h - 1), (212, 212, 212), 1, cv2.LINE_AA)
    return out


def _draw_callout_box(image_bgr: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> np.ndarray:
    out = image_bgr.copy()
    x0, y0, x1, y1 = box
    cv2.rectangle(out, (x0, y0), (x1, y1), color, 3, cv2.LINE_AA)
    cv2.rectangle(out, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _transient_overlay(frame_rgb: np.ndarray, transient_mask: np.ndarray, dino_mask: np.ndarray | None = None) -> np.ndarray:
    base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    out = base.copy()
    gt = transient_mask.astype(bool)
    out[gt] = (40, 60, 238)
    if dino_mask is not None:
        over = dino_mask.astype(bool)
        out[over & ~gt] = (58, 168, 255)
        out[over & gt] = (80, 45, 225)
        edge = cv2.morphologyEx(over.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
        out[edge] = (0, 200, 255)
    edge_gt = cv2.morphologyEx(gt.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    out[edge_gt] = (25, 25, 255)
    return cv2.addWeighted(out, 0.45, base, 0.55, 0.0)


def _draw_queries(
    frame_rgb: np.ndarray,
    coord_txy: np.ndarray,
    labels_transient: np.ndarray,
    keep: np.ndarray,
    frame_idx: int,
    *,
    point_radius: int,
    draw_rejected: bool,
) -> np.ndarray:
    out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    frame_mask = coord_txy[:, 0] == int(frame_idx)
    xy = coord_txy[frame_mask][:, 1:3]
    labels = labels_transient[frame_mask].astype(bool)
    keep_f = keep[frame_mask].astype(bool)
    if draw_rejected:
        for x, y in xy[~keep_f]:
            cv2.circle(out, (int(x), int(y)), max(1, point_radius - 1), (116, 116, 116), -1, cv2.LINE_AA)
    for (x, y), is_transient in zip(xy[keep_f], labels[keep_f]):
        color = (35, 35, 232) if bool(is_transient) else (62, 213, 111)
        cv2.circle(out, (int(x), int(y)), int(point_radius), color, -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), int(point_radius), (10, 10, 10), 1, cv2.LINE_AA)
    return out


def _static_mask_from_queries(
    *,
    coord_txy: np.ndarray,
    keep: np.ndarray,
    frame_idx: int,
    height: int,
    width: int,
    stride: int,
) -> np.ndarray:
    vals = np.zeros((coord_txy.shape[0],), dtype=np.float32)
    vals[keep.astype(bool)] = 1.0
    return _dense_grid_map(coord_txy, vals, frame_idx, height, width, stride, smooth=False) > 0.5


def _dino_static_query_keep(
    *,
    coord_txy: np.ndarray,
    dino_mask: np.ndarray,
) -> np.ndarray:
    t = np.clip(coord_txy[:, 0], 0, dino_mask.shape[0] - 1)
    x = np.clip(coord_txy[:, 1], 0, dino_mask.shape[2] - 1)
    y = np.clip(coord_txy[:, 2], 0, dino_mask.shape[1] - 1)
    return ~dino_mask[t, y, x].astype(bool)


def _draw_features(
    frame_rgb: np.ndarray,
    static_mask: np.ndarray,
    transient_gt: np.ndarray,
    frame_idx: int,
    detector: Any,
    max_features: int,
    *,
    dim_rejected: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    keypoints, _ = _detect_features(
        frame_rgb=frame_rgb,
        static_mask=static_mask.astype(bool),
        detector=detector,
        max_features=int(max_features),
    )
    pts = np.asarray([kp.pt for kp in keypoints], dtype=np.float32) if keypoints else np.zeros((0, 2), dtype=np.float32)
    contaminated = _sample_mask(transient_gt, pts, frame_idx)
    out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if dim_rejected and not bool(static_mask.all()):
        dim = np.clip(out.astype(np.float32) * 0.25 + np.array([34, 34, 34], dtype=np.float32), 0, 255).astype(np.uint8)
        out = np.where(static_mask[..., None].astype(bool), out, dim)
    for pt, bad in zip(pts, contaminated):
        color = (30, 30, 235) if bool(bad) else (255, 214, 75)
        cv2.circle(out, (int(round(pt[0])), int(round(pt[1]))), 3, color, 1, cv2.LINE_AA)
    return out, {
        "num_features": int(len(keypoints)),
        "feature_contamination": float(contaminated.mean()) if contaminated.size else 0.0,
    }


def _choose_zoom_box(transient_mask: np.ndarray, frame_rgb: np.ndarray, size: int) -> tuple[int, int, int, int]:
    h, w = transient_mask.shape[:2]
    gt = transient_mask.astype(np.float32)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    texture = cv2.Laplacian(gray, cv2.CV_32F)
    texture = np.abs(texture)
    score_map = cv2.GaussianBlur(gt, (15, 15), 0) + 0.35 * cv2.GaussianBlur(texture, (15, 15), 0)
    win = int(np.clip(size, 64, min(h, w)))
    kernel = np.ones((win, win), dtype=np.float32)
    score = cv2.filter2D(score_map, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    _, _, _, max_loc = cv2.minMaxLoc(score)
    cx, cy = max_loc
    x0 = int(np.clip(cx - win // 2, 0, max(0, w - win)))
    y0 = int(np.clip(cy - win // 2, 0, max(0, h - win)))
    return x0, y0, x0 + win, y0 + win


def _crop(image_bgr: np.ndarray, box: tuple[int, int, int, int], panel_hw: tuple[int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return _fit_panel(image_bgr[y0:y1, x0:x1], panel_hw)


def _heatmap_overlay(frame_rgb: np.ndarray, score_map: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    heat = cv2.applyColorMap((np.clip(score_map, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return cv2.addWeighted(base, 0.62, heat, 0.38, 0.0)


def _summary_strip(width: int, metrics: dict[str, dict[str, float]], case_name: str) -> np.ndarray:
    h = 146
    canvas = np.full((h, width, 3), 255, dtype=np.uint8)
    _put_text(
        canvas,
        "Aqua-D4RT removes transient contamination while preserving static geometry",
        (18, 10),
        size=26,
        color=(18, 18, 18),
        bold=True,
    )
    _put_text(canvas, case_name, (18, 43), size=15, color=(88, 88, 88))
    _put_text(
        canvas,
        "R118 WebUOT degraded stress test: 100 videos x 5 underwater degradations = 500 clips",
        (18, 66),
        size=15,
        color=(88, 88, 88),
    )
    items = [
        ("Raw", metrics["raw"], (90, 90, 90)),
        ("DINO-box", metrics["dino"], (58, 130, 210)),
        ("Aqua", metrics["aqua"], (40, 155, 98)),
    ]
    x = 20
    y = 94
    for name, vals, color in items:
        text = (
            f"{name}: query contam {100*vals['query_contam']:.1f}%, "
            f"static retain {100*vals['static_ret']:.1f}%, "
            f"ORB contam {100*vals['feat_contam']:.1f}%, "
            f"E {100*vals['e_success']:.0f}%"
        )
        _put_text(canvas, text, (x, y), size=16, color=color)
        y += 19
    return canvas


def _bar_panel(
    *,
    panel_hw: tuple[int, int],
    labels: list[str],
    values: list[float],
    ylabel: str,
    colors: list[tuple[int, int, int]],
    lower_better: bool,
) -> np.ndarray:
    h, w = panel_hw
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 48, 12, 18, 48
    plot_w = w - left - right
    plot_h = h - top - bottom
    cv2.line(canvas, (left, top), (left, top + plot_h), (55, 55, 55), 1, cv2.LINE_AA)
    cv2.line(canvas, (left, top + plot_h), (left + plot_w, top + plot_h), (55, 55, 55), 1, cv2.LINE_AA)
    _put_text(canvas, ylabel, (8, 4), size=12, color=(65, 65, 65))
    _put_text(canvas, "lower better" if lower_better else "higher better", (8, h - 20), size=11, color=(105, 105, 105))
    bar_w = max(22, plot_w // (len(values) * 2))
    gap = (plot_w - bar_w * len(values)) // max(1, len(values) + 1)
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        x0 = left + gap + i * (bar_w + gap)
        bh = int(round(np.clip(value, 0.0, 1.0) * plot_h))
        y0 = top + plot_h - bh
        cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, top + plot_h), color, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, top + plot_h), (40, 40, 40), 1, cv2.LINE_AA)
        _put_text(canvas, f"{100*value:.0f}", (x0 - 1, max(2, y0 - 18)), size=12, color=(35, 35, 35))
        _put_text(canvas, label, (x0 - 4, top + plot_h + 6), size=11, color=(55, 55, 55))
    return canvas


def _metric_panel(panel_hw: tuple[int, int], metrics: dict[str, dict[str, float]]) -> np.ndarray:
    labels = ["Raw", "DINO", "Aqua"]
    colors = [(150, 150, 150), (58, 165, 230), (70, 190, 125)]
    q = [metrics["raw"]["query_contam"], metrics["dino"]["query_contam"], metrics["aqua"]["query_contam"]]
    e = [metrics["raw"]["e_success"], metrics["dino"]["e_success"], metrics["aqua"]["e_success"]]
    top = _bar_panel(panel_hw=(panel_hw[0] // 2, panel_hw[1]), labels=labels, values=q,
                     ylabel="Query contamination (%)", colors=colors, lower_better=True)
    bottom = _bar_panel(panel_hw=(panel_hw[0] // 2, panel_hw[1]), labels=labels, values=e,
                        ylabel="E success (%)", colors=colors, lower_better=False)
    return np.concatenate([top, bottom], axis=0)


def _metric_card(
    panel_hw: tuple[int, int],
    *,
    accent: tuple[int, int, int],
    headline: str,
    big_text: str,
    rows: list[tuple[str, str, tuple[int, int, int]]],
    footer: str,
) -> np.ndarray:
    h, w = panel_hw
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (8, h - 1), accent, -1)
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (226, 226, 226), 1, cv2.LINE_AA)
    _put_text(canvas, headline, (22, 18), size=16, color=(55, 55, 55), bold=True)
    _put_text(canvas, big_text, (22, 52), size=34, color=accent, bold=True)
    y = 112
    for label, value, color in rows:
        _put_text(canvas, label, (24, y), size=14, color=(70, 70, 70), bold=True)
        _put_text(canvas, value, (24, y + 18), size=20, color=color, bold=True)
        y += 60
    _put_text(canvas, footer, (22, h - 46), size=13, color=(92, 92, 92))
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", default="output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt")
    parser.add_argument("--dino-mask-dir", required=True)
    parser.add_argument("--static-per-clip", required=True)
    parser.add_argument("--orb-per-clip", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--frame-id", type=int, default=0)
    parser.add_argument("--grid-stride", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--static-threshold", type=float, default=0.25)
    parser.add_argument("--detector", default="orb", choices=("orb", "sift"))
    parser.add_argument("--max-features", type=int, default=1200)
    parser.add_argument("--panel-size", type=int, default=300)
    parser.add_argument("--zoom-size", type=int, default=112)
    parser.add_argument("--point-radius", type=int, default=2)
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
    frame_idx = int(np.clip(int(args.frame_id), 0, video.shape[0] - 1))
    h, w = video.shape[1:3]
    transient_gt = (dynamic_mask | particle_mask).astype(bool)
    dino_mask, dino_source = _load_external_pred_mask(
        mask_dir=Path(args.dino_mask_dir),
        manifest_path=manifest_path,
        manifest=manifest,
        image_hw=image_hw,
        num_frames=int(video.shape[0]),
    )

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
    dino_keep = _dino_static_query_keep(coord_txy=coord_txy, dino_mask=dino_mask)

    raw_metrics_local = _query_metrics(raw_keep, labels_transient)
    aqua_metrics_local = _query_metrics(aqua_keep, labels_transient)
    dino_metrics_local = _query_metrics(dino_keep, labels_transient)

    static_item = _load_metric_item(Path(args.static_per_clip), manifest_path)
    orb_item = _load_metric_item(Path(args.orb_per_clip), manifest_path)
    aqua_variant = f"aqua_static_conf_ge_{float(args.static_threshold):.3f}".replace(".", "p")
    metrics = {
        "raw": {
            "query_contam": float(static_item["variant_metrics"]["all_d4rt_points"]["point_contamination"]),
            "static_ret": float(static_item["variant_metrics"]["all_d4rt_points"]["point_static_retention"]),
            "feat_contam": float(orb_item["variants"]["raw_all_pixels"]["summary"]["feature_contamination"]),
            "e_success": float(orb_item["variants"]["raw_all_pixels"]["summary"]["essential_success_rate"]),
        },
        "dino": {
            "query_contam": float(static_item["variant_metrics"]["dino_box_static"]["point_contamination"]),
            "static_ret": float(static_item["variant_metrics"]["dino_box_static"]["point_static_retention"]),
            "feat_contam": float(orb_item["variants"]["dino_box_static"]["summary"]["feature_contamination"]),
            "e_success": float(orb_item["variants"]["dino_box_static"]["summary"]["essential_success_rate"]),
        },
        "aqua": {
            "query_contam": float(static_item["variant_metrics"][aqua_variant]["point_contamination"]),
            "static_ret": float(static_item["variant_metrics"][aqua_variant]["point_static_retention"]),
            "feat_contam": float(orb_item["variants"][aqua_variant]["summary"]["feature_contamination"]),
            "e_success": float(orb_item["variants"][aqua_variant]["summary"]["essential_success_rate"]),
        },
    }

    panel_size = int(args.panel_size)
    panel_hw = (panel_size, panel_size)
    zoom_hw = (panel_size, panel_size)
    detector, _ = _make_detector(str(args.detector), int(args.max_features))
    raw_static_mask = np.ones((h, w), dtype=bool)
    aqua_static_mask = _static_mask_from_queries(
        coord_txy=coord_txy,
        keep=aqua_keep,
        frame_idx=frame_idx,
        height=h,
        width=w,
        stride=int(args.grid_stride),
    )
    dino_static_mask = ~dino_mask[frame_idx]

    zoom_box = _choose_zoom_box(transient_gt[frame_idx], video[frame_idx], int(args.zoom_size))

    input_bgr = _draw_callout_box(cv2.cvtColor(video[frame_idx], cv2.COLOR_RGB2BGR), zoom_box, (40, 40, 40))
    input_panel = _panel(
        "Input",
        "degraded underwater dynamic frame",
        _fit_panel(input_bgr, panel_hw),
    )

    gt_panel = _panel(
        "Transient / Detector Mask",
        "red=GT bbox transient, cyan=DINO-box",
        _fit_panel(_draw_callout_box(_transient_overlay(video[frame_idx], transient_gt[frame_idx], dino_mask[frame_idx]), zoom_box, (40, 40, 40)), panel_hw),
    )

    raw_query_img = _draw_callout_box(
        _draw_queries(video[frame_idx], coord_txy, labels_transient, raw_keep, frame_idx,
                      point_radius=int(args.point_radius), draw_rejected=False),
        zoom_box,
        (40, 40, 40),
    )
    raw_panel = _panel(
        "Raw D4RT Queries",
        f"green=static, red=transient ({100.0 * metrics['raw']['query_contam']:.1f}%)",
        _fit_panel(raw_query_img, panel_hw),
    )

    dino_query_img = _draw_callout_box(
        _draw_queries(video[frame_idx], coord_txy, labels_transient, dino_keep, frame_idx,
                      point_radius=int(args.point_radius), draw_rejected=True),
        zoom_box,
        (40, 40, 40),
    )
    dino_panel = _panel(
        "DINO-box Static Map",
        f"gray=rejected; only {100.0 * metrics['dino']['static_ret']:.1f}% static retention",
        _fit_panel(dino_query_img, panel_hw),
    )

    aqua_query_img = _draw_callout_box(
        _draw_queries(video[frame_idx], coord_txy, labels_transient, aqua_keep, frame_idx,
                      point_radius=int(args.point_radius), draw_rejected=True),
        zoom_box,
        (40, 40, 40),
    )
    aqua_panel = _panel(
        "Aqua Static Query Map",
        f"{100.0 * metrics['aqua']['query_contam']:.1f}% contam., {100.0 * metrics['aqua']['static_ret']:.1f}% static retained",
        _fit_panel(aqua_query_img, panel_hw),
    )

    static_map = _dense_grid_map(aqua["coord_txy"], aqua["static_probs"], frame_idx, h, w, int(args.grid_stride), smooth=True)
    heat_panel = _panel(
        "Aqua Static Confidence",
        "D4RT query-level reliability",
        _fit_panel(_draw_callout_box(_heatmap_overlay(video[frame_idx], static_map), zoom_box, (40, 40, 40)), panel_hw),
    )

    raw_feat, raw_frame_metrics = _draw_features(video[frame_idx], raw_static_mask, transient_gt, frame_idx, detector, int(args.max_features))
    dino_feat, dino_frame_metrics = _draw_features(video[frame_idx], dino_static_mask, transient_gt, frame_idx, detector, int(args.max_features))
    aqua_feat, aqua_frame_metrics = _draw_features(video[frame_idx], aqua_static_mask, transient_gt, frame_idx, detector, int(args.max_features))

    raw_zoom = _panel(
        "Zoom: Raw ORB",
        f"{raw_frame_metrics['num_features']} feat, {100.0 * raw_frame_metrics['feature_contamination']:.1f}% transient",
        _crop(raw_feat, zoom_box, zoom_hw),
    )
    dino_zoom = _panel(
        "Zoom: DINO-box ORB",
        f"{dino_frame_metrics['num_features']} feat, E success {100.0 * metrics['dino']['e_success']:.0f}%",
        _crop(dino_feat, zoom_box, zoom_hw),
    )
    aqua_zoom = _panel(
        "Zoom: Aqua ORB",
        f"{aqua_frame_metrics['num_features']} feat, {100.0 * aqua_frame_metrics['feature_contamination']:.1f}% transient",
        _crop(aqua_feat, zoom_box, zoom_hw),
    )
    card_hw = (raw_zoom.shape[0], panel_size)
    raw_card = _metric_card(
        card_hw,
        accent=(118, 118, 118),
        headline="Raw D4RT",
        big_text="contaminated",
        rows=[
            ("Query", f"{100.0 * metrics['raw']['query_contam']:.1f}% contam.", (90, 90, 90)),
            ("ORB", f"{100.0 * metrics['raw']['feat_contam']:.1f}% contam.", (30, 30, 220)),
            ("E success", f"{100.0 * metrics['raw']['e_success']:.0f}%", (90, 90, 90)),
        ],
        footer="Keeps geometry, but much of it is transient.",
    )
    dino_card = _metric_card(
        card_hw,
        accent=(58, 168, 230),
        headline="DINO-box Prefilter",
        big_text="over-masks",
        rows=[
            ("Query", f"{100.0 * metrics['dino']['query_contam']:.1f}% contam.", (58, 130, 210)),
            ("Static", f"{100.0 * metrics['dino']['static_ret']:.1f}% retained", (210, 90, 40)),
            ("E success", f"{100.0 * metrics['dino']['e_success']:.0f}%", (210, 90, 40)),
        ],
        footer="Clean-looking mask can destroy the map.",
    )
    aqua_card = _metric_card(
        card_hw,
        accent=(62, 180, 105),
        headline="Aqua-D4RT",
        big_text="clean + usable",
        rows=[
            ("Query", f"{100.0 * metrics['aqua']['query_contam']:.1f}% contam.", (62, 180, 105)),
            ("Static", f"{100.0 * metrics['aqua']['static_ret']:.1f}% retained", (62, 180, 105)),
            ("E success", f"{100.0 * metrics['aqua']['e_success']:.0f}%", (62, 180, 105)),
        ],
        footer="Query-level reliability preserves static structure.",
    )

    spacer_v = np.full((input_panel.shape[0], 12, 3), 255, dtype=np.uint8)
    row1 = np.concatenate([input_panel, spacer_v, gt_panel, spacer_v, raw_panel, spacer_v, dino_panel, spacer_v, aqua_panel, spacer_v, heat_panel], axis=1)
    spacer_v2 = np.full((raw_zoom.shape[0], 12, 3), 255, dtype=np.uint8)
    row2 = np.concatenate([raw_zoom, spacer_v2, dino_zoom, spacer_v2, aqua_zoom, spacer_v2, raw_card, spacer_v2, dino_card, spacer_v2, aqua_card], axis=1)
    if row2.shape[1] < row1.shape[1]:
        pad = np.full((row2.shape[0], row1.shape[1] - row2.shape[1], 3), 255, dtype=np.uint8)
        row2 = np.concatenate([row2, pad], axis=1)
    elif row2.shape[1] > row1.shape[1]:
        pad = np.full((row1.shape[0], row2.shape[1] - row1.shape[1], 3), 255, dtype=np.uint8)
        row1 = np.concatenate([row1, pad], axis=1)
    strip = _summary_strip(row1.shape[1], metrics, f"{manifest.get('name', manifest_path.parent.name)} | frame {frame_idx}")
    spacer_h = np.full((14, row1.shape[1], 3), 255, dtype=np.uint8)
    sheet = np.concatenate([strip, spacer_h, row1, spacer_h, row2], axis=0)

    png_path = output_dir / "aqua_degradation_hero.png"
    pdf_path = output_dir / "aqua_degradation_hero.pdf"
    cv2.imwrite(str(png_path), sheet)
    try:
        import matplotlib.pyplot as plt

        fig_w = sheet.shape[1] / 190.0
        fig_h = sheet.shape[0] / 190.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
        plt.close(fig)
    except Exception as exc:
        print(f"Warning: failed to save PDF ({exc})")
        pdf_path = None

    summary = {
        "manifest": str(manifest_path.resolve()),
        "clip_name": str(manifest.get("name", manifest_path.parent.name)),
        "frame_id": int(frame_idx),
        "static_threshold": float(args.static_threshold),
        "grid_stride": int(args.grid_stride),
        "dino_mask_source": dino_source,
        "zoom_box_xyxy": [int(v) for v in zoom_box],
        "clip_metrics_from_json": metrics,
        "query_metrics_recomputed_visual": {
            "raw": raw_metrics_local,
            "dino": dino_metrics_local,
            "aqua": aqua_metrics_local,
        },
        "frame_feature_metrics_recomputed_visual": {
            "raw": raw_frame_metrics,
            "dino": dino_frame_metrics,
            "aqua": aqua_frame_metrics,
        },
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()) if pdf_path is not None else None,
        },
        "notes": [
            "WebUOT GT masks are tracked-target bounding boxes; the visual is for query-map/front-end cleanliness, not fish contour segmentation.",
            "DINO-box is included as a strong detector prefilter baseline and can have lower contamination by over-masking static geometry.",
            "Aqua numbers use static_confidence threshold selected for R118 front-end operating point.",
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
