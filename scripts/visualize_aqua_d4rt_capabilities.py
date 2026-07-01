#!/usr/bin/env python3
"""Generate a D4RT-style capabilities figure for Aqua-D4RT.

The figure is meant for paper/project-page storytelling: Aqua-D4RT inherits
D4RT query tracking and 3D reconstruction, then filters the same query-level
3D representation into a cleaner static map.
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
from infer_track_3d import _grid_query_points, _infer_tracks, _resolve_device  # noqa: E402
from src.core import load_yaml_config, seed_everything  # noqa: E402
from src.eval.tasks import _encode_model_memory, _run_model_for_queries  # noqa: E402


def _safe_rate(num: int | float, den: int | float) -> float:
    return float(num) / float(max(float(den), 1.0))


def _fit(image_bgr: np.ndarray, size_hw: tuple[int, int], *, bg: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    out_h, out_w = size_hw
    h, w = image_bgr.shape[:2]
    scale = min(float(out_w) / float(max(1, w)), float(out_h) / float(max(1, h)))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((out_h, out_w, 3), bg, dtype=np.uint8)
    y0 = (out_h - new_h) // 2
    x0 = (out_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _panel_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    del h
    bar = np.full((28, w, 3), 255, dtype=np.uint8)
    cv2.putText(bar, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (18, 18, 18), 1, cv2.LINE_AA)
    out = np.concatenate([bar, image_bgr], axis=0)
    return out


def _video_strip(video_rgb: np.ndarray, frame_ids: list[int], size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    if not frame_ids:
        return canvas
    thumb_h = int(h * 0.74)
    thumb_w = int(w * 0.48)
    step = max(12, int((w - thumb_w) / max(1, len(frame_ids) - 1)))
    y_offsets = np.linspace(8, max(8, h - thumb_h - 8), num=len(frame_ids)).astype(int)
    for idx, frame_id in enumerate(frame_ids):
        frame = cv2.cvtColor(video_rgb[int(frame_id)], cv2.COLOR_RGB2BGR)
        thumb = _fit(frame, (thumb_h, thumb_w))
        x0 = min(w - thumb_w, idx * step)
        y0 = int(y_offsets[idx])
        shadow = canvas.copy()
        cv2.rectangle(shadow, (x0 + 4, y0 + 4), (x0 + thumb_w + 4, y0 + thumb_h + 4), (220, 220, 220), -1)
        canvas = cv2.addWeighted(shadow, 0.22, canvas, 0.78, 0.0)
        canvas[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = thumb
        cv2.rectangle(canvas, (x0, y0), (x0 + thumb_w - 1, y0 + thumb_h - 1), (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def _query_metrics(keep: np.ndarray, labels_transient: np.ndarray) -> dict[str, Any]:
    keep_b = keep.astype(bool)
    transient = labels_transient.astype(bool)
    static = ~transient
    kept = int(keep_b.sum())
    kept_transient = int(np.logical_and(keep_b, transient).sum())
    kept_static = int(np.logical_and(keep_b, static).sum())
    return {
        "kept": kept,
        "total": int(keep_b.size),
        "contamination": _safe_rate(kept_transient, kept),
        "static_retention": _safe_rate(kept_static, int(static.sum())),
        "kept_rate": _safe_rate(kept, keep_b.size),
    }


def _run_query_map(
    *,
    model: torch.nn.Module,
    video: np.ndarray,
    manifest: dict[str, Any],
    grid_stride: int,
    query_chunk_size: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    query_cpu, coord_txy = _grid_queries(video.shape[0], video.shape[1], video.shape[2], stride=int(grid_stride))
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
        "xyz": pred["xyz_3d"].numpy().astype(np.float32),
        "confidence_probs": confidence_probs,
        "dynamic_probs": dynamic_probs,
        "particle_probs": particle_probs,
        "static_probs": static_probs,
    }


def _draw_tracks(video_rgb: np.ndarray, tracks: dict[str, np.ndarray], target_frame: int, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = video_rgb.shape[1:3]
    frame = cv2.cvtColor(video_rgb[int(target_frame)], cv2.COLOR_RGB2BGR)
    colors = [
        (230, 50, 50),
        (40, 180, 90),
        (40, 120, 230),
        (230, 180, 40),
        (180, 70, 220),
        (30, 190, 190),
        (245, 100, 35),
        (90, 90, 240),
    ]
    uv = np.asarray(tracks["tracks_uv_norm"], dtype=np.float32)
    vis = np.asarray(tracks["tracks_visibility"], dtype=bool)
    for q_idx in range(uv.shape[0]):
        pts: list[tuple[int, int]] = []
        for t in range(min(int(target_frame) + 1, uv.shape[1])):
            if not vis[q_idx, t] or not np.isfinite(uv[q_idx, t]).all():
                continue
            x = int(np.clip(round(float(uv[q_idx, t, 0]) * (w - 1)), 0, w - 1))
            y = int(np.clip(round(float(uv[q_idx, t, 1]) * (h - 1)), 0, h - 1))
            pts.append((x, y))
        if len(pts) < 2:
            continue
        color = colors[q_idx % len(colors)]
        cv2.polylines(frame, [np.asarray(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        cv2.circle(frame, pts[0], 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, pts[0], 4, color, 1, cv2.LINE_AA)
        cv2.circle(frame, pts[-1], 4, color, -1, cv2.LINE_AA)
    return _fit(frame, size_hw)


def _robust_range(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(finite, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or float(hi - lo) < 1e-6:
        center = float(np.mean(finite))
        return center - 1.0, center + 1.0
    pad = 0.08 * float(hi - lo)
    return float(lo - pad), float(hi + pad)


def _project_points(points: np.ndarray, yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    center = np.nanmedian(pts, axis=0)
    pts = pts - center[None, :]
    yaw = np.deg2rad(float(yaw_deg))
    pitch = np.deg2rad(float(pitch_deg))
    ry = np.asarray(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float32,
    )
    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float32,
    )
    view = pts @ (ry @ rx).T
    return view[:, 0], -view[:, 1], view[:, 2]


def _render_cloud(
    *,
    points: np.ndarray,
    colors_rgb: np.ndarray,
    mask: np.ndarray,
    transient: np.ndarray,
    size_hw: tuple[int, int],
    max_points: int,
    seed: int,
    mark_transient: bool,
) -> np.ndarray:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        canvas = np.full((*size_hw, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, f"matplotlib unavailable: {exc}", (10, size_hw[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1)
        return canvas

    valid = np.asarray(mask, dtype=bool) & np.isfinite(points).all(axis=1)
    idx = np.flatnonzero(valid)
    if idx.size > int(max_points):
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(idx, size=int(max_points), replace=False))
    pts = points[idx]
    cols = np.asarray(colors_rgb[idx], dtype=np.float32) / 255.0
    if mark_transient:
        bad = np.asarray(transient[idx], dtype=bool)
        cols[bad] = np.asarray([0.92, 0.12, 0.10], dtype=np.float32)
    x, y, depth = _project_points(pts, yaw_deg=-28.0, pitch_deg=14.0)
    order = np.argsort(depth)
    xlim = _robust_range(x)
    ylim = _robust_range(y)
    fig_w = size_hw[1] / 120.0
    fig_h = size_hw[0] / 120.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    if order.size:
        ax.scatter(x[order], y[order], c=cols[order], s=2.0, linewidths=0, alpha=0.92)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    rgb = rgba[..., :3].copy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _metrics_badge(panel: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    out = panel.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (8, h - 34), (w - 8, h - 8), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(out, (8, h - 34), (w - 8, h - 8), color, 1, cv2.LINE_AA)
    cv2.putText(out, text, (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
    return out


def _caption_block(width: int, captions: list[tuple[str, str]]) -> np.ndarray:
    canvas = np.full((156, width, 3), 250, dtype=np.uint8)
    x = 18
    y = 28
    for title, body in captions:
        cv2.putText(canvas, f"{title}:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, body, (x + 18, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (55, 55, 55), 1, cv2.LINE_AA)
        y += 43
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", default="output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--target-frame", type=int, default=20)
    parser.add_argument("--grid-stride", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--static-threshold", type=float, default=0.55)
    parser.add_argument("--track-query-cols", type=int, default=4)
    parser.add_argument("--track-query-rows", type=int, default=3)
    parser.add_argument("--track-max-queries", type=int, default=12)
    parser.add_argument("--track-margin-ratio", type=float, default=0.14)
    parser.add_argument("--panel-width", type=int, default=270)
    parser.add_argument("--panel-height", type=int, default=150)
    parser.add_argument("--preview-max-points", type=int, default=26000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    device = _resolve_device(str(args.device))
    cfg = load_yaml_config(args.model_config)
    image_hw = tuple(int(v) for v in cfg["model"]["input"].get("image_size", [256, 256]))
    manifest_path = Path(args.manifest)
    video, dynamic_mask, particle_mask, manifest = _load_clip(
        manifest_path,
        image_hw=image_hw,
        max_frames=int(args.max_frames),
    )
    target_frame = int(np.clip(int(args.target_frame), 0, max(0, video.shape[0] - 1)))
    model = _load_model(Path(args.model_config), Path(args.ckpt_path), device=device)
    query_map = _run_query_map(
        model=model,
        video=video,
        manifest=manifest,
        grid_stride=int(args.grid_stride),
        query_chunk_size=int(args.query_chunk_size),
    )
    coord_txy = query_map["coord_txy"]
    labels_dynamic = dynamic_mask[coord_txy[:, 0], coord_txy[:, 2], coord_txy[:, 1]].astype(bool)
    labels_particle = particle_mask[coord_txy[:, 0], coord_txy[:, 2], coord_txy[:, 1]].astype(bool)
    labels_transient = labels_dynamic | labels_particle
    rgb = video[coord_txy[:, 0], coord_txy[:, 2], coord_txy[:, 1]]
    finite = np.isfinite(query_map["xyz"]).all(axis=1)
    raw_keep = finite
    aqua_keep = finite & (query_map["static_probs"] >= float(args.static_threshold))
    raw_metrics = _query_metrics(raw_keep, labels_transient)
    aqua_metrics = _query_metrics(aqua_keep, labels_transient)

    query_px = _grid_query_points(
        width=video.shape[2],
        height=video.shape[1],
        cols=int(args.track_query_cols),
        rows=int(args.track_query_rows),
        margin_ratio=float(args.track_margin_ratio),
        max_points=int(args.track_max_queries),
    )
    denom = np.asarray([max(video.shape[2] - 1, 1), max(video.shape[1] - 1, 1)], dtype=np.float32)
    query_uv_norm = np.clip(query_px / denom[None, :], 0.0, 1.0).astype(np.float32)
    tracks = _infer_tracks(
        model=model,
        video_model_rgb=video,
        query_uv_norm=query_uv_norm,
        query_chunk_size=max(1, min(int(args.query_chunk_size), 256)),
    )

    panel_hw = (int(args.panel_height), int(args.panel_width))
    frame_ids = np.linspace(0, video.shape[0] - 1, num=min(4, video.shape[0])).round().astype(int).tolist()
    input_panel = _panel_title(_video_strip(video, frame_ids, panel_hw), "Input Video")
    tracking_panel = _panel_title(_draw_tracks(video, tracks, target_frame, panel_hw), "3D Tracking")
    raw_cloud = _render_cloud(
        points=query_map["xyz"],
        colors_rgb=rgb,
        mask=raw_keep,
        transient=labels_transient,
        size_hw=panel_hw,
        max_points=int(args.preview_max_points),
        seed=int(args.seed),
        mark_transient=True,
    )
    raw_cloud = _metrics_badge(raw_cloud, f"raw contam {100.0 * raw_metrics['contamination']:.1f}%", (170, 45, 38))
    raw_panel = _panel_title(raw_cloud, "Raw D4RT Reconstruction")
    aqua_cloud = _render_cloud(
        points=query_map["xyz"],
        colors_rgb=rgb,
        mask=aqua_keep,
        transient=labels_transient,
        size_hw=panel_hw,
        max_points=int(args.preview_max_points),
        seed=int(args.seed) + 1,
        mark_transient=False,
    )
    aqua_cloud = _metrics_badge(
        aqua_cloud,
        f"Aqua {100.0 * aqua_metrics['contamination']:.1f}% contam, {100.0 * aqua_metrics['static_retention']:.1f}% retain",
        (15, 118, 100),
    )
    aqua_panel = _panel_title(aqua_cloud, "Aqua Static Reconstruction")

    gap = np.full((input_panel.shape[0], 14, 3), 255, dtype=np.uint8)
    body = np.concatenate([input_panel, gap, tracking_panel, gap, raw_panel, gap, aqua_panel], axis=1)
    header = np.full((58, body.shape[1], 3), 248, dtype=np.uint8)
    cv2.putText(header, "Aqua-D4RT Capabilities", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (22, 22, 22), 2, cv2.LINE_AA)
    captions = _caption_block(
        body.shape[1],
        [
            ("3D Tracking", "D4RT query tracks remain available on underwater clips."),
            ("3D Reconstruction", "Raw D4RT query points can include fish/snow transients."),
            ("Aqua Static Reconstruction", "static_confidence filters the same query map into cleaner static geometry."),
        ],
    )
    sheet = np.concatenate([header, body, captions], axis=0)
    cv2.rectangle(sheet, (0, 0), (sheet.shape[1] - 1, sheet.shape[0] - 1), (225, 225, 225), 1, cv2.LINE_AA)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "aqua_d4rt_capabilities.png"
    pdf_path = output_dir / "aqua_d4rt_capabilities.pdf"
    cv2.imwrite(str(png_path), sheet)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(sheet.shape[1] / 180.0, sheet.shape[0] / 180.0))
        ax.imshow(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    except Exception as exc:
        print(f"Warning: failed to save PDF: {exc}")
        pdf_path = None

    summary = {
        "manifest": str(manifest_path.resolve()),
        "clip_name": str(manifest.get("name", manifest_path.parent.name)),
        "dataset": str(manifest.get("dataset", "unknown")),
        "target_frame": target_frame,
        "grid_stride": int(args.grid_stride),
        "static_threshold": float(args.static_threshold),
        "num_query_points": int(coord_txy.shape[0]),
        "raw_query_metrics": raw_metrics,
        "aqua_static_query_metrics": aqua_metrics,
        "track_queries": int(query_uv_norm.shape[0]),
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()) if pdf_path is not None else None,
        },
        "notes": [
            "This is a D4RT-style capability visualization using real Aqua-D4RT/D4RT query predictions.",
            "3D reconstruction panels are query-level local predictions, not a globally fused SLAM map.",
            "Red points in the raw reconstruction mark available transient labels; WebUOT labels are bbox-level.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {png_path}")
    if pdf_path is not None:
        print(f"Saved {pdf_path}")
    print(f"Saved {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
