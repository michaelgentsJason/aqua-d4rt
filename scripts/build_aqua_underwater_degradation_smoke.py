#!/usr/bin/env python3
"""Build underwater degradation smoke visuals for Aqua-D4RT.

This script is intentionally geometry-preserving: it changes image appearance
only, so existing transient masks should remain pixel-aligned.  It is used for
R116 smoke validation before any full robustness sweep or retraining.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DegradationVariant:
    name: str
    label: str
    severity: float


VARIANTS: tuple[DegradationVariant, ...] = (
    DegradationVariant("uneven_illumination", "Uneven light", 0.72),
    DegradationVariant("turbidity_backscatter", "Turbidity", 0.68),
    DegradationVariant("low_light_noise", "Low light + noise", 0.70),
    DegradationVariant("blur_flicker", "Blur + flicker", 0.62),
    DegradationVariant("combined_hard", "Combined stress", 0.72),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read frame: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"Failed to write frame: {path}")


def _load_masks(manifest: dict[str, Any], num_frames: int, frame_hw: tuple[int, int]) -> dict[str, np.ndarray]:
    labels_path = Path(str(manifest.get("labels_npz", "")))
    if not labels_path.exists():
        raise FileNotFoundError(f"labels_npz not found: {labels_path}")
    labels = np.load(labels_path)
    h, w = frame_hw

    def read_mask(key: str) -> np.ndarray:
        if key not in labels.files:
            return np.zeros((num_frames, h, w), dtype=bool)
        arr = np.asarray(labels[key])[:num_frames].astype(bool)
        if arr.shape[1:3] != (h, w):
            arr = np.stack(
                [
                    cv2.resize(arr[t].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    for t in range(arr.shape[0])
                ],
                axis=0,
            )
        return arr

    dynamic = read_mask("dynamic_object_mask")
    particle = read_mask("particle_mask")
    if "transient_mask" in labels.files:
        transient = read_mask("transient_mask")
    else:
        transient = dynamic | particle
    return {"dynamic": dynamic, "particle": particle, "transient": transient}


def _load_clip(manifest_path: Path, max_frames: int) -> tuple[dict[str, Any], list[np.ndarray], dict[str, np.ndarray]]:
    manifest = _read_json(manifest_path)
    frame_paths = [Path(str(path)) for path in manifest.get("frames", [])]
    if not frame_paths:
        frames_dir = Path(str(manifest.get("frames_dir", "")))
        frame_paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in manifest: {manifest_path}")
    frame_paths = frame_paths[: max(1, int(max_frames))]
    frames = [_read_rgb(path) for path in frame_paths]
    h, w = frames[0].shape[:2]
    masks = _load_masks(manifest, num_frames=len(frames), frame_hw=(h, w))
    manifest["_resolved_frames"] = [str(path.resolve()) for path in frame_paths]
    return manifest, frames, masks


def _smooth_field(h: int, w: int, rng: np.random.Generator, low_res: int = 5) -> np.ndarray:
    small = rng.uniform(0.65, 1.35, size=(low_res, low_res)).astype(np.float32)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(h, w) / 8.0)
    field = field / max(float(field.mean()), 1e-6)
    return field.astype(np.float32)


def _caustic_field(h: int, w: int, frame_idx: int, severity: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    phase = 0.37 * float(frame_idx)
    f1 = np.sin(xx * 0.045 + yy * 0.018 + phase)
    f2 = np.sin(xx * 0.018 - yy * 0.052 - phase * 0.7)
    caustic = 1.0 + float(severity) * 0.18 * (f1 + 0.65 * f2)
    return np.clip(caustic, 0.55, 1.45).astype(np.float32)


def _apply_uneven_illumination(
    rgb: np.ndarray,
    *,
    frame_idx: int,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    img = rgb.astype(np.float32) / 255.0
    field = _smooth_field(h, w, rng, low_res=6)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = w * (0.45 + 0.16 * np.sin(frame_idx * 0.21))
    cy = h * (0.42 + 0.10 * np.cos(frame_idx * 0.17))
    radius = max(h, w) * (0.55 - 0.10 * severity)
    spotlight = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / max(radius * radius, 1.0)))
    vignette = 1.0 - severity * 0.42 * np.clip(((xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2) / ((w / 2.0) ** 2 + (h / 2.0) ** 2), 0.0, 1.0)
    illum = (1.0 - severity * 0.25) + severity * 0.35 * spotlight
    illum = illum * vignette * (0.72 + 0.28 * field) * _caustic_field(h, w, frame_idx, severity)
    out = img * illum[..., None]
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _apply_turbidity_backscatter(
    rgb: np.ndarray,
    *,
    frame_idx: int,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    img = rgb.astype(np.float32) / 255.0
    water = np.asarray([0.05, 0.43, 0.55], dtype=np.float32).reshape(1, 1, 3)
    attenuation = 1.0 - 0.56 * severity
    out = img * attenuation + water * (1.0 - attenuation)
    out = (out - 0.5) * (1.0 - 0.36 * severity) + 0.5
    haze = _smooth_field(h, w, rng, low_res=4)
    haze = np.clip((haze - haze.min()) / max(float(haze.max() - haze.min()), 1e-6), 0.0, 1.0)
    out = out * (1.0 - 0.18 * severity * haze[..., None]) + water * (0.18 * severity * haze[..., None])

    speckles = np.zeros((h, w), dtype=np.float32)
    count = int(round((h * w) * (0.00018 + 0.00042 * severity)))
    for _ in range(max(1, count)):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        r = float(rng.uniform(0.7, 2.2 + 1.2 * severity))
        alpha = float(rng.uniform(0.08, 0.34 + 0.18 * severity))
        ys, xs, local = _disk_mask(h, w, x, y, r)
        speckles[ys, xs] = np.maximum(speckles[ys, xs], local.astype(np.float32) * alpha)
    speckle_color = np.asarray([0.74, 0.92, 1.0], dtype=np.float32).reshape(1, 1, 3)
    out = out * (1.0 - speckles[..., None]) + speckle_color * speckles[..., None]
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _apply_low_light_noise(
    rgb: np.ndarray,
    *,
    frame_idx: int,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    img = rgb.astype(np.float32) / 255.0
    exposure = 1.0 - 0.62 * severity
    exposure *= 1.0 + 0.12 * severity * np.sin(frame_idx * 0.73)
    channel = np.asarray([1.0 - 0.50 * severity, 1.0 - 0.15 * severity, 1.0 + 0.08 * severity], dtype=np.float32)
    out = np.clip(img * exposure * channel.reshape(1, 1, 3), 0.0, 1.0)
    gamma = 1.0 + 0.75 * severity
    out = np.power(np.clip(out, 0.0, 1.0), gamma)
    noise_sigma = 0.018 + 0.045 * severity
    noise = rng.normal(0.0, noise_sigma, size=out.shape).astype(np.float32)
    shot = rng.normal(0.0, 0.05 * severity * np.sqrt(np.clip(out, 0.0, 1.0) + 1e-4), size=out.shape).astype(np.float32)
    out = out + noise + shot
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    length = max(3, int(length))
    if length % 2 == 0:
        length += 1
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    mat = cv2.getRotationMatrix2D((length / 2.0 - 0.5, length / 2.0 - 0.5), float(angle_deg), 1.0)
    kernel = cv2.warpAffine(kernel, mat, (length, length))
    s = float(kernel.sum())
    if s > 0:
        kernel /= s
    return kernel


def _apply_blur_flicker(
    rgb: np.ndarray,
    *,
    frame_idx: int,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    img = rgb.astype(np.float32) / 255.0
    exposure = 1.0 + severity * 0.22 * np.sin(frame_idx * 0.61 + 0.4)
    img = np.clip(img * exposure, 0.0, 1.0)
    length = int(round(3 + severity * 10 + 2 * np.sin(frame_idx * 0.43)))
    angle = -35.0 + 70.0 * (0.5 + 0.5 * np.sin(frame_idx * 0.19))
    kernel = _motion_blur_kernel(length, angle)
    out = cv2.filter2D(img, -1, kernel)
    if frame_idx % 3 == 1:
        k = max(3, int(round(3 + severity * 5)))
        if k % 2 == 0:
            k += 1
        out = cv2.GaussianBlur(out, (k, k), sigmaX=0.6 + 1.4 * severity)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _apply_variant(
    rgb: np.ndarray,
    variant: DegradationVariant,
    *,
    frame_idx: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if variant.name == "uneven_illumination":
        return _apply_uneven_illumination(rgb, frame_idx=frame_idx, severity=variant.severity, rng=rng)
    if variant.name == "turbidity_backscatter":
        return _apply_turbidity_backscatter(rgb, frame_idx=frame_idx, severity=variant.severity, rng=rng)
    if variant.name == "low_light_noise":
        return _apply_low_light_noise(rgb, frame_idx=frame_idx, severity=variant.severity, rng=rng)
    if variant.name == "blur_flicker":
        return _apply_blur_flicker(rgb, frame_idx=frame_idx, severity=variant.severity, rng=rng)
    if variant.name == "combined_hard":
        out = _apply_uneven_illumination(rgb, frame_idx=frame_idx, severity=variant.severity, rng=rng)
        out = _apply_turbidity_backscatter(out, frame_idx=frame_idx, severity=variant.severity * 0.85, rng=rng)
        out = _apply_low_light_noise(out, frame_idx=frame_idx, severity=variant.severity * 0.65, rng=rng)
        out = _apply_blur_flicker(out, frame_idx=frame_idx, severity=variant.severity * 0.55, rng=rng)
        return out
    raise ValueError(f"Unknown variant: {variant.name}")


def _disk_mask(h: int, w: int, cx: float, cy: float, radius: float) -> tuple[slice, slice, np.ndarray]:
    r = max(0.5, float(radius))
    x0 = max(0, int(np.floor(cx - r - 1)))
    x1 = min(w, int(np.ceil(cx + r + 2)))
    y0 = max(0, int(np.floor(cy - r - 1)))
    y1 = min(h, int(np.ceil(cy + r + 2)))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    local = (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2 <= r * r
    return slice(y0, y1), slice(x0, x1), local


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, color_rgb: tuple[int, int, int] = (255, 72, 54)) -> np.ndarray:
    out = rgb.copy()
    mask_b = mask.astype(bool)
    if mask_b.any():
        color = np.asarray(color_rgb, dtype=np.float32).reshape(1, 3)
        out_f = out.astype(np.float32)
        out_f[mask_b] = out_f[mask_b] * 0.55 + color * 0.45
        out = np.clip(out_f, 0, 255).astype(np.uint8)
    return out


def _fit_panel(rgb: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    ph, pw = size_hw
    h, w = rgb.shape[:2]
    scale = min(float(pw) / float(max(1, w)), float(ph) / float(max(1, h)))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((ph, pw, 3), 245, dtype=np.uint8)
    y0 = (ph - nh) // 2
    x0 = (pw - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _title_panel(rgb: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    h, w = rgb.shape[:2]
    bar_h = 36 if subtitle else 26
    bar = np.full((bar_h, w, 3), 255, dtype=np.uint8)
    cv2.putText(bar, title[:36], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(bar, subtitle[:52], (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (85, 85, 85), 1, cv2.LINE_AA)
    out = np.concatenate([bar, rgb], axis=0)
    cv2.rectangle(out, (0, 0), (w - 1, h + bar_h - 1), (205, 205, 205), 1, cv2.LINE_AA)
    return out


def _metric_summary(frames: list[np.ndarray]) -> dict[str, float]:
    if not frames:
        return {}
    brightness: list[float] = []
    contrast: list[float] = []
    lap_var: list[float] = []
    red_blue: list[float] = []
    for rgb in frames:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        brightness.append(float(gray.mean()))
        contrast.append(float(gray.std()))
        lap_var.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
        r = float(rgb[..., 0].mean()) + 1e-6
        b = float(rgb[..., 2].mean()) + 1e-6
        red_blue.append(float(r / b))
    return {
        "brightness_mean": float(np.mean(brightness)),
        "contrast_mean": float(np.mean(contrast)),
        "laplacian_var_mean": float(np.mean(lap_var)),
        "red_blue_ratio_mean": float(np.mean(red_blue)),
    }


def _write_video(path: Path, frames_rgb: list[np.ndarray], fps: float) -> bool:
    if not frames_rgb:
        return False
    h, w = frames_rgb[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        return False
    try:
        for rgb in frames_rgb:
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return path.exists() and path.stat().st_size > 0


def _make_contact_sheet(
    *,
    original: list[np.ndarray],
    degraded_by_variant: dict[str, list[np.ndarray]],
    masks: dict[str, np.ndarray],
    variant_lookup: dict[str, DegradationVariant],
    output_path: Path,
    panel_size: int,
    max_frames: int,
) -> None:
    transient = masks["transient"]
    picks = np.linspace(0, len(original) - 1, num=min(max_frames, len(original))).round().astype(int)
    rows: list[np.ndarray] = []
    for frame_idx in picks:
        idx = int(frame_idx)
        row_panels = [
            _title_panel(
                _fit_panel(original[idx], (panel_size, panel_size)),
                f"Original f{idx:02d}",
                "input frame",
            ),
            _title_panel(
                _fit_panel(_overlay_mask(original[idx], transient[idx]), (panel_size, panel_size)),
                "Original + mask",
                "alignment reference",
            ),
        ]
        for name, frames in degraded_by_variant.items():
            variant = variant_lookup[name]
            row_panels.append(
                _title_panel(
                    _fit_panel(_overlay_mask(frames[idx], transient[idx]), (panel_size, panel_size)),
                    variant.label,
                    "same mask overlay",
                )
            )
        rows.append(np.concatenate(row_panels, axis=1))
    sheet = np.concatenate(rows, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def _make_alignment_sheet(
    *,
    original: list[np.ndarray],
    degraded_by_variant: dict[str, list[np.ndarray]],
    masks: dict[str, np.ndarray],
    variant_lookup: dict[str, DegradationVariant],
    output_path: Path,
    panel_size: int,
    frame_idx: int,
) -> None:
    transient = masks["transient"]
    idx = int(np.clip(frame_idx, 0, len(original) - 1))
    panels = [
        _title_panel(_fit_panel(original[idx], (panel_size, panel_size)), "Original", f"frame {idx}"),
        _title_panel(_fit_panel(_overlay_mask(original[idx], transient[idx]), (panel_size, panel_size)), "Original mask", "reference"),
    ]
    for name, frames in degraded_by_variant.items():
        variant = variant_lookup[name]
        panels.append(
            _title_panel(
                _fit_panel(_overlay_mask(frames[idx], transient[idx]), (panel_size, panel_size)),
                variant.label,
                "mask should align",
            )
        )
    sheet = np.concatenate(panels, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def _alignment_report(frames: list[np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, Any]:
    h, w = frames[0].shape[:2]
    n = len(frames)
    report: dict[str, Any] = {
        "num_frames": n,
        "frame_hw": [h, w],
        "all_masks_have_frame_shape": True,
        "mask_coverage": {},
        "nonempty_frames": {},
    }
    for key, arr in masks.items():
        ok = arr.shape == (n, h, w)
        report["all_masks_have_frame_shape"] = bool(report["all_masks_have_frame_shape"] and ok)
        report["mask_coverage"][key] = float(arr.mean()) if arr.size else 0.0
        report["nonempty_frames"][key] = int(np.any(arr, axis=(1, 2)).sum()) if arr.ndim == 3 else 0
    return report


def _write_degraded_manifest(
    *,
    output_path: Path,
    source_manifest_path: Path,
    source_manifest: dict[str, Any],
    variant: DegradationVariant,
    frame_paths: list[Path],
    labels_npz: Path,
    preview_mp4: Path,
) -> None:
    payload = dict(source_manifest)
    payload.update(
        {
            "name": f"{source_manifest.get('name', source_manifest_path.stem)}_{variant.name}",
            "source_manifest": str(source_manifest_path.resolve()),
            "degradation_variant": {
                "name": variant.name,
                "label": variant.label,
                "severity": variant.severity,
                "geometry_preserving": True,
            },
            "frames": [str(path.resolve()) for path in frame_paths],
            "frames_dir": str(frame_paths[0].parent.resolve()) if frame_paths else "",
            "labels_npz": str(labels_npz.resolve()),
            "preview_mp4": str(preview_mp4.resolve()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "notes": [
                "R116 smoke degradation is appearance-only; original masks are reused and should remain pixel-aligned.",
                "Use for degradation robustness evaluation, not as new ground-truth dynamic data.",
            ],
        }
    )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--panel-size", type=int, default=192)
    parser.add_argument("--sheet-frames", type=int, default=4)
    parser.add_argument("--alignment-frame", type=int, default=16)
    parser.add_argument("--variants", default=",".join(v.name for v in VARIANTS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, frames, masks = _load_clip(manifest_path, max_frames=int(args.max_frames))
    clip_name = str(manifest.get("name", manifest_path.parent.name))
    variant_names = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    variant_lookup = {variant.name: variant for variant in VARIANTS}
    missing = [name for name in variant_names if name not in variant_lookup]
    if missing:
        raise ValueError(f"Unknown degradation variants: {missing}")

    labels_npz = output_dir / "labels_reused_masks.npz"
    np.savez_compressed(
        labels_npz,
        dynamic_object_mask=masks["dynamic"],
        particle_mask=masks["particle"],
        transient_mask=masks["transient"],
    )

    degraded_by_variant: dict[str, list[np.ndarray]] = {}
    variant_summaries: dict[str, Any] = {}
    fps = float(manifest.get("fps", 10.0))
    for variant_idx, name in enumerate(variant_names):
        variant = variant_lookup[name]
        rng = np.random.default_rng(int(args.seed) + 10007 * (variant_idx + 1))
        degraded: list[np.ndarray] = []
        for frame_idx, rgb in enumerate(frames):
            degraded.append(_apply_variant(rgb, variant, frame_idx=frame_idx, rng=rng))
        degraded_by_variant[name] = degraded

        variant_dir = output_dir / name
        frames_dir = variant_dir / "frames"
        frame_paths: list[Path] = []
        for frame_idx, rgb in enumerate(degraded):
            path = frames_dir / f"frame_{frame_idx:06d}.png"
            _write_rgb(path, rgb)
            frame_paths.append(path)
        preview_path = variant_dir / "preview.mp4"
        overlay_preview_path = variant_dir / "preview_mask_overlay.mp4"
        _write_video(preview_path, degraded, fps=fps)
        _write_video(
            overlay_preview_path,
            [_overlay_mask(rgb, masks["transient"][idx]) for idx, rgb in enumerate(degraded)],
            fps=fps,
        )
        manifest_out = variant_dir / "manifest.json"
        _write_degraded_manifest(
            output_path=manifest_out,
            source_manifest_path=manifest_path,
            source_manifest=manifest,
            variant=variant,
            frame_paths=frame_paths,
            labels_npz=labels_npz,
            preview_mp4=preview_path,
        )
        variant_summaries[name] = {
            "label": variant.label,
            "severity": variant.severity,
            "frames_dir": str(frames_dir.resolve()),
            "manifest": str(manifest_out.resolve()),
            "preview_mp4": str(preview_path.resolve()),
            "preview_mask_overlay_mp4": str(overlay_preview_path.resolve()),
            "appearance_metrics": _metric_summary(degraded),
        }

    contact_sheet = output_dir / "degradation_contact_sheet.png"
    alignment_sheet = output_dir / "mask_alignment_sheet.png"
    _make_contact_sheet(
        original=frames,
        degraded_by_variant=degraded_by_variant,
        masks=masks,
        variant_lookup=variant_lookup,
        output_path=contact_sheet,
        panel_size=int(args.panel_size),
        max_frames=int(args.sheet_frames),
    )
    _make_alignment_sheet(
        original=frames,
        degraded_by_variant=degraded_by_variant,
        masks=masks,
        variant_lookup=variant_lookup,
        output_path=alignment_sheet,
        panel_size=int(args.panel_size),
        frame_idx=int(args.alignment_frame),
    )

    original_preview = output_dir / "original_preview.mp4"
    original_overlay_preview = output_dir / "original_mask_overlay.mp4"
    _write_video(original_preview, frames, fps=fps)
    _write_video(
        original_overlay_preview,
        [_overlay_mask(rgb, masks["transient"][idx]) for idx, rgb in enumerate(frames)],
        fps=fps,
    )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "clip_name": clip_name,
        "source_manifest": str(manifest_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "geometry_preserving": True,
        "alignment": _alignment_report(frames, masks),
        "original_appearance_metrics": _metric_summary(frames),
        "variants": variant_summaries,
        "outputs": {
            "labels_reused_masks": str(labels_npz.resolve()),
            "contact_sheet": str(contact_sheet.resolve()),
            "mask_alignment_sheet": str(alignment_sheet.resolve()),
            "original_preview_mp4": str(original_preview.resolve()),
            "original_mask_overlay_mp4": str(original_overlay_preview.resolve()),
        },
        "claim_gate": [
            "This is R116 smoke data only; it validates degradation visual realism and mask alignment.",
            "No model result or robustness claim should be made until R117/R118 metrics are run.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {contact_sheet}")
    print(f"Saved {alignment_sheet}")
    print(f"Saved {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
