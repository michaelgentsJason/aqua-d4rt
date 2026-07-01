#!/usr/bin/env python3
"""Build synthetic underwater transient clips with fish cutouts and particles."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _frame_sort_key(path: Path) -> tuple[str, int, str]:
    match = re.search(r"(\d+)(?=\D*$)", path.stem)
    frame_id = int(match.group(1)) if match else -1
    prefix = path.stem[: match.start(1)] if match else path.stem
    return prefix, frame_id, path.name


def _load_background_frames(path: Path, max_frames: int) -> tuple[np.ndarray, dict[str, Any]]:
    path = path.expanduser()
    if path.is_file() and path.name == "manifest.json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        frame_paths = [Path(item) for item in manifest["frames"]]
        source = str(path)
    else:
        frame_dir = path / "frames" if (path / "frames").exists() else path
        exts = {".png", ".jpg", ".jpeg"}
        frame_paths = [item for item in frame_dir.iterdir() if item.is_file() and item.suffix.lower() in exts]
        frame_paths.sort(key=_frame_sort_key)
        manifest = {"frames_dir": str(frame_dir.resolve())}
        source = str(frame_dir)
    if max_frames > 0:
        frame_paths = frame_paths[: int(max_frames)]
    if not frame_paths:
        raise RuntimeError(f"No background frames found under {path}")

    frames: list[np.ndarray] = []
    for frame_path in frame_paths:
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read background frame: {frame_path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0), {"source": source, "used_frames": [str(p) for p in frame_paths], **manifest}


def _polygon_to_mask(height: int, width: int, segmentation: Any) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not isinstance(segmentation, list):
        return mask
    for poly in segmentation:
        arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] < 3:
            continue
        pts = np.rint(arr).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, max(width - 1, 0))
        pts[:, 1] = np.clip(pts[:, 1], 0, max(height - 1, 0))
        cv2.fillPoly(mask, [pts], 255)
    return mask


def _load_fish_cutout(entry: dict[str, Any], ann: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    image_bgr = cv2.imread(str(entry["image_path"]), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    mask = _polygon_to_mask(height, width, ann.get("segmentation"))
    if int(mask.sum()) <= 0:
        return None

    x, y, bw, bh = [float(v) for v in ann.get("bbox", [0, 0, width, height])]
    pad = max(4, int(round(0.15 * max(bw, bh))))
    x0 = max(0, int(math.floor(x)) - pad)
    y0 = max(0, int(math.floor(y)) - pad)
    x1 = min(width, int(math.ceil(x + bw)) + pad)
    y1 = min(height, int(math.ceil(y + bh)) + pad)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = image[y0:y1, x0:x1].copy()
    alpha = mask[y0:y1, x0:x1].copy()
    if int(alpha.sum()) <= 0:
        return None
    if bool(rng.random() < 0.5):
        crop = np.ascontiguousarray(crop[:, ::-1])
        alpha = np.ascontiguousarray(alpha[:, ::-1])
    return crop, alpha


def _apply_underwater_degradation(frames: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    arr = frames.astype(np.float32) / 255.0
    strength = float(np.clip(strength, 0.0, 1.0))
    red_attenuation = float(rng.uniform(0.52, 0.78))
    green_gain = float(rng.uniform(0.88, 1.04))
    blue_gain = float(rng.uniform(0.96, 1.12))
    channel = np.asarray([red_attenuation, green_gain, blue_gain], dtype=np.float32).reshape(1, 1, 1, 3)
    water = np.asarray([0.04, 0.36, 0.46], dtype=np.float32).reshape(1, 1, 1, 3)
    haze = float(rng.uniform(0.05, 0.22) * strength)
    contrast = float(rng.uniform(0.72, 0.96))
    out = arr * channel
    out = out * (1.0 - haze) + water * haze
    out = (out - 0.5) * contrast + 0.5
    out = np.clip(out, 0.0, 1.0)
    if strength > 0.4 and bool(rng.random() < 0.6):
        k = int(rng.choice([3, 5]))
        out_u8 = (out * 255.0).astype(np.uint8)
        out_u8 = np.stack([cv2.GaussianBlur(frame, (k, k), sigmaX=0.0) for frame in out_u8], axis=0)
        return out_u8
    return (out * 255.0).astype(np.uint8)


def _blend_cutout(frame: np.ndarray, cutout: np.ndarray, alpha: np.ndarray, x: int, y: int, opacity: float) -> np.ndarray:
    h, w = frame.shape[:2]
    ch, cw = cutout.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(w, int(x) + cw)
    y1 = min(h, int(y) + ch)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((h, w), dtype=bool)

    sx0 = x0 - int(x)
    sy0 = y0 - int(y)
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)
    local_alpha = (alpha[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0) * float(opacity)
    mask = local_alpha > 0.05
    if not bool(mask.any()):
        return np.zeros((h, w), dtype=bool)

    a = local_alpha[:, :, None]
    patch = frame[y0:y1, x0:x1].astype(np.float32)
    fish = cutout[sy0:sy1, sx0:sx1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.clip(patch * (1.0 - a) + fish * a, 0, 255).astype(np.uint8)
    out_mask = np.zeros((h, w), dtype=bool)
    out_mask[y0:y1, x0:x1] = mask
    return out_mask


def _draw_particles(
    frame: np.ndarray,
    rng: np.random.Generator,
    count: int,
    radius_min: float,
    radius_max: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(max(0, int(count))):
        x = int(rng.integers(0, max(w, 1)))
        y = int(rng.integers(0, max(h, 1)))
        radius = float(rng.uniform(radius_min, radius_max))
        alpha = float(rng.uniform(0.20, 0.72))
        color = tuple(int(v) for v in rng.integers([145, 180, 190], [235, 255, 255]))
        if bool(rng.random() < 0.24):
            length = int(rng.uniform(3, 12))
            angle = float(rng.uniform(-0.9, 0.9))
            x2 = int(round(x + math.cos(angle) * length))
            y2 = int(round(y + math.sin(angle) * length))
            cv2.line(overlay, (x, y), (x2, y2), color, thickness=max(1, int(round(radius))))
            cv2.line(mask, (x, y), (x2, y2), 255, thickness=max(1, int(round(radius))))
        else:
            cv2.circle(overlay, (x, y), max(1, int(round(radius))), color, thickness=-1)
            cv2.circle(mask, (x, y), max(1, int(round(radius))), 255, thickness=-1)
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)
    if bool(mask.any()):
        mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=0.0)
    return mask > 8


def _write_mp4(frame_paths: list[Path], output_path: Path, fps: float) -> bool:
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        return False
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        return False
    try:
        for path in frame_paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            writer.write(frame)
    finally:
        writer.release()
    return output_path.exists() and output_path.stat().st_size > 0


def _write_mask_preview(corrupt_frames: np.ndarray, dynamic_masks: np.ndarray, particle_masks: np.ndarray, output_path: Path, fps: float) -> bool:
    h, w = corrupt_frames.shape[1:3]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        return False
    try:
        for frame_rgb, dyn, par in zip(corrupt_frames, dynamic_masks, particle_masks):
            vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            tint = vis.copy()
            tint[dyn] = (40, 80, 255)
            tint[par] = (255, 230, 80)
            vis = cv2.addWeighted(tint, 0.38, vis, 0.62, 0.0)
            writer.write(vis)
    finally:
        writer.release()
    return output_path.exists() and output_path.stat().st_size > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", default="data/aqua_smoke/underwater_caves_sonar_32")
    parser.add_argument("--fish-manifest", default="data/watermask_uiis/fish_subset_train_0200.json")
    parser.add_argument("--output-dir", default="data/aqua_synth/watermask_caves_32")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--output-width", type=int, default=384)
    parser.add_argument("--output-height", type=int, default=288)
    parser.add_argument("--fish-tracks", type=int, default=4)
    parser.add_argument("--fish-scale-min", type=float, default=0.18)
    parser.add_argument("--fish-scale-max", type=float, default=0.36)
    parser.add_argument("--particles-min", type=int, default=60)
    parser.add_argument("--particles-max", type=int, default=180)
    parser.add_argument("--particle-radius-min", type=float, default=0.7)
    parser.add_argument("--particle-radius-max", type=float, default=2.4)
    parser.add_argument("--underwater-strength", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    clean_dir = output_dir / "frames_clean"
    dynamic_dir = output_dir / "masks" / "dynamic_object"
    particle_dir = output_dir / "masks" / "particle"
    transient_dir = output_dir / "masks" / "transient"
    for path in (frames_dir, clean_dir, dynamic_dir, particle_dir, transient_dir):
        path.mkdir(parents=True, exist_ok=True)

    bg_frames, bg_manifest = _load_background_frames(Path(args.background), max_frames=int(args.num_frames))
    out_hw = (max(1, int(args.output_height)), max(1, int(args.output_width)))
    clean_frames = np.stack(
        [cv2.resize(frame, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA) for frame in bg_frames],
        axis=0,
    )
    corrupt_frames = _apply_underwater_degradation(clean_frames, rng=rng, strength=float(args.underwater_strength))

    fish_manifest = json.loads(Path(args.fish_manifest).read_text(encoding="utf-8"))
    entries = [entry for entry in fish_manifest["entries"] if entry.get("annotations")]
    if not entries:
        raise RuntimeError(f"No fish entries found in {args.fish_manifest}")

    t, h, w = corrupt_frames.shape[:3]
    dynamic_masks = np.zeros((t, h, w), dtype=bool)
    particle_masks = np.zeros((t, h, w), dtype=bool)
    track_records: list[dict[str, Any]] = []
    for track_idx in range(max(0, int(args.fish_tracks))):
        cutout = None
        chosen_entry = None
        chosen_ann = None
        for _ in range(64):
            chosen_entry = entries[int(rng.integers(0, len(entries)))]
            anns = chosen_entry["annotations"]
            chosen_ann = anns[int(rng.integers(0, len(anns)))]
            cutout = _load_fish_cutout(chosen_entry, chosen_ann, rng)
            if cutout is not None:
                break
        if cutout is None or chosen_entry is None or chosen_ann is None:
            continue
        fish_rgb, fish_alpha = cutout
        target_width = int(round(float(rng.uniform(args.fish_scale_min, args.fish_scale_max)) * w))
        scale = target_width / max(1, fish_rgb.shape[1])
        target_height = max(1, int(round(fish_rgb.shape[0] * scale)))
        target_width = max(1, target_width)
        fish_rgb = cv2.resize(fish_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
        fish_alpha = cv2.resize(fish_alpha, (target_width, target_height), interpolation=cv2.INTER_AREA)

        y0 = float(rng.uniform(0.15 * h, 0.78 * h))
        x_start = float(rng.choice([-target_width - rng.uniform(0, 0.25 * w), w + rng.uniform(0, 0.25 * w)]))
        direction = -1.0 if x_start > w else 1.0
        travel = float(rng.uniform(0.75 * w, 1.35 * w)) * direction
        opacity = float(rng.uniform(0.68, 0.96))
        phase = float(rng.uniform(0, 2.0 * math.pi))
        centers: list[list[float]] = []
        for ti in range(t):
            frac = float(ti) / float(max(t - 1, 1))
            cx = x_start + travel * frac
            cy = y0 + math.sin(frac * 2.0 * math.pi + phase) * float(rng.uniform(0.02 * h, 0.07 * h))
            x = int(round(cx - 0.5 * target_width))
            y = int(round(cy - 0.5 * target_height))
            mask = _blend_cutout(corrupt_frames[ti], fish_rgb, fish_alpha, x=x, y=y, opacity=opacity)
            dynamic_masks[ti] |= mask
            centers.append([float(cx), float(cy)])
        track_records.append(
            {
                "track_id": track_idx,
                "source_image": chosen_entry["file_name"],
                "source_annotation_id": int(chosen_ann["id"]),
                "target_size_wh": [int(target_width), int(target_height)],
                "opacity": opacity,
                "centers_xy": centers,
            }
        )

    for ti in range(t):
        count = int(rng.integers(int(args.particles_min), int(args.particles_max) + 1))
        particle_masks[ti] = _draw_particles(
            corrupt_frames[ti],
            rng=rng,
            count=count,
            radius_min=float(args.particle_radius_min),
            radius_max=float(args.particle_radius_max),
        )

    transient_masks = dynamic_masks | particle_masks
    frame_paths: list[Path] = []
    clean_paths: list[Path] = []
    dynamic_paths: list[Path] = []
    particle_paths: list[Path] = []
    transient_paths: list[Path] = []
    for ti in range(t):
        frame_path = frames_dir / f"frame_{ti:06d}.png"
        clean_path = clean_dir / f"frame_{ti:06d}.png"
        dyn_path = dynamic_dir / f"frame_{ti:06d}.png"
        par_path = particle_dir / f"frame_{ti:06d}.png"
        trans_path = transient_dir / f"frame_{ti:06d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(corrupt_frames[ti], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(clean_path), cv2.cvtColor(clean_frames[ti], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(dyn_path), dynamic_masks[ti].astype(np.uint8) * 255)
        cv2.imwrite(str(par_path), particle_masks[ti].astype(np.uint8) * 255)
        cv2.imwrite(str(trans_path), transient_masks[ti].astype(np.uint8) * 255)
        frame_paths.append(frame_path)
        clean_paths.append(clean_path)
        dynamic_paths.append(dyn_path)
        particle_paths.append(par_path)
        transient_paths.append(trans_path)

    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_path = labels_dir / "transient_masks.npz"
    np.savez_compressed(
        labels_path,
        dynamic_object_mask=dynamic_masks,
        particle_mask=particle_masks,
        transient_mask=transient_masks,
    )
    preview_path = output_dir / "preview_corrupt.mp4"
    mask_preview_path = output_dir / "preview_masks.mp4"
    preview_ok = _write_mp4(frame_paths, preview_path, fps=float(args.fps))
    mask_preview_ok = _write_mask_preview(corrupt_frames, dynamic_masks, particle_masks, mask_preview_path, fps=float(args.fps))

    manifest = {
        "name": output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "background": bg_manifest,
        "fish_manifest": str(Path(args.fish_manifest).resolve()),
        "num_frames": int(t),
        "height": int(h),
        "width": int(w),
        "fps": float(args.fps),
        "frames": [str(path.resolve()) for path in frame_paths],
        "clean_frames": [str(path.resolve()) for path in clean_paths],
        "dynamic_object_masks": [str(path.resolve()) for path in dynamic_paths],
        "particle_masks": [str(path.resolve()) for path in particle_paths],
        "transient_masks": [str(path.resolve()) for path in transient_paths],
        "labels_npz": str(labels_path.resolve()),
        "preview_corrupt_mp4": str(preview_path.resolve()) if preview_ok else None,
        "preview_masks_mp4": str(mask_preview_path.resolve()) if mask_preview_ok else None,
        "fish_tracks": track_records,
        "mask_coverage": {
            "dynamic_object": float(dynamic_masks.mean()),
            "particle": float(particle_masks.mean()),
            "transient": float(transient_masks.mean()),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Synthetic clip: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Labels: {labels_path}")
    print(f"Dynamic coverage: {manifest['mask_coverage']['dynamic_object']:.4f}")
    print(f"Particle coverage: {manifest['mask_coverage']['particle']:.4f}")
    print(f"Transient coverage: {manifest['mask_coverage']['transient']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
