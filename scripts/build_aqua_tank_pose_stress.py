#!/usr/bin/env python3
"""Build GT-pose Tank stress clips for Aqua-D4RT downstream validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from build_aqua_synthetic_transients import (  # noqa: E402
    _draw_particles,
    _load_fish_cutout,
    _write_mask_preview,
    _write_mp4,
)


def _frame_sort_key(path: Path) -> tuple[str, int, str]:
    match = re.search(r"(\d+)(?=\D*$)", path.stem)
    frame_id = int(match.group(1)) if match else -1
    prefix = path.stem[: match.start(1)] if match else path.stem
    return prefix, frame_id, path.name


def _timestamp_from_path(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse timestamp from frame name: {path.name}")
    return int(match.group(1))


def _read_gt_csv(path: Path) -> dict[str, np.ndarray]:
    rows: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                [
                    float(row["timestamp[ns]"]),
                    float(row["pos_x[m]"]),
                    float(row["pos_y[m]"]),
                    float(row["pos_z[m]"]),
                    float(row["orient_qx"]),
                    float(row["orient_qy"]),
                    float(row["orient_qz"]),
                    float(row["orient_qw"]),
                ]
            )
    if not rows:
        raise RuntimeError(f"No GT rows found: {path}")
    arr = np.asarray(rows, dtype=np.float64)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return {
        "timestamp_ns": arr[:, 0],
        "position": arr[:, 1:4],
        "quaternion_xyzw": arr[:, 4:8],
    }


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quaternion(q0.astype(np.float64))
    q1 = _normalize_quaternion(q1.astype(np.float64))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalize_quaternion((1.0 - alpha) * q0 + alpha * q1)
    theta_0 = math.acos(dot)
    theta = theta_0 * float(alpha)
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _normalize_quaternion((s0 * q0) + (s1 * q1))


def _interpolate_gt(gt: dict[str, np.ndarray], timestamp_ns: int) -> tuple[np.ndarray, np.ndarray]:
    times = gt["timestamp_ns"]
    ts = float(timestamp_ns)
    if ts <= float(times[0]):
        return gt["position"][0].copy(), _normalize_quaternion(gt["quaternion_xyzw"][0])
    if ts >= float(times[-1]):
        return gt["position"][-1].copy(), _normalize_quaternion(gt["quaternion_xyzw"][-1])
    idx = int(np.searchsorted(times, ts, side="right"))
    lo = idx - 1
    hi = idx
    denom = float(times[hi] - times[lo])
    alpha = 0.0 if denom <= 0 else (ts - float(times[lo])) / denom
    pos = (1.0 - alpha) * gt["position"][lo] + alpha * gt["position"][hi]
    quat = _slerp(gt["quaternion_xyzw"][lo], gt["quaternion_xyzw"][hi], alpha)
    return pos, quat


def _load_fish_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = [entry for entry in payload.get("entries", []) if entry.get("annotations")]
    if not entries:
        raise RuntimeError(f"No fish entries with annotations found: {path}")
    return entries


def _read_manifest_list(path: str | Path) -> list[str]:
    items: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            items.append(item)
    return items


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
    out = np.zeros((h, w), dtype=bool)
    out[y0:y1, x0:x1] = mask
    return out


def _variant_params(name: str) -> dict[str, Any]:
    table = {
        "clean": {
            "fish_tracks": 0,
            "particles_min": 0,
            "particles_max": 0,
            "fish_scale": (0.0, 0.0),
            "particle_radius": (0.0, 0.0),
        },
        "fish-low": {
            "fish_tracks": 2,
            "particles_min": 15,
            "particles_max": 45,
            "fish_scale": (0.16, 0.28),
            "particle_radius": (0.5, 1.5),
        },
        "fish-med": {
            "fish_tracks": 4,
            "particles_min": 45,
            "particles_max": 100,
            "fish_scale": (0.22, 0.40),
            "particle_radius": (0.7, 2.2),
        },
        "fish-high": {
            "fish_tracks": 7,
            "particles_min": 80,
            "particles_max": 180,
            "fish_scale": (0.28, 0.55),
            "particle_radius": (0.8, 2.8),
        },
        "fish-extreme": {
            "fish_tracks": 10,
            "particles_min": 120,
            "particles_max": 260,
            "fish_scale": (0.34, 0.68),
            "particle_radius": (0.9, 3.0),
        },
        "snow-med": {
            "fish_tracks": 0,
            "particles_min": 110,
            "particles_max": 240,
            "fish_scale": (0.0, 0.0),
            "particle_radius": (0.7, 2.6),
        },
        "snow-high": {
            "fish_tracks": 1,
            "particles_min": 220,
            "particles_max": 420,
            "fish_scale": (0.18, 0.30),
            "particle_radius": (0.8, 3.4),
        },
        "mixed-fish-snow": {
            "fish_tracks": 5,
            "particles_min": 180,
            "particles_max": 360,
            "fish_scale": (0.24, 0.48),
            "particle_radius": (0.8, 3.2),
        },
    }
    if name not in table:
        raise ValueError(f"Unknown variant {name!r}; choose from {', '.join(table)}")
    return table[name]


def _copy_resize_frames(
    frame_paths: list[Path],
    *,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    frames: list[np.ndarray] = []
    for path in frame_paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read frame: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if output_width > 0 and output_height > 0:
            rgb = cv2.resize(rgb, (int(output_width), int(output_height)), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    return np.stack(frames, axis=0)


def _inject_transients(
    frames: np.ndarray,
    *,
    variant: str,
    fish_entries: list[dict[str, Any]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    params = _variant_params(variant)
    out = frames.copy()
    t, h, w = out.shape[:3]
    dynamic = np.zeros((t, h, w), dtype=np.bool_)
    particle = np.zeros((t, h, w), dtype=np.bool_)
    records: list[dict[str, Any]] = []
    for track_idx in range(int(params["fish_tracks"])):
        cutout = None
        chosen_entry = None
        chosen_ann = None
        for _ in range(96):
            chosen_entry = fish_entries[int(rng.integers(0, len(fish_entries)))]
            anns = chosen_entry["annotations"]
            chosen_ann = anns[int(rng.integers(0, len(anns)))]
            cutout = _load_fish_cutout(chosen_entry, chosen_ann, rng)
            if cutout is not None:
                break
        if cutout is None or chosen_entry is None or chosen_ann is None:
            continue
        fish_rgb, fish_alpha = cutout
        scale_lo, scale_hi = params["fish_scale"]
        target_width = int(round(float(rng.uniform(scale_lo, scale_hi)) * w))
        scale = target_width / max(1, fish_rgb.shape[1])
        target_height = max(1, int(round(fish_rgb.shape[0] * scale)))
        target_width = max(1, target_width)
        fish_rgb = cv2.resize(fish_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
        fish_alpha = cv2.resize(fish_alpha, (target_width, target_height), interpolation=cv2.INTER_AREA)
        if bool(rng.random() < 0.45):
            fish_rgb = np.ascontiguousarray(fish_rgb[:, ::-1])
            fish_alpha = np.ascontiguousarray(fish_alpha[:, ::-1])

        y0 = float(rng.uniform(0.12 * h, 0.82 * h))
        x_start = float(rng.choice([-target_width - rng.uniform(0, 0.30 * w), w + rng.uniform(0, 0.30 * w)]))
        direction = -1.0 if x_start > w else 1.0
        travel = float(rng.uniform(0.85 * w, 1.55 * w)) * direction
        opacity = float(rng.uniform(0.70, 0.98))
        phase = float(rng.uniform(0, 2.0 * math.pi))
        centers: list[list[float]] = []
        for ti in range(t):
            frac = float(ti) / float(max(t - 1, 1))
            cx = x_start + travel * frac
            cy = y0 + math.sin(frac * 2.0 * math.pi + phase) * float(rng.uniform(0.02 * h, 0.08 * h))
            x = int(round(cx - 0.5 * target_width))
            y = int(round(cy - 0.5 * target_height))
            mask = _blend_cutout(out[ti], fish_rgb, fish_alpha, x=x, y=y, opacity=opacity)
            dynamic[ti] |= mask
            centers.append([float(cx), float(cy)])
        records.append(
            {
                "track_id": int(track_idx),
                "source_image": chosen_entry.get("file_name"),
                "source_annotation_id": int(chosen_ann.get("id", -1)),
                "target_size_wh": [int(target_width), int(target_height)],
                "opacity": float(opacity),
                "centers_xy": centers,
            }
        )

    radius_min, radius_max = params["particle_radius"]
    for ti in range(t):
        count = int(rng.integers(int(params["particles_min"]), int(params["particles_max"]) + 1))
        if count <= 0:
            continue
        particle[ti] = _draw_particles(
            out[ti],
            rng=rng,
            count=count,
            radius_min=float(radius_min),
            radius_max=float(radius_max),
        )
    return out, dynamic, particle, records


def _write_frames_csv(
    path: Path,
    *,
    frame_paths: list[Path],
    source_paths: list[Path],
    timestamps: list[int],
    gt_records: list[tuple[np.ndarray, np.ndarray]],
    transient_paths: list[Path],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "image_path",
                "source_image_path",
                "timestamp_ns",
                "pos_x",
                "pos_y",
                "pos_z",
                "orient_qx",
                "orient_qy",
                "orient_qz",
                "orient_qw",
                "transient_mask_path",
            ],
        )
        writer.writeheader()
        for idx, (frame_path, source_path, ts, gt, mask_path) in enumerate(
            zip(frame_paths, source_paths, timestamps, gt_records, transient_paths)
        ):
            pos, quat = gt
            writer.writerow(
                {
                    "frame_index": int(idx),
                    "image_path": str(frame_path.resolve()),
                    "source_image_path": str(source_path.resolve()),
                    "timestamp_ns": int(ts),
                    "pos_x": float(pos[0]),
                    "pos_y": float(pos[1]),
                    "pos_z": float(pos[2]),
                    "orient_qx": float(quat[0]),
                    "orient_qy": float(quat[1]),
                    "orient_qz": float(quat[2]),
                    "orient_qw": float(quat[3]),
                    "transient_mask_path": str(mask_path.resolve()),
                }
            )


def _write_variant(
    *,
    variant: str,
    output_dir: Path,
    benchmark_name: str,
    clip_id: str,
    start_index: int,
    stride: int,
    generation_seed: int,
    frames: np.ndarray,
    source_paths: list[Path],
    timestamps: list[int],
    gt: dict[str, np.ndarray],
    fish_entries: list[dict[str, Any]],
    rng: np.random.Generator,
    fps: float,
    source_root: Path,
    write_previews: bool,
) -> dict[str, Any]:
    variant_dir = output_dir / variant
    frames_dir = variant_dir / "frames"
    clean_dir = variant_dir / "frames_clean"
    dynamic_dir = variant_dir / "masks" / "dynamic_object"
    particle_dir = variant_dir / "masks" / "particle"
    transient_dir = variant_dir / "masks" / "transient"
    labels_dir = variant_dir / "labels"
    for path in (frames_dir, clean_dir, dynamic_dir, particle_dir, transient_dir, labels_dir):
        path.mkdir(parents=True, exist_ok=True)

    corrupt, dynamic, particle, fish_records = _inject_transients(
        frames,
        variant=variant,
        fish_entries=fish_entries,
        rng=rng,
    )
    transient = dynamic | particle
    frame_paths: list[Path] = []
    clean_paths: list[Path] = []
    dynamic_paths: list[Path] = []
    particle_paths: list[Path] = []
    transient_paths: list[Path] = []
    gt_records: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in range(corrupt.shape[0]):
        frame_path = frames_dir / f"frame_{idx:06d}.png"
        clean_path = clean_dir / f"frame_{idx:06d}.png"
        dyn_path = dynamic_dir / f"frame_{idx:06d}.png"
        par_path = particle_dir / f"frame_{idx:06d}.png"
        trans_path = transient_dir / f"frame_{idx:06d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(corrupt[idx], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(clean_path), cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(dyn_path), dynamic[idx].astype(np.uint8) * 255)
        cv2.imwrite(str(par_path), particle[idx].astype(np.uint8) * 255)
        cv2.imwrite(str(trans_path), transient[idx].astype(np.uint8) * 255)
        frame_paths.append(frame_path)
        clean_paths.append(clean_path)
        dynamic_paths.append(dyn_path)
        particle_paths.append(par_path)
        transient_paths.append(trans_path)
        gt_records.append(_interpolate_gt(gt, timestamps[idx]))

    labels_path = labels_dir / "transient_masks.npz"
    np.savez_compressed(
        labels_path,
        dynamic_object_mask=dynamic,
        particle_mask=particle,
        transient_mask=transient,
    )
    frames_csv = variant_dir / "frames.csv"
    _write_frames_csv(
        frames_csv,
        frame_paths=frame_paths,
        source_paths=source_paths,
        timestamps=timestamps,
        gt_records=gt_records,
        transient_paths=transient_paths,
    )
    preview_path: Path | None = None
    mask_preview_path: Path | None = None
    if bool(write_previews):
        preview_path = variant_dir / "preview.mp4"
        mask_preview_path = variant_dir / "preview_masks.mp4"
        _write_mp4(frame_paths, preview_path, fps=float(fps))
        _write_mask_preview(corrupt, dynamic, particle, mask_preview_path, fps=float(fps))

    manifest_name = f"{benchmark_name}_{variant}" if not clip_id else f"{benchmark_name}_{clip_id}_{variant}"
    manifest = {
        "name": manifest_name,
        "dataset": benchmark_name,
        "variant": variant,
        "clip_id": clip_id,
        "start_index": int(start_index),
        "stride": int(stride),
        "generation_seed": int(generation_seed),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root.resolve()),
        "num_frames": int(corrupt.shape[0]),
        "height": int(corrupt.shape[1]),
        "width": int(corrupt.shape[2]),
        "fps": float(fps),
        "frames": [str(path.resolve()) for path in frame_paths],
        "clean_frames": [str(path.resolve()) for path in clean_paths],
        "dynamic_object_masks": [str(path.resolve()) for path in dynamic_paths],
        "particle_masks": [str(path.resolve()) for path in particle_paths],
        "transient_masks": [str(path.resolve()) for path in transient_paths],
        "labels_npz": str(labels_path.resolve()),
        "frames_csv": str(frames_csv.resolve()),
        "preview_mp4": str(preview_path.resolve()) if preview_path is not None else None,
        "preview_masks_mp4": str(mask_preview_path.resolve()) if mask_preview_path is not None else None,
        "fish_tracks": fish_records,
        "mask_coverage": {
            "dynamic_object": float(dynamic.mean()) if dynamic.size else 0.0,
            "particle": float(particle.mean()) if particle.size else 0.0,
            "transient": float(transient.mean()) if transient.size else 0.0,
        },
    }
    manifest_path = variant_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    window_dir = variant_dir / "aqua_windows"
    window_dir.mkdir(parents=True, exist_ok=True)
    window_manifests: list[str] = []
    for start in range(0, len(frame_paths), 32):
        stop = min(start + 32, len(frame_paths))
        if stop - start < 2:
            continue
        window_manifest = dict(manifest)
        window_manifest["name"] = f"{manifest['name']}_w{start:04d}_{stop - 1:04d}"
        window_manifest["window_start"] = int(start)
        window_manifest["window_stop_exclusive"] = int(stop)
        window_manifest["frames"] = [str(path.resolve()) for path in frame_paths[start:stop]]
        window_manifest["clean_frames"] = [str(path.resolve()) for path in clean_paths[start:stop]]
        window_manifest["dynamic_object_masks"] = [str(path.resolve()) for path in dynamic_paths[start:stop]]
        window_manifest["particle_masks"] = [str(path.resolve()) for path in particle_paths[start:stop]]
        window_manifest["transient_masks"] = [str(path.resolve()) for path in transient_paths[start:stop]]
        labels_w = window_dir / f"labels_w{start:04d}_{stop - 1:04d}.npz"
        frames_csv_w = window_dir / f"frames_w{start:04d}_{stop - 1:04d}.csv"
        np.savez_compressed(
            labels_w,
            dynamic_object_mask=dynamic[start:stop],
            particle_mask=particle[start:stop],
            transient_mask=transient[start:stop],
        )
        _write_frames_csv(
            frames_csv_w,
            frame_paths=frame_paths[start:stop],
            source_paths=source_paths[start:stop],
            timestamps=timestamps[start:stop],
            gt_records=gt_records[start:stop],
            transient_paths=transient_paths[start:stop],
        )
        window_manifest["labels_npz"] = str(labels_w.resolve())
        window_manifest["frames_csv"] = str(frames_csv_w.resolve())
        path_w = window_dir / f"manifest_w{start:04d}_{stop - 1:04d}.json"
        path_w.write_text(json.dumps(window_manifest, indent=2), encoding="utf-8")
        window_manifests.append(str(path_w.resolve()))
    (variant_dir / "aqua_window_manifests.txt").write_text("\n".join(window_manifests) + "\n", encoding="utf-8")
    return {
        "variant": variant,
        "clip_id": clip_id,
        "start_index": int(start_index),
        "stride": int(stride),
        "generation_seed": int(generation_seed),
        "manifest": str(manifest_path.resolve()),
        "frames_csv": str(frames_csv.resolve()),
        "window_manifest_list": str((variant_dir / "aqua_window_manifests.txt").resolve()),
        "mask_coverage": manifest["mask_coverage"],
        "num_windows": len(window_manifests),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", default="data/real_underwater/tank_short_test/extracted/short_test/IMG_R")
    parser.add_argument("--gt-csv", default="data/real_underwater/tank_short_test/extracted/short_test/gt.csv")
    parser.add_argument("--tank-yaml", default="data/real_underwater/tank_short_test/short_test.yaml")
    parser.add_argument("--fish-manifest", default="data/watermask_uiis/fish_subset_train_0120.json")
    parser.add_argument("--output-root", default="data/real_underwater/tank_pose_stress")
    parser.add_argument("--variants", default="clean,fish-low,fish-med,fish-high,snow-high")
    parser.add_argument("--num-frames", type=int, default=128)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--start-indices",
        default=None,
        help="Comma-separated start indices for an expanded benchmark. Defaults to --start-index.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output-width", type=int, default=0)
    parser.add_argument("--output-height", type=int, default=0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated generation seeds for an expanded benchmark. Defaults to --seed.",
    )
    parser.add_argument("--allow-short-clips", action="store_true")
    parser.add_argument("--no-previews", action="store_true", help="Skip MP4 previews to reduce expanded-benchmark build time and disk use.")
    return parser.parse_args()


def _parse_int_list(value: str | None, *, default: int) -> list[int]:
    if value is None or not str(value).strip():
        return [int(default)]
    out = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not out:
        return [int(default)]
    return out


def _select_frames(
    frame_paths: list[Path],
    *,
    start_index: int,
    stride: int,
    num_frames: int,
    allow_short_clips: bool,
) -> list[Path]:
    selected = frame_paths[int(start_index) :: max(1, int(stride))]
    selected = selected[: int(num_frames)]
    if not selected:
        raise RuntimeError(f"No frames selected from start_index={start_index}")
    if len(selected) < int(num_frames) and not bool(allow_short_clips):
        raise RuntimeError(
            f"Only {len(selected)} frames selected from start_index={start_index}, "
            f"but --num-frames={num_frames}. Use --allow-short-clips to keep it."
        )
    return selected


def main() -> int:
    args = parse_args()
    image_dir = Path(args.image_dir)
    frame_paths = [path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    frame_paths.sort(key=_frame_sort_key)
    gt = _read_gt_csv(Path(args.gt_csv))
    fish_entries = _load_fish_manifest(Path(args.fish_manifest))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    start_indices = _parse_int_list(args.start_indices, default=int(args.start_index))
    generation_seeds = _parse_int_list(args.seeds, default=int(args.seed))
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    multi_clip = len(start_indices) > 1 or len(generation_seeds) > 1
    benchmark_name = output_root.name
    all_records: list[dict[str, Any]] = []
    all_window_manifest_lists: list[str] = []

    for start_index in start_indices:
        selected = _select_frames(
            frame_paths,
            start_index=int(start_index),
            stride=int(args.stride),
            num_frames=int(args.num_frames),
            allow_short_clips=bool(args.allow_short_clips),
        )
        timestamps = [_timestamp_from_path(path) for path in selected]
        frames = _copy_resize_frames(
            selected,
            output_width=int(args.output_width),
            output_height=int(args.output_height),
        )
        for generation_seed in generation_seeds:
            clip_id = f"start{int(start_index):04d}_seed{int(generation_seed)}" if multi_clip else ""
            run_output_root = output_root / clip_id if multi_clip else output_root
            run_output_root.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(int(generation_seed))
            records: list[dict[str, Any]] = []
            for variant in variants:
                label = f"{clip_id}/{variant}" if clip_id else variant
                print(f"Building Tank stress variant: {label}", flush=True)
                records.append(
                    _write_variant(
                        variant=variant,
                        output_dir=run_output_root,
                        benchmark_name=benchmark_name,
                        clip_id=clip_id,
                        start_index=int(start_index),
                        stride=int(args.stride),
                        generation_seed=int(generation_seed),
                        frames=frames,
                        source_paths=selected,
                        timestamps=timestamps,
                        gt=gt,
                        fish_entries=fish_entries,
                        rng=np.random.default_rng(int(rng.integers(1, 2**31 - 1))),
                        fps=float(args.fps),
                        source_root=image_dir,
                        write_previews=not bool(args.no_previews),
                    )
                )
            manifest_list_run = run_output_root / "manifests.txt"
            manifest_list_run.write_text("\n".join(record["manifest"] for record in records) + "\n", encoding="utf-8")
            window_manifest_list_run = run_output_root / "window_manifests.txt"
            run_windows: list[str] = []
            for record in records:
                run_windows.extend(_read_manifest_list(record["window_manifest_list"]))
            window_manifest_list_run.write_text("\n".join(run_windows) + ("\n" if run_windows else ""), encoding="utf-8")
            all_window_manifest_lists.append(str(window_manifest_list_run.resolve()))
            run_summary = {
                "name": run_output_root.name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "image_dir": str(image_dir.resolve()),
                "gt_csv": str(Path(args.gt_csv).resolve()),
                "tank_yaml": str(Path(args.tank_yaml).resolve()),
                "fish_manifest": str(Path(args.fish_manifest).resolve()),
                "num_frames": len(selected),
                "start_index": int(start_index),
                "stride": int(args.stride),
                "seed": int(generation_seed),
                "manifest_list": str(manifest_list_run.resolve()),
                "window_manifest_list": str(window_manifest_list_run.resolve()),
                "variants": records,
            }
            (run_output_root / "benchmark_manifest.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
            all_records.extend(records)

    manifest_list = output_root / "manifests.txt"
    manifest_list.write_text("\n".join(record["manifest"] for record in all_records) + "\n", encoding="utf-8")
    window_manifest_list = output_root / "window_manifests.txt"
    all_windows: list[str] = []
    for path in all_window_manifest_lists:
        all_windows.extend(_read_manifest_list(path))
    window_manifest_list.write_text("\n".join(all_windows) + ("\n" if all_windows else ""), encoding="utf-8")
    summary = {
        "name": output_root.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_dir": str(image_dir.resolve()),
        "gt_csv": str(Path(args.gt_csv).resolve()),
        "tank_yaml": str(Path(args.tank_yaml).resolve()),
        "fish_manifest": str(Path(args.fish_manifest).resolve()),
        "num_frames": int(args.num_frames),
        "start_index": int(start_indices[0]) if len(start_indices) == 1 else None,
        "start_indices": [int(v) for v in start_indices],
        "stride": int(args.stride),
        "seed": int(generation_seeds[0]) if len(generation_seeds) == 1 else None,
        "seeds": [int(v) for v in generation_seeds],
        "manifest_list": str(manifest_list.resolve()),
        "window_manifest_list": str(window_manifest_list.resolve()),
        "variants": all_records,
    }
    (output_root / "benchmark_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Tank pose stress benchmark: {output_root}")
    for record in all_records:
        cov = record["mask_coverage"]
        print(
            f"- {record['variant']}: transient={cov['transient']:.4f} "
            f"dynamic={cov['dynamic_object']:.4f} particle={cov['particle']:.4f}"
        )
    print(f"Manifest list: {manifest_list}")
    print(f"Window manifest list: {window_manifest_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
