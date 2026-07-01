#!/usr/bin/env python3
"""Prepare AQUALOC harbor raw frames as Aqua-D4RT external sanity clips.

The generated manifests are intentionally lightweight: they make AQUALOC
usable by the existing Aqua static-map and ORB proxy evaluators. Clean clips use
all-false transient masks. Optional injected variants place synthetic fish/snow
transients on real AQUALOC backgrounds, so those results must be reported as
external-background stress tests rather than naturally dynamic AQUALOC labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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


def _frame_id(path: Path) -> int:
    match = re.search(r"(\d+)(?=\D*$)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse frame id from {path.name}")
    return int(match.group(1))


def _read_trajectory(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        frame = int(round(float(parts[0])))
        pos = np.asarray([float(v) for v in parts[1:4]], dtype=np.float64)
        quat = np.asarray([float(v) for v in parts[4:8]], dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        if norm > 1e-12:
            quat = quat / norm
        else:
            quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        poses[frame] = (pos, quat)
    if not poses:
        raise RuntimeError(f"No AQUALOC trajectory rows found: {path}")
    return poses


def _nearest_pose(
    poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_id: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if frame_id in poses:
        pos, quat = poses[frame_id]
        return pos.copy(), quat.copy(), int(frame_id)
    keys = np.asarray(sorted(poses), dtype=np.int64)
    nearest = int(keys[int(np.argmin(np.abs(keys - int(frame_id))))])
    pos, quat = poses[nearest]
    return pos.copy(), quat.copy(), nearest


def _load_fish_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = [entry for entry in payload.get("entries", []) if entry.get("annotations")]
    if not entries:
        raise RuntimeError(f"No fish entries with annotations found: {path}")
    return entries


def _read_calib_yaml(path: Path) -> dict[str, Any]:
    # The AQUALOC calibration file is tiny and Kalibr-like. Avoid adding a yaml
    # dependency here; preserve the source file path and parse only common fields.
    text = path.read_text(encoding="utf-8")
    calib: dict[str, Any] = {"source": str(path.resolve())}
    for key in ("intrinsics", "resolution", "distortion_coeffs"):
        match = re.search(rf"{key}:\s*\[([^\]]+)\]", text)
        if match:
            calib[key] = [float(v.strip()) for v in match.group(1).split(",") if v.strip()]
    for key in ("camera_model", "distortion_model", "rostopic"):
        match = re.search(rf"{key}:\s*([^\n]+)", text)
        if match:
            calib[key] = match.group(1).strip()
    return calib


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
    obj = cutout[sy0:sy1, sx0:sx1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.clip(patch * (1.0 - a) + obj * a, 0, 255).astype(np.uint8)
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
        "snow-high": {
            "fish_tracks": 0,
            "particles_min": 220,
            "particles_max": 420,
            "fish_scale": (0.0, 0.0),
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
        target_width = max(1, int(round(float(rng.uniform(scale_lo, scale_hi)) * w)))
        scale = target_width / max(1, fish_rgb.shape[1])
        target_height = max(1, int(round(fish_rgb.shape[0] * scale)))
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
    source_frame_ids: list[int],
    poses: dict[int, tuple[np.ndarray, np.ndarray]],
    transient_paths: list[Path],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "image_path",
                "source_image_path",
                "source_frame_id",
                "nearest_gt_frame_id",
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
        for idx, (frame_path, source_path, source_frame_id, mask_path) in enumerate(
            zip(frame_paths, source_paths, source_frame_ids, transient_paths)
        ):
            pos, quat, nearest = _nearest_pose(poses, source_frame_id)
            writer.writerow(
                {
                    "frame_index": int(idx),
                    "image_path": str(frame_path.resolve()),
                    "source_image_path": str(source_path.resolve()),
                    "source_frame_id": int(source_frame_id),
                    "nearest_gt_frame_id": int(nearest),
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
    output_root: Path,
    clip_name: str,
    selected_paths: list[Path],
    selected_frame_ids: list[int],
    frames: np.ndarray,
    poses: dict[int, tuple[np.ndarray, np.ndarray]],
    calib: dict[str, Any],
    fish_entries: list[dict[str, Any]],
    seed: int,
    fps: float,
    write_previews: bool,
) -> dict[str, Any]:
    variant_dir = output_root / f"{clip_name}_{variant}"
    frames_dir = variant_dir / "frames"
    clean_dir = variant_dir / "frames_clean"
    dynamic_dir = variant_dir / "masks" / "dynamic_object"
    particle_dir = variant_dir / "masks" / "particle"
    transient_dir = variant_dir / "masks" / "transient"
    labels_dir = variant_dir / "labels"
    for path in (frames_dir, clean_dir, dynamic_dir, particle_dir, transient_dir, labels_dir):
        path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed))
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
        source_paths=selected_paths,
        source_frame_ids=selected_frame_ids,
        poses=poses,
        transient_paths=transient_paths,
    )

    preview_path: Path | None = None
    mask_preview_path: Path | None = None
    if write_previews:
        preview_path = variant_dir / "preview.mp4"
        mask_preview_path = variant_dir / "preview_masks.mp4"
        _write_mp4(frame_paths, preview_path, fps=float(fps))
        _write_mask_preview(corrupt, dynamic, particle, mask_preview_path, fps=float(fps))

    manifest = {
        "name": f"{clip_name}_{variant}",
        "dataset": "AQUALOC-harbor-sequence-07-external-sanity",
        "source_dataset": "AQUALOC",
        "sequence": "harbor_sequence_07",
        "variant": variant,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_frames": int(corrupt.shape[0]),
        "height": int(corrupt.shape[1]),
        "width": int(corrupt.shape[2]),
        "fps": float(fps),
        "source_frame_ids": [int(v) for v in selected_frame_ids],
        "frames": [str(path.resolve()) for path in frame_paths],
        "clean_frames": [str(path.resolve()) for path in clean_paths],
        "dynamic_object_masks": [str(path.resolve()) for path in dynamic_paths],
        "particle_masks": [str(path.resolve()) for path in particle_paths],
        "transient_masks": [str(path.resolve()) for path in transient_paths],
        "labels_npz": str(labels_path.resolve()),
        "frames_csv": str(frames_csv.resolve()),
        "camera_calibration": calib,
        "preview_mp4": str(preview_path.resolve()) if preview_path is not None else None,
        "preview_masks_mp4": str(mask_preview_path.resolve()) if mask_preview_path is not None else None,
        "fish_tracks": fish_records,
        "mask_coverage": {
            "dynamic_object": float(dynamic.mean()) if dynamic.size else 0.0,
            "particle": float(particle.mean()) if particle.size else 0.0,
            "transient": float(transient.mean()) if transient.size else 0.0,
        },
        "claim_caveat": (
            "Clean variant has all-false transient labels. Injected variants use synthetic fish/snow "
            "overlays on real AQUALOC backgrounds and are external-background stress tests."
        ),
    }
    manifest_path = variant_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "variant": variant,
        "manifest": str(manifest_path.resolve()),
        "frames_csv": str(frames_csv.resolve()),
        "mask_coverage": manifest["mask_coverage"],
    }


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _select_frames(
    frame_paths: list[Path],
    *,
    start_index: int,
    stride: int,
    num_frames: int,
) -> list[Path]:
    selected = frame_paths[int(start_index) :: max(1, int(stride))]
    selected = selected[: int(num_frames)]
    if len(selected) < int(num_frames):
        raise RuntimeError(
            f"Only {len(selected)} frames selected from start={start_index}, stride={stride}; "
            f"need {num_frames}."
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        default="data/external_underwater/aqualoc/harbor_sequence_07/raw/raw_data/harbor_images_sequence_07",
    )
    parser.add_argument(
        "--trajectory",
        default="data/external_underwater/aqualoc/harbor_sequence_07/downloads/new_harbor_colmap_traj_sequence_07.txt",
    )
    parser.add_argument(
        "--calib-yaml",
        default="data/external_underwater/aqualoc/harbor_sequence_07/downloads/harbor_camera_calib.yaml",
    )
    parser.add_argument("--fish-manifest", default="data/watermask_uiis/fish_subset_train_0120.json")
    parser.add_argument("--output-root", default="data/external_underwater/aqualoc/harbor_sequence_07/aqua_sanity")
    parser.add_argument("--variants", default="clean,mixed-fish-snow,snow-high")
    parser.add_argument("--start-indices", default="0,96,192")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output-width", type=int, default=256)
    parser.add_argument("--output-height", type=int, default=256)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--no-previews", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"AQUALOC image dir not found: {image_dir}")
    frame_paths = [path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    frame_paths.sort(key=_frame_sort_key)
    if not frame_paths:
        raise RuntimeError(f"No frames found in {image_dir}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    poses = _read_trajectory(Path(args.trajectory))
    calib = _read_calib_yaml(Path(args.calib_yaml))
    fish_entries = _load_fish_manifest(Path(args.fish_manifest))
    start_indices = _parse_int_list(str(args.start_indices))
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]

    all_records: list[dict[str, Any]] = []
    for start in start_indices:
        selected = _select_frames(
            frame_paths,
            start_index=int(start),
            stride=int(args.stride),
            num_frames=int(args.num_frames),
        )
        selected_frame_ids = [_frame_id(path) for path in selected]
        frames = _copy_resize_frames(
            selected,
            output_width=int(args.output_width),
            output_height=int(args.output_height),
        )
        clip_name = f"aqualoc_harbor07_start{int(start):04d}_stride{int(args.stride)}"
        for variant_idx, variant in enumerate(variants):
            print(f"Building AQUALOC sanity clip: {clip_name}/{variant}", flush=True)
            record = _write_variant(
                variant=variant,
                output_root=output_root,
                clip_name=clip_name,
                selected_paths=selected,
                selected_frame_ids=selected_frame_ids,
                frames=frames,
                poses=poses,
                calib=calib,
                fish_entries=fish_entries,
                seed=int(args.seed) + int(start) * 1009 + int(variant_idx) * 9176,
                fps=float(args.fps),
                write_previews=not bool(args.no_previews),
            )
            record["start_index"] = int(start)
            record["stride"] = int(args.stride)
            all_records.append(record)

    manifest_list = output_root / "manifests.txt"
    manifest_list.write_text("\n".join(record["manifest"] for record in all_records) + "\n", encoding="utf-8")
    summary = {
        "name": output_root.name,
        "dataset": "AQUALOC harbor sequence 07 external sanity",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_dir": str(image_dir.resolve()),
        "trajectory": str(Path(args.trajectory).resolve()),
        "calib_yaml": str(Path(args.calib_yaml).resolve()),
        "fish_manifest": str(Path(args.fish_manifest).resolve()),
        "num_source_frames": len(frame_paths),
        "num_frames_per_clip": int(args.num_frames),
        "start_indices": [int(v) for v in start_indices],
        "stride": int(args.stride),
        "variants": all_records,
        "manifest_list": str(manifest_list.resolve()),
        "caveat": (
            "AQUALOC does not provide natural dynamic-object labels here. Clean clips are all-static sanity checks; "
            "injected variants are real-background synthetic-transient stress tests."
        ),
    }
    (output_root / "benchmark_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"AQUALOC sanity benchmark: {output_root}")
    for record in all_records:
        cov = record["mask_coverage"]
        print(
            f"- {Path(record['manifest']).parent.name}: transient={cov['transient']:.4f} "
            f"dynamic={cov['dynamic_object']:.4f} particle={cov['particle']:.4f}"
        )
    print(f"Manifest list: {manifest_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
