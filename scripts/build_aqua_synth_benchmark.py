#!/usr/bin/env python3
"""Build a multi-clip Aqua-D4RT synthetic benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _manifest_frames(background: Path) -> list[str]:
    manifest_path = background / "manifest.json" if background.is_dir() else background
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [str(item) for item in payload.get("frames", [])]


def _split_counts(num_clips: int, train: int, val: int, test: int) -> dict[str, int]:
    explicit = int(train) + int(val) + int(test)
    if explicit > 0:
        if explicit != int(num_clips):
            raise ValueError(f"Explicit train/val/test counts sum to {explicit}, expected {num_clips}")
        return {"train": int(train), "val": int(val), "test": int(test)}
    n_train = int(round(0.70 * int(num_clips)))
    n_val = int(round(0.15 * int(num_clips)))
    n_test = int(num_clips) - n_train - n_val
    return {"train": n_train, "val": n_val, "test": n_test}


def _write_list(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", default="data/aqua_smoke/underwater_caves_sonar_32/manifest.json")
    parser.add_argument("--fish-manifest", default="data/watermask_uiis/fish_subset_train_0120.json")
    parser.add_argument("--output-root", default="data/aqua_synth_benchmark/watermask_caves_100")
    parser.add_argument("--num-clips", type=int, default=100)
    parser.add_argument("--train-count", type=int, default=0)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    splits_dir = output_root / "splits"
    output_root.mkdir(parents=True, exist_ok=True)
    counts = _split_counts(
        num_clips=int(args.num_clips),
        train=int(args.train_count),
        val=int(args.val_count),
        test=int(args.test_count),
    )
    split_names: list[str] = []
    for split, count in counts.items():
        split_names.extend([split] * int(count))
    if len(split_names) != int(args.num_clips):
        raise RuntimeError("Invalid split construction")

    rng = np.random.default_rng(int(args.seed))
    clip_records: list[dict[str, Any]] = []
    split_manifests: dict[str, list[str]] = {key: [] for key in counts}
    bg_frames = _manifest_frames(Path(args.background))
    if bg_frames and len(bg_frames) < int(args.num_frames):
        print(
            f"Warning: background manifest has {len(bg_frames)} frames, fewer than requested {args.num_frames}; "
            "builder will use all available frames.",
            flush=True,
        )

    builder = REPO_ROOT / "scripts" / "build_aqua_synthetic_transients.py"
    for clip_idx, split in enumerate(split_names):
        clip_seed = int(rng.integers(1, 2**31 - 1))
        fish_tracks = int(rng.integers(2, 8))
        particles_min = int(rng.integers(35, 130))
        particles_max = int(rng.integers(max(particles_min + 20, 80), 260))
        particle_radius_min = float(rng.uniform(0.45, 1.1))
        particle_radius_max = float(rng.uniform(max(1.5, particle_radius_min + 0.7), 3.4))
        fish_scale_min = float(rng.uniform(0.10, 0.22))
        fish_scale_max = float(rng.uniform(max(0.24, fish_scale_min + 0.08), 0.46))
        underwater_strength = float(rng.uniform(0.45, 0.95))
        clip_dir = output_root / split / f"clip_{clip_idx:04d}"
        manifest_path = clip_dir / "manifest.json"
        cmd = [
            str(args.python_bin),
            str(builder),
            "--background",
            str(args.background),
            "--fish-manifest",
            str(args.fish_manifest),
            "--output-dir",
            str(clip_dir),
            "--num-frames",
            str(int(args.num_frames)),
            "--fish-tracks",
            str(fish_tracks),
            "--fish-scale-min",
            f"{fish_scale_min:.4f}",
            "--fish-scale-max",
            f"{fish_scale_max:.4f}",
            "--particles-min",
            str(particles_min),
            "--particles-max",
            str(particles_max),
            "--particle-radius-min",
            f"{particle_radius_min:.4f}",
            "--particle-radius-max",
            f"{particle_radius_max:.4f}",
            "--underwater-strength",
            f"{underwater_strength:.4f}",
            "--seed",
            str(clip_seed),
        ]
        record = {
            "clip_index": int(clip_idx),
            "split": split,
            "seed": clip_seed,
            "manifest": str(manifest_path.resolve()),
            "output_dir": str(clip_dir.resolve()),
            "params": {
                "fish_tracks": fish_tracks,
                "fish_scale_min": fish_scale_min,
                "fish_scale_max": fish_scale_max,
                "particles_min": particles_min,
                "particles_max": particles_max,
                "particle_radius_min": particle_radius_min,
                "particle_radius_max": particle_radius_max,
                "underwater_strength": underwater_strength,
            },
        }
        clip_records.append(record)
        split_manifests[split].append(str(manifest_path.resolve()))
        if args.dry_run:
            print("DRY", " ".join(cmd))
            continue
        if manifest_path.exists() and not bool(args.overwrite):
            print(f"[{clip_idx + 1}/{args.num_clips}] exists: {manifest_path}", flush=True)
            continue
        print(f"[{clip_idx + 1}/{args.num_clips}] building {split}/{clip_dir.name}", flush=True)
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    for split, items in split_manifests.items():
        _write_list(splits_dir / f"{split}_manifests.txt", items)

    summary = {
        "name": output_root.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root.resolve()),
        "background": str(Path(args.background).resolve()),
        "fish_manifest": str(Path(args.fish_manifest).resolve()),
        "num_clips": int(args.num_clips),
        "num_frames": int(args.num_frames),
        "seed": int(args.seed),
        "split_counts": counts,
        "split_files": {
            split: str((splits_dir / f"{split}_manifests.txt").resolve()) for split in split_manifests
        },
        "clips": clip_records,
    }
    (output_root / "benchmark_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Benchmark manifest: {output_root / 'benchmark_manifest.json'}")
    for split, items in split_manifests.items():
        print(f"{split}: {len(items)} manifests -> {splits_dir / f'{split}_manifests.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
