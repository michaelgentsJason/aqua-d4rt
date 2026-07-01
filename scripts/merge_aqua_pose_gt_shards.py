#!/usr/bin/env python3
"""Merge multiple eval_aqua_pose_gt_validation shard outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_aqua_pose_gt_validation import (  # noqa: E402
    _aggregate,
    _aggregate_by_stress_variant,
    _aggregate_seed_stability,
    _write_seed_stability_csv,
    _write_summary_by_stress_csv,
    _write_summary_csv,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def _load_per_clip(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected per_clip list in {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", default="tmp/aqua_tank_pose_stress_v2_stress4_multiseed_t090_shards")
    parser.add_argument("--output-dir", default="tmp/aqua_tank_pose_stress_v2_stress4_multiseed_t090_merged")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shard_root = Path(args.shard_root)
    output_dir = Path(args.output_dir)
    per_clip: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []

    for eval_dir in sorted(shard_root.glob("*/eval")):
        per_clip_path = eval_dir / "per_clip_metrics.json"
        aggregate_path = eval_dir / "aggregate_metrics.json"
        if not per_clip_path.exists():
            if bool(args.require_complete):
                raise RuntimeError(f"Missing shard per_clip_metrics.json: {per_clip_path}")
            continue
        shard_records = _load_per_clip(per_clip_path)
        per_clip.extend(shard_records)
        shards.append(
            {
                "shard": str(eval_dir.parent.name),
                "eval_dir": str(eval_dir.resolve()),
                "num_clips": len(shard_records),
                "aggregate_metrics": str(aggregate_path.resolve()) if aggregate_path.exists() else None,
            }
        )

    if not per_clip:
        raise RuntimeError(f"No completed shards found under {shard_root}")

    aggregate = _aggregate(per_clip)
    aggregate_by_stress = _aggregate_by_stress_variant(per_clip)
    aggregate_seed_stability = _aggregate_seed_stability(per_clip)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_clip_metrics.json").write_text(json.dumps(per_clip, indent=2, default=_json_default), encoding="utf-8")
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "shard_root": str(shard_root.resolve()),
                    "num_shards": len(shards),
                    "shards": shards,
                    "num_clips": len(per_clip),
                },
                "aggregate": aggregate,
                "aggregate_by_stress_variant": aggregate_by_stress,
                "aggregate_seed_stability": aggregate_seed_stability,
            },
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    _write_summary_csv(output_dir / "summary_table.csv", aggregate)
    _write_summary_by_stress_csv(output_dir / "summary_by_stress_variant.csv", aggregate_by_stress)
    _write_seed_stability_csv(output_dir / "summary_seed_stability.csv", aggregate_seed_stability)
    print(f"Merged {len(per_clip)} clips from {len(shards)} shards into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
