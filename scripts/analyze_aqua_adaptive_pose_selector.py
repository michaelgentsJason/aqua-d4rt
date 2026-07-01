#!/usr/bin/env python3
"""Analyze post-hoc adaptive selection on Aqua Tank GT-pose eval outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


VARIANTS = (
    "raw",
    "aqua_inpaint",
    "aqua_learned_retain_t0p90_inpaint",
    "aqua_pose_soft_t0p90",
)


def _mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return sum(values) / len(values) if values else float("nan")


def _variant_for_seed(clip: dict[str, Any], base_variant: str, seed: int) -> dict[str, Any] | None:
    return clip.get("variants", {}).get(f"{base_variant}__seed{seed}")


def _metric(record: dict[str, Any], name: str) -> float:
    if name == "success":
        return float(bool(record.get("success", False)))
    if name == "pose_eval_success":
        return float(bool(record.get("pose_metrics", {}).get("pose_eval_success", False)))
    if name == "input_registration_rate":
        return float(record.get("input_registration_rate", 0.0) or 0.0)
    if name in {"ate_rmse", "rpe_trans_rmse"}:
        value = record.get("pose_metrics", {}).get(name)
        return float(value) if value is not None else float("nan")
    if name == "num_points3D":
        return float(record.get("best_reconstruction", {}).get("num_points3D", 0.0) or 0.0)
    if name in {"feature_contamination", "match_contamination_mean", "essential_success_rate"}:
        value = record.get("frontend_metrics", {}).get(name)
        return float(value) if value is not None else float("nan")
    raise KeyError(name)


def _score(record: dict[str, Any], *, prefer_pose: bool = False) -> float:
    success = _metric(record, "success")
    reg = _metric(record, "input_registration_rate")
    points = _metric(record, "num_points3D")
    reproj = float(record.get("best_reconstruction", {}).get("mean_reprojection_error", 10.0) or 10.0)
    track = float(record.get("best_reconstruction", {}).get("mean_track_length", 0.0) or 0.0)
    essential = _metric(record, "essential_success_rate")
    score = 2.0 * success + 3.0 * reg + 0.15 * math.log1p(max(points, 0.0))
    score += 0.10 * min(track, 10.0) + 0.5 * essential - 0.05 * reproj
    if prefer_pose and _metric(record, "pose_eval_success") > 0:
        ate = _metric(record, "ate_rmse")
        rpe = _metric(record, "rpe_trans_rmse")
        if math.isfinite(ate):
            score -= 2.0 * ate
        if math.isfinite(rpe):
            score -= 2.0 * rpe
    return score


def _choose_records(clip: dict[str, Any], seed: int, policy: str) -> tuple[str, dict[str, Any]]:
    records = {
        variant: _variant_for_seed(clip, variant, seed)
        for variant in VARIANTS
    }
    records = {k: v for k, v in records.items() if v is not None}
    if not records:
        raise RuntimeError(f"No records for {clip.get('manifest')} seed {seed}")

    if policy in records:
        return policy, records[policy]

    if policy == "oracle_min_ate":
        valid = {
            k: v for k, v in records.items()
            if _metric(v, "pose_eval_success") > 0.0 and math.isfinite(_metric(v, "ate_rmse"))
        }
        if valid:
            return min(valid.items(), key=lambda kv: (_metric(kv[1], "ate_rmse"), -_metric(kv[1], "input_registration_rate")))
        return max(records.items(), key=lambda kv: _metric(kv[1], "input_registration_rate"))

    if policy == "oracle_min_rpe":
        valid = {
            k: v for k, v in records.items()
            if _metric(v, "pose_eval_success") > 0.0 and math.isfinite(_metric(v, "rpe_trans_rmse"))
        }
        if valid:
            return min(valid.items(), key=lambda kv: (_metric(kv[1], "rpe_trans_rmse"), -_metric(kv[1], "input_registration_rate")))
        return max(records.items(), key=lambda kv: _metric(kv[1], "input_registration_rate"))

    if policy == "reconstruction_quality":
        candidates = {k: v for k, v in records.items() if k != "raw"}
        return max(candidates.items(), key=lambda kv: _score(kv[1], prefer_pose=False))

    if policy == "reconstruction_quality_with_raw_fallback":
        best_nonraw_name, best_nonraw = max(
            ((k, v) for k, v in records.items() if k != "raw"),
            key=lambda kv: _score(kv[1], prefer_pose=False),
        )
        raw = records["raw"]
        if _metric(best_nonraw, "success") < 1.0 and _metric(raw, "success") >= 1.0:
            return "raw", raw
        if _metric(best_nonraw, "input_registration_rate") + 0.20 < _metric(raw, "input_registration_rate"):
            return "raw", raw
        return best_nonraw_name, best_nonraw

    raise ValueError(f"Unknown policy: {policy}")


def _summarize(selected: list[tuple[str, str, int, dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {"records": len(selected)}
    for metric in (
        "success",
        "pose_eval_success",
        "input_registration_rate",
        "num_points3D",
        "ate_rmse",
        "rpe_trans_rmse",
        "feature_contamination",
        "match_contamination_mean",
    ):
        out[metric] = _mean([_metric(record, metric) for _, _, _, record in selected])
    counts: dict[str, int] = defaultdict(int)
    for name, _, _, _ in selected:
        counts[name] += 1
    out["selection_counts"] = dict(sorted(counts.items()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-clip",
        default="tmp/aqua_tank_pose_stress_v2_stress4_multiseed_t090_merged/per_clip_metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp/aqua_tank_pose_stress_v2_stress4_multiseed_t090_adaptive_analysis",
    )
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()

    per_clip = json.loads(Path(args.per_clip).read_text(encoding="utf-8"))
    seeds = [int(item) for item in str(args.seeds).split(",") if item.strip()]
    policies = list(VARIANTS) + [
        "reconstruction_quality",
        "reconstruction_quality_with_raw_fallback",
        "oracle_min_ate",
        "oracle_min_rpe",
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    detail_rows: list[dict[str, Any]] = []
    for policy in policies:
        selected: list[tuple[str, str, int, dict[str, Any]]] = []
        for clip in per_clip:
            stress = str(clip.get("variant", "unknown"))
            for seed in seeds:
                chosen_name, record = _choose_records(clip, seed, policy)
                selected.append((chosen_name, stress, seed, record))
                detail_rows.append(
                    {
                        "policy": policy,
                        "stress_variant": stress,
                        "seed": seed,
                        "chosen_variant": chosen_name,
                        "pose_eval_success": _metric(record, "pose_eval_success"),
                        "input_registration_rate": _metric(record, "input_registration_rate"),
                        "ate_rmse": _metric(record, "ate_rmse"),
                        "rpe_trans_rmse": _metric(record, "rpe_trans_rmse"),
                        "feature_contamination": _metric(record, "feature_contamination"),
                        "match_contamination": _metric(record, "match_contamination_mean"),
                    }
                )
        summaries[policy] = _summarize(selected)

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "records",
                "success",
                "pose_eval_success",
                "input_registration_rate",
                "num_points3D",
                "ate_rmse",
                "rpe_trans_rmse",
                "feature_contamination",
                "match_contamination_mean",
                "selection_counts",
            ],
        )
        writer.writeheader()
        for policy, summary in summaries.items():
            row = {"policy": policy, **summary}
            row["selection_counts"] = json.dumps(row["selection_counts"], sort_keys=True)
            writer.writerow(row)

    with (output_dir / "selection_details.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    print(f"Saved: {output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
