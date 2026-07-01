#!/usr/bin/env python3
"""Analyze deployable hard/pose-soft selection on Aqua pose-eval outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    valid = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(valid) / len(valid)) if valid else float("nan")


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


def _find_record(clip: dict[str, Any], base_variant: str, seed: int) -> dict[str, Any] | None:
    variants = clip.get("variants", {})
    return variants.get(f"{base_variant}__seed{int(seed)}") or variants.get(base_variant)


def _threshold_tag(threshold: float) -> str:
    return f"t{float(threshold):.2f}".replace(".", "p")


def _retention_stats(clip: dict[str, Any], threshold_tag: str) -> dict[str, float]:
    mask = clip.get("mask_coverage", {})
    meta = clip.get("retention_meta", {})
    soft_meta = meta.get(f"aqua_pose_soft_{threshold_tag}", {})
    learned_meta = meta.get(f"aqua_learned_retain_{threshold_tag}_inpaint", {})
    aqua_meta = clip.get("aqua_meta", {})
    window_meta = aqua_meta.get("window_meta", []) if isinstance(aqua_meta, dict) else []
    dynamic_prob_mean = float(aqua_meta.get("dynamic_prob_mean", 0.0) or 0.0)
    particle_prob_mean = float(aqua_meta.get("particle_prob_mean", 0.0) or 0.0)
    if window_meta:
        dynamic_prob_mean = _mean([float(item.get("dynamic_prob_mean", 0.0) or 0.0) for item in window_meta])
        particle_prob_mean = _mean([float(item.get("particle_prob_mean", 0.0) or 0.0) for item in window_meta])
    return {
        "dynamic_gt_coverage": float(mask.get("dynamic_object", 0.0) or 0.0),
        "particle_gt_coverage": float(mask.get("particle", 0.0) or 0.0),
        "transient_gt_coverage": float(mask.get("transient", 0.0) or 0.0),
        "aqua_rejected": float(mask.get("aqua_rejected", 0.0) or 0.0),
        "learned_retained": float(mask.get("learned_retained", 0.0) or 0.0),
        "pose_soft_retained": float(mask.get("pose_soft_retained", 0.0) or 0.0),
        "pose_soft_mean_weight": float(mask.get("pose_soft_mean_weight", 0.0) or soft_meta.get("mean_retention_weight", 0.0) or 0.0),
        "mean_selected_geometry_score": float(soft_meta.get("mean_selected_geometry_score", 0.0) or 0.0),
        "candidate_positive_rate": float(learned_meta.get("positive_rate", soft_meta.get("positive_rate", 0.0)) or 0.0),
        "candidate_gt_transient_rate": float(learned_meta.get("gt_transient_rate", soft_meta.get("gt_transient_rate", 0.0)) or 0.0),
        "candidate_stable_rate": float(learned_meta.get("stable_rate", soft_meta.get("stable_rate", 0.0)) or 0.0),
        "aqua_dynamic_prob_mean": dynamic_prob_mean,
        "aqua_particle_prob_mean": particle_prob_mean,
    }


def _choose_policy(stats: dict[str, float], policy: str) -> tuple[str, str]:
    if policy == "always_hard":
        return "hard", "always_hard"
    if policy == "always_soft":
        return "pose_soft", "always_soft"
    if policy == "dynamic_guard":
        if stats["dynamic_gt_coverage"] >= 0.25:
            return "hard", "dynamic_coverage_guard"
        return "pose_soft", "default_pose_soft"
    if policy == "aqua_dynamic_guard":
        if stats["aqua_dynamic_prob_mean"] >= 0.18:
            return "hard", "aqua_dynamic_prob_guard"
        return "pose_soft", "default_pose_soft"
    if policy == "aqua_dynamic_high_guard":
        if stats["aqua_dynamic_prob_mean"] >= 0.30:
            return "hard", "aqua_dynamic_high_prob_guard"
        return "pose_soft", "default_pose_soft"
    if policy == "conservative_soft":
        if stats["candidate_gt_transient_rate"] >= 0.55:
            return "hard", "candidate_transient_rate_guard"
        if stats["aqua_rejected"] >= 0.50 and stats["candidate_positive_rate"] < 0.09:
            return "hard", "high_rejection_low_positive_guard"
        if stats["pose_soft_mean_weight"] >= 0.60 and stats["candidate_positive_rate"] >= 0.08:
            return "pose_soft", "strong_soft_support"
        return "hard", "fallback_hard"
    raise ValueError(f"Unknown policy: {policy}")


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"records": len(rows)}
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
        out[metric] = _mean([float(row[metric]) for row in rows])
    counts: dict[str, int] = defaultdict(int)
    reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["chosen_variant"])] += 1
        reasons[str(row["reason"])] += 1
    out["selection_counts"] = dict(sorted(counts.items()))
    out["reason_counts"] = dict(sorted(reasons.items()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-clip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.73)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument(
        "--policies",
        default="always_hard,always_soft,aqua_dynamic_high_guard,dynamic_guard,aqua_dynamic_guard,conservative_soft",
    )
    args = parser.parse_args()

    per_clip = json.loads(Path(args.per_clip).read_text(encoding="utf-8"))
    seeds = [int(item) for item in str(args.seeds).split(",") if item.strip()]
    policies = [item.strip() for item in str(args.policies).split(",") if item.strip()]
    tag = _threshold_tag(float(args.threshold))
    hard_name = f"aqua_learned_retain_{tag}_inpaint"
    soft_name = f"aqua_pose_soft_{tag}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    detail_rows: list[dict[str, Any]] = []

    for policy in policies:
        selected_rows: list[dict[str, Any]] = []
        for clip in per_clip:
            stats = _retention_stats(clip, tag)
            for seed in seeds:
                chosen_kind, reason = _choose_policy(stats, policy)
                base = soft_name if chosen_kind == "pose_soft" else hard_name
                record = _find_record(clip, base, seed)
                if record is None:
                    continue
                row = {
                    "policy": policy,
                    "stress_variant": str(clip.get("variant", "unknown")),
                    "clip_name": str(clip.get("clip_name", "")),
                    "seed": int(seed),
                    "chosen_variant": base,
                    "reason": reason,
                    "success": _metric(record, "success"),
                    "pose_eval_success": _metric(record, "pose_eval_success"),
                    "input_registration_rate": _metric(record, "input_registration_rate"),
                    "num_points3D": _metric(record, "num_points3D"),
                    "ate_rmse": _metric(record, "ate_rmse"),
                    "rpe_trans_rmse": _metric(record, "rpe_trans_rmse"),
                    "feature_contamination": _metric(record, "feature_contamination"),
                    "match_contamination_mean": _metric(record, "match_contamination_mean"),
                    **stats,
                }
                selected_rows.append(row)
                detail_rows.append(row)
        summaries[policy] = _summarize(selected_rows)

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
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
            "reason_counts",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for policy, summary in summaries.items():
            row = {"policy": policy, **summary}
            row["selection_counts"] = json.dumps(row["selection_counts"], sort_keys=True)
            row["reason_counts"] = json.dumps(row["reason_counts"], sort_keys=True)
            writer.writerow(row)

    with (output_dir / "selection_details.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()) if detail_rows else ["policy"])
        writer.writeheader()
        writer.writerows(detail_rows)

    print(f"Saved: {output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
