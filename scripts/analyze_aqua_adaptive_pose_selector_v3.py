#!/usr/bin/env python3
"""Analyze deployable v3 pose-soft selectors with raw reconstruction fallback."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


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
    if name == "num_points3D":
        return float(record.get("best_reconstruction", {}).get("num_points3D", 0.0) or 0.0)
    if name in {"ate_rmse", "rpe_trans_rmse"}:
        value = record.get("pose_metrics", {}).get(name)
        return float(value) if value is not None else float("nan")
    if name in {"feature_contamination", "match_contamination_mean", "essential_success_rate"}:
        value = record.get("frontend_metrics", {}).get(name)
        return float(value) if value is not None else float("nan")
    raise KeyError(name)


def _find_record(clip: dict[str, Any], base_variant: str, seed: int) -> dict[str, Any] | None:
    variants = clip.get("variants", {})
    seeded = variants.get(f"{base_variant}__seed{int(seed)}")
    if seeded is not None:
        return seeded
    # Single-seed evaluator runs keep unseeded variant names.
    return variants.get(base_variant)


def _threshold_tag(threshold: float) -> str:
    return f"t{float(threshold):.2f}".replace(".", "p")


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


def _record_row(
    *,
    policy: str,
    clip: dict[str, Any],
    seed: int,
    chosen_variant: str,
    reason: str,
    record: dict[str, Any],
    raw_record: dict[str, Any] | None,
    soft_record: dict[str, Any] | None,
) -> dict[str, Any]:
    raw_reg = _metric(raw_record, "input_registration_rate") if raw_record is not None else float("nan")
    soft_reg = _metric(soft_record, "input_registration_rate") if soft_record is not None else float("nan")
    return {
        "policy": policy,
        "stress_variant": str(clip.get("variant", "unknown")),
        "clip_name": str(clip.get("clip_name", "")),
        "seed": int(seed),
        "chosen_variant": chosen_variant,
        "reason": reason,
        "success": _metric(record, "success"),
        "pose_eval_success": _metric(record, "pose_eval_success"),
        "input_registration_rate": _metric(record, "input_registration_rate"),
        "num_points3D": _metric(record, "num_points3D"),
        "ate_rmse": _metric(record, "ate_rmse"),
        "rpe_trans_rmse": _metric(record, "rpe_trans_rmse"),
        "feature_contamination": _metric(record, "feature_contamination"),
        "match_contamination_mean": _metric(record, "match_contamination_mean"),
        "raw_input_registration_rate": raw_reg,
        "soft_input_registration_rate": soft_reg,
    }


def _make_soft_raw_fallback_policy(
    *,
    raw_min_registration: float,
    raw_margin: float,
    raw_name: str,
    soft_name: str,
) -> Callable[[dict[str, dict[str, Any]]], tuple[str, str]]:
    def choose(records: dict[str, dict[str, Any]]) -> tuple[str, str]:
        raw = records[raw_name]
        soft = records[soft_name]
        raw_reg = _metric(raw, "input_registration_rate")
        soft_reg = _metric(soft, "input_registration_rate")
        if raw_reg >= raw_min_registration and soft_reg + raw_margin < raw_reg:
            return raw_name, "raw_high_registration_fallback"
        return soft_name, "default_pose_soft"

    return choose


def _float_list(value: str) -> list[float]:
    return [float(item) for item in str(value).split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-clip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.73)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--raw-min-registration", type=float, default=0.60)
    parser.add_argument("--raw-margin", type=float, default=0.10)
    parser.add_argument(
        "--raw-min-registration-grid",
        default="",
        help="Optional comma-separated grid; overrides --raw-min-registration for fallback policies.",
    )
    parser.add_argument(
        "--raw-margin-grid",
        default="",
        help="Optional comma-separated grid; overrides --raw-margin for fallback policies.",
    )
    args = parser.parse_args()

    per_clip = json.loads(Path(args.per_clip).read_text(encoding="utf-8"))
    seeds = [int(item) for item in str(args.seeds).split(",") if item.strip()]
    tag = _threshold_tag(float(args.threshold))
    raw_name = "raw"
    aqua_name = "aqua_inpaint"
    hard_name = f"aqua_learned_retain_{tag}_inpaint"
    soft_name = f"aqua_pose_soft_{tag}"

    policies: dict[str, Callable[[dict[str, dict[str, Any]]], tuple[str, str]]] = {
        "raw": lambda records: (raw_name, "baseline_raw"),
        "aqua_inpaint": lambda records: (aqua_name, "baseline_aqua_inpaint"),
        "v3_hard": lambda records: (hard_name, "baseline_v3_hard"),
        "v3_pose_soft": lambda records: (soft_name, "baseline_v3_pose_soft"),
    }
    raw_min_grid = _float_list(args.raw_min_registration_grid) if str(args.raw_min_registration_grid).strip() else [float(args.raw_min_registration)]
    raw_margin_grid = _float_list(args.raw_margin_grid) if str(args.raw_margin_grid).strip() else [float(args.raw_margin)]
    for raw_min_registration in raw_min_grid:
        for raw_margin in raw_margin_grid:
            policies[
                f"soft_raw_fallback_margin{raw_margin:.2f}_rawmin{raw_min_registration:.2f}"
            ] = _make_soft_raw_fallback_policy(
                raw_min_registration=float(raw_min_registration),
                raw_margin=float(raw_margin),
                raw_name=raw_name,
                soft_name=soft_name,
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    detail_rows: list[dict[str, Any]] = []

    for policy, chooser in policies.items():
        selected_rows: list[dict[str, Any]] = []
        for clip in per_clip:
            for seed in seeds:
                records = {
                    raw_name: _find_record(clip, raw_name, seed),
                    aqua_name: _find_record(clip, aqua_name, seed),
                    hard_name: _find_record(clip, hard_name, seed),
                    soft_name: _find_record(clip, soft_name, seed),
                }
                if any(record is None for record in records.values()):
                    continue
                typed_records = {name: record for name, record in records.items() if record is not None}
                chosen_variant, reason = chooser(typed_records)
                record = typed_records[chosen_variant]
                row = _record_row(
                    policy=policy,
                    clip=clip,
                    seed=seed,
                    chosen_variant=chosen_variant,
                    reason=reason,
                    record=record,
                    raw_record=typed_records[raw_name],
                    soft_record=typed_records[soft_name],
                )
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
