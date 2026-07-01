#!/usr/bin/env python3
"""Analyze clip-level adaptive Aqua static-threshold selectors.

The script reuses existing static-map and ORB/SfM proxy per-clip outputs. It
does not run the Aqua model. The goal is to test whether a deployable
clip-level selector can improve the contamination/retention/front-end Pareto
over a single fixed static-confidence threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


THRESHOLD_RE = re.compile(r"aqua_static_conf_ge_([0-9]+)p([0-9]+)")


@dataclass(frozen=True)
class SelectorPolicy:
    name: str
    low_threshold: float = 0.15
    mid_threshold: float = 0.35
    high_threshold: float = 0.55
    min_success: float = 0.75
    min_features_per_frame: float = 120.0
    min_matches_per_pair: float = 50.0
    min_inlier_rate: float = 0.55
    min_static_mask_fraction: float = 0.10


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _threshold_from_variant(name: str) -> float | None:
    match = THRESHOLD_RE.search(name)
    if not match:
        return None
    return float(f"{int(match.group(1))}.{match.group(2)}")


def _variant_for_threshold(threshold: float) -> str:
    return f"aqua_static_conf_ge_{threshold:.3f}".replace(".", "p").rstrip("0").rstrip("p")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _summary(orb_variant: dict[str, Any]) -> dict[str, float]:
    data = orb_variant.get("summary", orb_variant)
    frame_metrics = orb_variant.get("frame_metrics", [])
    static_mask_values = [
        _safe_float(item.get("static_mask_fraction"))
        for item in frame_metrics
        if math.isfinite(_safe_float(item.get("static_mask_fraction")))
    ]
    return {
        "total_features": _safe_float(data.get("total_features"), 0.0),
        "features_per_frame": _safe_float(data.get("features_per_frame_mean"), 0.0),
        "feature_contamination": _safe_float(data.get("feature_contamination")),
        "total_matches": _safe_float(data.get("total_matches"), 0.0),
        "pairs": _safe_float(data.get("pairs"), 0.0),
        "matches_per_pair": _safe_float(data.get("matches_per_pair_mean"), 0.0),
        "match_contamination": _safe_float(data.get("match_contamination_mean")),
        "essential_success": _safe_float(data.get("essential_success_rate")),
        "essential_inliers_per_pair": _safe_float(data.get("essential_inliers_per_pair_mean"), 0.0),
        "essential_inlier_rate": _safe_float(data.get("essential_inlier_rate_mean")),
        "static_mask_fraction": sum(static_mask_values) / float(len(static_mask_values)) if static_mask_values else float("nan"),
    }


def _static_metrics(static_variant: dict[str, Any]) -> dict[str, float]:
    return {
        "kept_points": _safe_float(static_variant.get("kept_points"), 0.0),
        "kept_static_points": _safe_float(static_variant.get("kept_static_points"), 0.0),
        "kept_transient_points": _safe_float(static_variant.get("kept_transient_points"), 0.0),
        "query_contamination": _safe_float(static_variant.get("point_contamination")),
        "static_retention": _safe_float(static_variant.get("point_static_retention")),
        "transient_rejection": _safe_float(static_variant.get("point_transient_rejection")),
        "voxel_contamination": _safe_float(static_variant.get("voxel_contamination_any")),
        "voxel_retention": _safe_float(static_variant.get("voxel_static_support_retention")),
    }


def _clip_key(record: dict[str, Any]) -> str:
    return str(record.get("clip_name") or record.get("manifest") or "")


def _threshold_variants(variants: dict[str, Any]) -> dict[float, str]:
    out: dict[float, str] = {}
    for name in variants:
        threshold = _threshold_from_variant(name)
        if threshold is not None:
            out[round(threshold, 6)] = name
    return out


def _threshold_rows_for_clip(static_record: dict[str, Any], orb_record: dict[str, Any]) -> list[dict[str, Any]]:
    static_variants = static_record.get("variant_metrics", {})
    orb_variants = orb_record.get("variants", {})
    static_by_threshold = _threshold_variants(static_variants)
    orb_by_threshold = _threshold_variants(orb_variants)
    rows: list[dict[str, Any]] = []
    for threshold in sorted(set(static_by_threshold) & set(orb_by_threshold)):
        static_name = static_by_threshold[threshold]
        orb_name = orb_by_threshold[threshold]
        row: dict[str, Any] = {
            "threshold": threshold,
            "variant": static_name,
        }
        row.update(_static_metrics(static_variants[static_name]))
        row.update(_summary(orb_variants[orb_name]))
        rows.append(row)
    return rows


def _nearest_threshold(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    if not rows:
        raise ValueError("No threshold rows available.")
    return min(rows, key=lambda row: abs(float(row["threshold"]) - float(threshold)))


def _raw_row(static_record: dict[str, Any], orb_record: dict[str, Any]) -> dict[str, Any]:
    static_variants = static_record.get("variant_metrics", {})
    orb_variants = orb_record.get("variants", {})
    row: dict[str, Any] = {"threshold": None, "variant": "raw_all_pixels"}
    row.update(_static_metrics(static_variants.get("all_d4rt_points", {})))
    row.update(_summary(orb_variants.get("raw_all_pixels", {})))
    return row


def _oracle_select(
    rows: list[dict[str, Any]],
    *,
    min_essential_success: float,
    min_static_retention: float,
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if _safe_float(row.get("essential_success")) >= min_essential_success
        and _safe_float(row.get("static_retention")) >= min_static_retention
    ]
    if not candidates:
        candidates = [row for row in rows if _safe_float(row.get("essential_success")) >= min_essential_success]
    if not candidates:
        candidates = rows
    return min(
        candidates,
        key=lambda row: (
            _safe_float(row.get("feature_contamination"), 1.0),
            _safe_float(row.get("match_contamination"), 1.0),
            -_safe_float(row.get("essential_success"), 0.0),
            -_safe_float(row.get("static_retention"), 0.0),
        ),
    )


def _deployable_rule_select(rows: list[dict[str, Any]], policy: SelectorPolicy) -> tuple[dict[str, Any], str]:
    """Select the highest clean threshold that passes ORB diagnostics.

    This selector intentionally does not use GT contamination or retention.
    It is multi-pass because ORB diagnostics must be computed for candidate
    thresholds, but it is deployable in the same spirit as the R099
    self-diagnostic selector.
    """

    candidates = [
        row
        for row in rows
        if float(policy.low_threshold) <= _safe_float(row.get("threshold"), -1.0) <= float(policy.high_threshold)
    ]
    if not candidates:
        candidates = rows

    passing = [
        row
        for row in candidates
        if _safe_float(row.get("essential_success"), 0.0) >= policy.min_success
        and _safe_float(row.get("features_per_frame"), 0.0) >= policy.min_features_per_frame
        and _safe_float(row.get("matches_per_pair"), 0.0) >= policy.min_matches_per_pair
        and _safe_float(row.get("essential_inlier_rate"), 0.0) >= policy.min_inlier_rate
        and _safe_float(row.get("static_mask_fraction"), 1.0) >= policy.min_static_mask_fraction
    ]
    if passing:
        selected = max(
            passing,
            key=lambda row: (
                _safe_float(row.get("threshold"), 0.0),
                _safe_float(row.get("essential_inlier_rate"), 0.0),
            ),
        )
        return selected, "highest_threshold_passing_orb_diagnostics"

    # If no threshold passes, choose the lowest candidate to protect feature
    # survival/registration rather than pretending a clean map is deployable.
    selected = min(
        candidates,
        key=lambda row: (
            _safe_float(row.get("threshold"), 0.0),
            -_safe_float(row.get("essential_success"), 0.0),
        ),
    )
    return selected, "fallback_lowest_threshold_for_feature_survival"


def _score_aggregate(
    agg: dict[str, float],
    *,
    min_success: float,
    min_retention: float,
    success_penalty: float,
    retention_penalty: float,
    feature_weight: float,
    match_weight: float,
) -> float:
    feature_contam = _safe_float(agg.get("feature_contamination"), 1.0)
    match_contam = _safe_float(agg.get("match_contamination"), 1.0)
    success = _safe_float(agg.get("essential_success"), 0.0)
    retention = _safe_float(agg.get("static_retention"), 0.0)
    return (
        feature_weight * feature_contam
        + match_weight * match_contam
        + success_penalty * max(0.0, float(min_success) - success)
        + retention_penalty * max(0.0, float(min_retention) - retention)
    )


def _grid_search_policy(
    joined: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    low_threshold: float,
    mid_threshold: float,
    high_threshold: float,
    min_success: float,
    min_retention: float,
    success_penalty: float,
    retention_penalty: float,
    feature_weight: float,
    match_weight: float,
) -> tuple[SelectorPolicy, dict[str, float]]:
    best_policy: SelectorPolicy | None = None
    best_agg: dict[str, float] | None = None
    best_score = float("inf")
    success_values = [0.50, 0.65, 0.75, 0.80, 0.85]
    feature_values = [60.0, 100.0, 140.0, 180.0, 240.0]
    match_values = [15.0, 30.0, 50.0, 80.0, 120.0]
    inlier_values = [0.45, 0.55, 0.65, 0.75]
    mask_values = [0.03, 0.08, 0.12, 0.18, 0.25]
    for min_success in success_values:
        for min_features in feature_values:
            for min_matches in match_values:
                for min_inlier in inlier_values:
                    for min_mask in mask_values:
                        policy = SelectorPolicy(
                            name="orb_selector_tuned",
                            low_threshold=low_threshold,
                            mid_threshold=mid_threshold,
                            high_threshold=high_threshold,
                            min_success=min_success,
                            min_features_per_frame=min_features,
                            min_matches_per_pair=min_matches,
                            min_inlier_rate=min_inlier,
                            min_static_mask_fraction=min_mask,
                        )
                        rows: list[dict[str, Any]] = []
                        for static_record, orb_record in joined:
                            threshold_rows = _threshold_rows_for_clip(static_record, orb_record)
                            if not threshold_rows:
                                continue
                            selected, _ = _deployable_rule_select(threshold_rows, policy)
                            rows.append(_attach_totals(selected, static_record))
                        agg = _aggregate(rows)
                        score = _score_aggregate(
                            agg,
                            min_success=min_success,
                            min_retention=min_retention,
                            success_penalty=success_penalty,
                            retention_penalty=retention_penalty,
                            feature_weight=feature_weight,
                            match_weight=match_weight,
                        )
                        if score < best_score:
                            best_score = score
                            best_policy = policy
                            best_agg = agg
    if best_policy is None or best_agg is None:
        raise ValueError("Policy grid search failed.")
    return best_policy, best_agg


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    sums = {
        "kept_points": 0.0,
        "kept_static_points": 0.0,
        "kept_transient_points": 0.0,
        "total_features": 0.0,
        "contaminated_features": 0.0,
        "total_matches": 0.0,
        "total_pairs": 0.0,
        "success_pairs": 0.0,
        "match_contamination_sum": 0.0,
        "essential_success_sum": 0.0,
        "essential_inliers_sum": 0.0,
        "essential_inlier_rate_sum": 0.0,
        "features_per_frame_sum": 0.0,
        "matches_per_pair_sum": 0.0,
    }
    for row in rows:
        kept_points = _safe_float(row.get("kept_points"), 0.0)
        kept_static = _safe_float(row.get("kept_static_points"), 0.0)
        kept_transient = _safe_float(row.get("kept_transient_points"), 0.0)
        total_features = _safe_float(row.get("total_features"), 0.0)
        total_matches = _safe_float(row.get("total_matches"), 0.0)
        pairs = _safe_float(row.get("pairs"), 16.0)
        feature_contam = _safe_float(row.get("feature_contamination"), 0.0)
        match_contam = _safe_float(row.get("match_contamination"), 0.0)
        essential_success = _safe_float(row.get("essential_success"), 0.0)
        sums["kept_points"] += kept_points
        sums["kept_static_points"] += kept_static
        sums["kept_transient_points"] += kept_transient
        sums["total_features"] += total_features
        sums["contaminated_features"] += total_features * feature_contam
        sums["total_matches"] += total_matches
        sums["total_pairs"] += pairs
        sums["success_pairs"] += pairs * essential_success
        sums["match_contamination_sum"] += match_contam
        sums["essential_success_sum"] += essential_success
        sums["essential_inliers_sum"] += _safe_float(row.get("essential_inliers_per_pair"), 0.0)
        sums["essential_inlier_rate_sum"] += _safe_float(row.get("essential_inlier_rate"), 0.0)
        sums["features_per_frame_sum"] += _safe_float(row.get("features_per_frame"), 0.0)
        sums["matches_per_pair_sum"] += _safe_float(row.get("matches_per_pair"), 0.0)

    n = max(1, len(rows))
    kept_points = sums["kept_points"]
    kept_static = sums["kept_static_points"]
    kept_transient = sums["kept_transient_points"]
    total_static = kept_static
    for row in rows:
        total_static += max(0.0, _safe_float(row.get("total_static_points"), 0.0) - _safe_float(row.get("kept_static_points"), 0.0))
    # Some static evaluators do not copy total_static_points into selected rows.
    if total_static <= 0:
        total_static = kept_static
    return {
        "clips": float(len(rows)),
        "query_contamination": kept_transient / kept_points if kept_points > 0 else float("nan"),
        "static_retention": kept_static / total_static if total_static > 0 else float("nan"),
        "feature_contamination": sums["contaminated_features"] / sums["total_features"] if sums["total_features"] > 0 else float("nan"),
        "match_contamination": sums["match_contamination_sum"] / n,
        "essential_success": sums["success_pairs"] / sums["total_pairs"] if sums["total_pairs"] > 0 else sums["essential_success_sum"] / n,
        "features_per_frame": sums["features_per_frame_sum"] / n,
        "matches_per_pair": sums["matches_per_pair_sum"] / n,
        "essential_inliers_per_pair": sums["essential_inliers_sum"] / n,
        "essential_inlier_rate": sums["essential_inlier_rate_sum"] / n,
    }


def _attach_totals(row: dict[str, Any], static_record: dict[str, Any]) -> dict[str, Any]:
    raw_static = static_record.get("variant_metrics", {}).get("all_d4rt_points", {})
    out = dict(row)
    out["total_static_points"] = _safe_float(raw_static.get("total_static_points"), 0.0)
    out["total_transient_points"] = _safe_float(raw_static.get("total_transient_points"), 0.0)
    return out


def _load_joined(static_path: Path, orb_path: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    static_records = {_clip_key(record): record for record in _read_json(static_path)}
    orb_records = {_clip_key(record): record for record in _read_json(orb_path)}
    common = sorted(set(static_records) & set(orb_records))
    if not common:
        raise ValueError(f"No overlapping clip names between {static_path} and {orb_path}")
    return [(static_records[key], orb_records[key]) for key in common]


def _fmt_pct(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{100.0 * value:.4f}"


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "selector",
        "clips",
        "query_contamination_pct",
        "static_retention_pct",
        "feature_contamination_pct",
        "match_contamination_pct",
        "essential_success_pct",
        "features_per_frame",
        "matches_per_pair",
        "selection_counts",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "selector": row["selector"],
                    "clips": int(row["clips"]),
                    "query_contamination_pct": _fmt_pct(row["query_contamination"]),
                    "static_retention_pct": _fmt_pct(row["static_retention"]),
                    "feature_contamination_pct": _fmt_pct(row["feature_contamination"]),
                    "match_contamination_pct": _fmt_pct(row["match_contamination"]),
                    "essential_success_pct": _fmt_pct(row["essential_success"]),
                    "features_per_frame": f"{row['features_per_frame']:.4f}",
                    "matches_per_pair": f"{row['matches_per_pair']:.4f}",
                    "selection_counts": json.dumps(row.get("selection_counts", {}), sort_keys=True),
                }
            )


def _write_per_clip_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "clip_name",
        "selector",
        "selected_threshold",
        "selection_reason",
        "query_contamination_pct",
        "static_retention_pct",
        "feature_contamination_pct",
        "match_contamination_pct",
        "essential_success_pct",
        "features_per_frame",
        "matches_per_pair",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "clip_name": row["clip_name"],
                    "selector": row["selector"],
                    "selected_threshold": "" if row.get("threshold") is None else f"{float(row['threshold']):.2f}",
                    "selection_reason": row.get("selection_reason", ""),
                    "query_contamination_pct": _fmt_pct(row["query_contamination"]),
                    "static_retention_pct": _fmt_pct(row["static_retention"]),
                    "feature_contamination_pct": _fmt_pct(row["feature_contamination"]),
                    "match_contamination_pct": _fmt_pct(row["match_contamination"]),
                    "essential_success_pct": _fmt_pct(row["essential_success"]),
                    "features_per_frame": f"{row['features_per_frame']:.4f}",
                    "matches_per_pair": f"{row['matches_per_pair']:.4f}",
                }
            )


def analyze_dataset(
    *,
    name: str,
    static_per_clip: Path,
    orb_per_clip: Path,
    output_dir: Path,
    policy: SelectorPolicy,
    tuned_policy: SelectorPolicy | None,
    oracle_min_success: float,
    oracle_min_retention: float,
) -> dict[str, Any]:
    joined = _load_joined(static_per_clip, orb_per_clip)
    selector_rows: dict[str, list[dict[str, Any]]] = {
        "raw": [],
        f"fixed_{policy.low_threshold:.2f}": [],
        f"fixed_{policy.mid_threshold:.2f}": [],
        f"fixed_{policy.high_threshold:.2f}": [],
        policy.name: [],
        "orb_selector_tuned": [],
        "oracle": [],
    }
    per_clip_rows: list[dict[str, Any]] = []
    selection_counts: dict[str, dict[str, int]] = {key: {} for key in selector_rows}

    for static_record, orb_record in joined:
        clip_name = _clip_key(static_record)
        rows = _threshold_rows_for_clip(static_record, orb_record)
        if not rows:
            continue
        raw = _attach_totals(_raw_row(static_record, orb_record), static_record)
        fixed_low = _attach_totals(_nearest_threshold(rows, policy.low_threshold), static_record)
        fixed_mid = _attach_totals(_nearest_threshold(rows, policy.mid_threshold), static_record)
        fixed_high = _attach_totals(_nearest_threshold(rows, policy.high_threshold), static_record)
        selected_rule, reason = _deployable_rule_select(rows, policy)
        selected_rule = _attach_totals(selected_rule, static_record)
        if tuned_policy is None:
            selected_tuned, tuned_reason = selected_rule, "same_as_default_rule"
        else:
            selected_tuned, tuned_reason = _deployable_rule_select(rows, tuned_policy)
            selected_tuned = _attach_totals(selected_tuned, static_record)
        selected_oracle = _attach_totals(
            _oracle_select(rows, min_essential_success=oracle_min_success, min_static_retention=oracle_min_retention),
            static_record,
        )
        chosen = {
            "raw": (raw, "raw"),
            f"fixed_{policy.low_threshold:.2f}": (fixed_low, "fixed"),
            f"fixed_{policy.mid_threshold:.2f}": (fixed_mid, "fixed"),
            f"fixed_{policy.high_threshold:.2f}": (fixed_high, "fixed"),
            policy.name: (selected_rule, reason),
            "orb_selector_tuned": (selected_tuned, tuned_reason),
            "oracle": (selected_oracle, "oracle_upper_bound"),
        }
        for selector, (row, row_reason) in chosen.items():
            out = dict(row)
            out["clip_name"] = clip_name
            out["selector"] = selector
            out["selection_reason"] = row_reason
            selector_rows[selector].append(out)
            threshold_label = "raw" if out.get("threshold") is None else f"{float(out['threshold']):.2f}"
            selection_counts[selector][threshold_label] = selection_counts[selector].get(threshold_label, 0) + 1
            per_clip_rows.append(out)

    summary_rows: list[dict[str, Any]] = []
    for selector, rows_for_selector in selector_rows.items():
        agg = _aggregate(rows_for_selector)
        agg["selector"] = selector
        agg["selection_counts"] = selection_counts[selector]
        summary_rows.append(agg)

    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(dataset_dir / "summary_table.csv", summary_rows)
    _write_per_clip_csv(dataset_dir / "per_clip_selection.csv", per_clip_rows)
    summary = {
        "dataset": name,
        "static_per_clip": str(static_per_clip.resolve()),
        "orb_per_clip": str(orb_per_clip.resolve()),
        "policy": policy.__dict__,
        "tuned_policy": None if tuned_policy is None else tuned_policy.__dict__,
        "oracle": {
            "min_essential_success": oracle_min_success,
            "min_static_retention": oracle_min_retention,
        },
        "summary": summary_rows,
        "outputs": {
            "summary_table": str((dataset_dir / "summary_table.csv").resolve()),
            "per_clip_selection": str((dataset_dir / "per_clip_selection.csv").resolve()),
        },
    }
    (dataset_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_dataset_arg(values: list[str]) -> list[tuple[str, Path, Path]]:
    out: list[tuple[str, Path, Path]] = []
    for value in values:
        parts = value.split("=")
        if len(parts) != 3:
            raise ValueError("Dataset must be NAME=STATIC_PER_CLIP_JSON=ORB_PER_CLIP_JSON")
        name, static_path, orb_path = parts
        out.append((name.strip(), Path(static_path.strip()), Path(orb_path.strip())))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME=STATIC_JSON=ORB_JSON",
        help="Per-clip static-map and ORB proxy JSON files.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--low-threshold", type=float, default=0.15)
    parser.add_argument("--mid-threshold", type=float, default=0.35)
    parser.add_argument("--high-threshold", type=float, default=0.55)
    parser.add_argument("--min-success", type=float, default=0.75)
    parser.add_argument("--min-features-per-frame", type=float, default=120.0)
    parser.add_argument("--min-matches-per-pair", type=float, default=50.0)
    parser.add_argument("--min-inlier-rate", type=float, default=0.55)
    parser.add_argument("--min-static-mask-fraction", type=float, default=0.10)
    parser.add_argument(
        "--tune-on",
        default=None,
        help="Optional dataset name to tune rule parameters on; evaluate tuned policy on all datasets.",
    )
    parser.add_argument("--tune-min-success", type=float, default=0.80)
    parser.add_argument("--tune-min-retention", type=float, default=0.80)
    parser.add_argument("--tune-success-penalty", type=float, default=3.0)
    parser.add_argument("--tune-retention-penalty", type=float, default=1.5)
    parser.add_argument("--tune-feature-weight", type=float, default=1.0)
    parser.add_argument("--tune-match-weight", type=float, default=0.5)
    parser.add_argument("--oracle-min-success", type=float, default=0.75)
    parser.add_argument("--oracle-min-retention", type=float, default=0.80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = SelectorPolicy(
        name=(
            f"orb_selector_low{args.low_threshold:.2f}_high{args.high_threshold:.2f}"
        ),
        low_threshold=float(args.low_threshold),
        mid_threshold=float(args.mid_threshold),
        high_threshold=float(args.high_threshold),
        min_success=float(args.min_success),
        min_features_per_frame=float(args.min_features_per_frame),
        min_matches_per_pair=float(args.min_matches_per_pair),
        min_inlier_rate=float(args.min_inlier_rate),
        min_static_mask_fraction=float(args.min_static_mask_fraction),
    )
    datasets = _parse_dataset_arg(args.dataset)
    tuned_policy: SelectorPolicy | None = None
    if args.tune_on:
        matches = [item for item in datasets if item[0] == str(args.tune_on)]
        if not matches:
            raise ValueError(f"--tune-on={args.tune_on!r} does not match any --dataset name")
        tune_name, tune_static, tune_orb = matches[0]
        tuned_policy, tuned_agg = _grid_search_policy(
            _load_joined(tune_static, tune_orb),
            low_threshold=float(args.low_threshold),
            mid_threshold=float(args.mid_threshold),
            high_threshold=float(args.high_threshold),
            min_success=float(args.tune_min_success),
            min_retention=float(args.tune_min_retention),
            success_penalty=float(args.tune_success_penalty),
            retention_penalty=float(args.tune_retention_penalty),
            feature_weight=float(args.tune_feature_weight),
            match_weight=float(args.tune_match_weight),
        )
        tuned_dir = out_dir / "tuned_policy"
        tuned_dir.mkdir(parents=True, exist_ok=True)
        (tuned_dir / "summary.json").write_text(
            json.dumps(
                {
                    "tune_dataset": tune_name,
                    "policy": tuned_policy.__dict__,
                    "train_aggregate": tuned_agg,
                    "objective": {
                        "min_success": float(args.tune_min_success),
                        "min_retention": float(args.tune_min_retention),
                        "success_penalty": float(args.tune_success_penalty),
                        "retention_penalty": float(args.tune_retention_penalty),
                        "feature_weight": float(args.tune_feature_weight),
                        "match_weight": float(args.tune_match_weight),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    summaries = []
    for name, static_path, orb_path in datasets:
        summaries.append(
            analyze_dataset(
                name=name,
                static_per_clip=static_path,
                orb_per_clip=orb_path,
                output_dir=out_dir,
                policy=policy,
                tuned_policy=tuned_policy,
                oracle_min_success=float(args.oracle_min_success),
                oracle_min_retention=float(args.oracle_min_retention),
            )
        )
    index = {
        "policy": policy.__dict__,
        "tuned_policy": None if tuned_policy is None else tuned_policy.__dict__,
        "datasets": [
            {
                "dataset": item["dataset"],
                "outputs": item["outputs"],
            }
            for item in summaries
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Saved selector analysis to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
