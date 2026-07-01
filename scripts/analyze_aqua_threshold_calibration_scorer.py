#!/usr/bin/env python3
"""Train a frozen-output Aqua static-threshold calibration scorer.

This script is intentionally checkpoint-free: it reuses existing static-map and
ORB/SfM proxy per-clip JSON files, trains a tiny ridge-regression scorer over
deployable clip/threshold diagnostics, and selects a static-confidence threshold
per clip. Ground-truth contamination/retention metrics are used only to build
training targets and to evaluate selected thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from analyze_aqua_adaptive_threshold_selector import (
    _aggregate,
    _attach_totals,
    _clip_key,
    _load_joined,
    _nearest_threshold,
    _raw_row,
    _safe_float,
    _threshold_rows_for_clip,
    _write_per_clip_csv,
    _write_summary_csv,
)


@dataclass
class ScorerConfig:
    min_threshold: float = 0.05
    max_threshold: float = 0.90
    min_success: float = 0.80
    min_retention: float = 0.80
    query_weight: float = 0.75
    feature_weight: float = 1.00
    match_weight: float = 0.50
    success_penalty: float = 3.00
    retention_penalty: float = 1.50
    ridge_alpha: float = 1.0


@dataclass
class RidgeScorer:
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_std = (x - self.mean) / self.std
        x_aug = np.concatenate([np.ones((x_std.shape[0], 1), dtype=np.float64), x_std], axis=1)
        return x_aug @ self.coef

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "coef": self.coef.tolist(),
        }


def _read_clip_names(path: Path | None) -> set[str]:
    if path is None:
        return set()
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        p = Path(value)
        if p.name == "manifest.json":
            names.add(p.parent.name)
        else:
            names.add(p.stem or p.name)
    return names


def _score_stats(static_record: dict[str, Any]) -> dict[str, float]:
    stats = static_record.get("score_stats", {})
    return {
        "dynamic_prob_mean": _safe_float(stats.get("dynamic_prob_mean"), 0.0),
        "particle_prob_mean": _safe_float(stats.get("particle_prob_mean"), 0.0),
        "confidence_prob_mean": _safe_float(stats.get("confidence_prob_mean"), 0.0),
        "visibility_prob_mean": _safe_float(stats.get("visibility_prob_mean"), 0.0),
        "static_confidence_mean": _safe_float(stats.get("static_confidence_mean"), 0.0),
    }


def _feature_vector(row: dict[str, Any], static_record: dict[str, Any]) -> tuple[list[str], list[float]]:
    stats = _score_stats(static_record)
    threshold = _safe_float(row.get("threshold"), 0.0)
    features_per_frame = _safe_float(row.get("features_per_frame"), 0.0)
    matches_per_pair = _safe_float(row.get("matches_per_pair"), 0.0)
    inliers_per_pair = _safe_float(row.get("essential_inliers_per_pair"), 0.0)
    essential_success = _safe_float(row.get("essential_success"), 0.0)
    inlier_rate = _safe_float(row.get("essential_inlier_rate"), 0.0)
    static_mask_fraction = _safe_float(row.get("static_mask_fraction"), 0.0)

    items = [
        ("threshold", threshold),
        ("threshold_sq", threshold * threshold),
        ("dynamic_prob_mean", stats["dynamic_prob_mean"]),
        ("particle_prob_mean", stats["particle_prob_mean"]),
        ("confidence_prob_mean", stats["confidence_prob_mean"]),
        ("visibility_prob_mean", stats["visibility_prob_mean"]),
        ("static_confidence_mean", stats["static_confidence_mean"]),
        ("log_features_per_frame", math.log1p(max(0.0, features_per_frame))),
        ("log_matches_per_pair", math.log1p(max(0.0, matches_per_pair))),
        ("log_inliers_per_pair", math.log1p(max(0.0, inliers_per_pair))),
        ("essential_success", essential_success),
        ("essential_inlier_rate", inlier_rate),
        ("static_mask_fraction", static_mask_fraction),
        ("threshold_x_success", threshold * essential_success),
        ("threshold_x_static_mask", threshold * static_mask_fraction),
    ]
    return [name for name, _ in items], [float(value) for _, value in items]


def _candidate_cost(row: dict[str, Any], config: ScorerConfig) -> float:
    query = _safe_float(row.get("query_contamination"), 1.0)
    feature = _safe_float(row.get("feature_contamination"), 1.0)
    match = _safe_float(row.get("match_contamination"), 1.0)
    success = _safe_float(row.get("essential_success"), 0.0)
    retention = _safe_float(row.get("static_retention"), 0.0)
    return (
        config.query_weight * query
        + config.feature_weight * feature
        + config.match_weight * match
        + config.success_penalty * max(0.0, config.min_success - success)
        + config.retention_penalty * max(0.0, config.min_retention - retention)
    )


def _candidate_rows(
    static_record: dict[str, Any],
    orb_record: dict[str, Any],
    config: ScorerConfig,
) -> list[dict[str, Any]]:
    rows = []
    for row in _threshold_rows_for_clip(static_record, orb_record):
        threshold = _safe_float(row.get("threshold"), -1.0)
        if config.min_threshold <= threshold <= config.max_threshold:
            rows.append(_attach_totals(row, static_record))
    return rows


def _filter_joined(
    joined: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    include_names: set[str],
    exclude_names: set[str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out = []
    for static_record, orb_record in joined:
        name = _clip_key(static_record)
        if include_names and name not in include_names:
            continue
        if exclude_names and name in exclude_names:
            continue
        out.append((static_record, orb_record))
    return out


def _fit_ridge(
    joined: list[tuple[dict[str, Any], dict[str, Any]]],
    config: ScorerConfig,
) -> tuple[RidgeScorer, dict[str, Any]]:
    xs: list[list[float]] = []
    ys: list[float] = []
    feature_names: list[str] | None = None
    selected_oracle_rows: list[dict[str, Any]] = []

    for static_record, orb_record in joined:
        rows = _candidate_rows(static_record, orb_record, config)
        if not rows:
            continue
        best = min(rows, key=lambda row: _candidate_cost(row, config))
        selected_oracle_rows.append(best)
        for row in rows:
            names, values = _feature_vector(row, static_record)
            if feature_names is None:
                feature_names = names
            xs.append(values)
            ys.append(_candidate_cost(row, config))

    if not xs or feature_names is None:
        raise ValueError("No training candidates found for calibration scorer.")

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    x_std = (x - mean) / std
    x_aug = np.concatenate([np.ones((x_std.shape[0], 1), dtype=np.float64), x_std], axis=1)
    reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(config.ridge_alpha)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
    scorer = RidgeScorer(feature_names=feature_names, mean=mean, std=std, coef=coef)
    train_pred = scorer.predict(x)
    train_rmse = float(np.sqrt(np.mean((train_pred - y) ** 2)))
    train_oracle_agg = _aggregate(selected_oracle_rows)
    diagnostics = {
        "train_clips": len(joined),
        "train_candidates": int(x.shape[0]),
        "train_cost_mean": float(y.mean()),
        "train_cost_rmse": train_rmse,
        "train_objective_oracle": train_oracle_agg,
    }
    return scorer, diagnostics


def _select_with_scorer(
    rows: list[dict[str, Any]],
    static_record: dict[str, Any],
    scorer: RidgeScorer,
) -> tuple[dict[str, Any], float]:
    x_rows = []
    for row in rows:
        _, values = _feature_vector(row, static_record)
        x_rows.append(values)
    x = np.asarray(x_rows, dtype=np.float64)
    pred = scorer.predict(x)
    idx = int(np.argmin(pred))
    selected = dict(rows[idx])
    selected["predicted_cost"] = float(pred[idx])
    return selected, float(pred[idx])


def _objective_oracle(rows: list[dict[str, Any]], config: ScorerConfig) -> dict[str, Any]:
    selected = min(rows, key=lambda row: _candidate_cost(row, config))
    out = dict(selected)
    out["objective_cost"] = _candidate_cost(out, config)
    return out


def _summarize_selection(
    *,
    name: str,
    joined: list[tuple[dict[str, Any], dict[str, Any]]],
    config: ScorerConfig,
    scorer: RidgeScorer,
    output_dir: Path,
) -> dict[str, Any]:
    selector_rows: dict[str, list[dict[str, Any]]] = {
        "raw": [],
        "fixed_0.15": [],
        "fixed_0.35": [],
        "fixed_0.55": [],
        "calibration_scorer": [],
        "objective_oracle": [],
    }
    selection_counts: dict[str, dict[str, int]] = {key: {} for key in selector_rows}
    per_clip_rows: list[dict[str, Any]] = []

    for static_record, orb_record in joined:
        rows = _candidate_rows(static_record, orb_record, config)
        if not rows:
            continue
        raw = _attach_totals(_raw_row(static_record, orb_record), static_record)
        choices = {
            "raw": (raw, "raw"),
            "fixed_0.15": (_nearest_threshold(rows, 0.15), "fixed"),
            "fixed_0.35": (_nearest_threshold(rows, 0.35), "fixed"),
            "fixed_0.55": (_nearest_threshold(rows, 0.55), "fixed"),
            "calibration_scorer": (_select_with_scorer(rows, static_record, scorer)[0], "ridge_predicted_lowest_cost"),
            "objective_oracle": (_objective_oracle(rows, config), "gt_objective_upper_bound"),
        }
        clip_name = _clip_key(static_record)
        for selector, (row, reason) in choices.items():
            out = _attach_totals(row, static_record)
            out["clip_name"] = clip_name
            out["selector"] = selector
            out["selection_reason"] = reason
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
        "clips": len(joined),
        "config": config.__dict__,
        "summary": summary_rows,
        "outputs": {
            "summary_table": str((dataset_dir / "summary_table.csv").resolve()),
            "per_clip_selection": str((dataset_dir / "per_clip_selection.csv").resolve()),
        },
    }
    (dataset_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_dataset(values: list[str]) -> list[tuple[str, Path, Path]]:
    out = []
    for value in values:
        parts = value.split("=")
        if len(parts) != 3:
            raise ValueError("Dataset must be NAME=STATIC_PER_CLIP_JSON=ORB_PER_CLIP_JSON")
        out.append((parts[0].strip(), Path(parts[1].strip()), Path(parts[2].strip())))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=STATIC_JSON=ORB_JSON")
    parser.add_argument("--train-dataset", required=True, help="Dataset name to train the scorer on.")
    parser.add_argument("--train-include-clips", default=None, help="Optional file of clip names or manifest paths.")
    parser.add_argument("--train-exclude-clips", default=None, help="Optional file of clip names or manifest paths.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-threshold", type=float, default=0.05)
    parser.add_argument("--max-threshold", type=float, default=0.90)
    parser.add_argument("--min-success", type=float, default=0.80)
    parser.add_argument("--min-retention", type=float, default=0.80)
    parser.add_argument("--query-weight", type=float, default=0.75)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--match-weight", type=float, default=0.5)
    parser.add_argument("--success-penalty", type=float, default=3.0)
    parser.add_argument("--retention-penalty", type=float, default=1.5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ScorerConfig(
        min_threshold=float(args.min_threshold),
        max_threshold=float(args.max_threshold),
        min_success=float(args.min_success),
        min_retention=float(args.min_retention),
        query_weight=float(args.query_weight),
        feature_weight=float(args.feature_weight),
        match_weight=float(args.match_weight),
        success_penalty=float(args.success_penalty),
        retention_penalty=float(args.retention_penalty),
        ridge_alpha=float(args.ridge_alpha),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = _parse_dataset(args.dataset)
    joined_by_name = {name: _load_joined(static_path, orb_path) for name, static_path, orb_path in datasets}
    if args.train_dataset not in joined_by_name:
        raise ValueError(f"Train dataset {args.train_dataset!r} not among {[name for name, _, _ in datasets]}")

    include_names = _read_clip_names(Path(args.train_include_clips)) if args.train_include_clips else set()
    exclude_names = _read_clip_names(Path(args.train_exclude_clips)) if args.train_exclude_clips else set()
    train_joined = _filter_joined(
        joined_by_name[args.train_dataset],
        include_names=include_names,
        exclude_names=exclude_names,
    )
    scorer, train_diagnostics = _fit_ridge(train_joined, config)
    (output_dir / "scorer.json").write_text(json.dumps(scorer.to_json(), indent=2), encoding="utf-8")

    summaries = []
    for name, _, _ in datasets:
        summaries.append(
            _summarize_selection(
                name=name,
                joined=joined_by_name[name],
                config=config,
                scorer=scorer,
                output_dir=output_dir,
            )
        )

    index = {
        "method": "ridge_threshold_calibration_scorer",
        "train_dataset": args.train_dataset,
        "train_include_clips": str(Path(args.train_include_clips).resolve()) if args.train_include_clips else None,
        "train_exclude_clips": str(Path(args.train_exclude_clips).resolve()) if args.train_exclude_clips else None,
        "config": config.__dict__,
        "scorer": scorer.to_json(),
        "train_diagnostics": train_diagnostics,
        "datasets": [{"dataset": item["dataset"], "outputs": item["outputs"]} for item in summaries],
    }
    (output_dir / "summary.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Saved calibration scorer analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
