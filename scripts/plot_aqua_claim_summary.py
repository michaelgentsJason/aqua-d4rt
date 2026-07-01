#!/usr/bin/env python3
"""Plot a compact Aqua-D4RT claim summary figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_policy_csv(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            policy = str(row["policy"])
            rows[policy] = {}
            for key, value in row.items():
                if key in {"policy", "selection_counts", "reason_counts"}:
                    continue
                rows[policy][key] = float(value)
    return rows


def _annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.1f}", scale: float = 1.0) -> None:
    for bar in bars:
        value = bar.get_height() * scale
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.015 * max(1.0, ax.get_ylim()[1]),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="tmp/aqua_case_visuals/aqua_claim_summary")
    parser.add_argument(
        "--r099-summary",
        default="tmp/aqua_adaptive_v3_t073_full_stress4_selector_v3_rawfallback/summary.csv",
    )
    parser.add_argument(
        "--r101-summary",
        default="tmp/aqua_adaptive_v3_t073_allstress_seed42_selector_v3_rawfallback/summary.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    r099 = _read_policy_csv(Path(args.r099_summary))
    r101 = _read_policy_csv(Path(args.r101_summary))
    r099_final = r099["soft_raw_fallback_margin0.10_rawmin0.60"]
    r101_final = r101["soft_raw_fallback_margin0.10_rawmin0.60"]

    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    colors = {"raw": "#6b7280", "aqua": "#0f766e", "accent": "#2563eb", "warn": "#b45309"}
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.4))

    ax = axes[0, 0]
    datasets = ["Synthetic", "WebUOT"]
    raw_contam = np.asarray([10.82, 10.86])
    aqua_contam = np.asarray([0.39, 4.47])
    x = np.arange(len(datasets))
    width = 0.36
    b1 = ax.bar(x - width / 2, raw_contam, width, label="Raw query map", color=colors["raw"])
    b2 = ax.bar(x + width / 2, aqua_contam, width, label="Aqua static", color=colors["aqua"])
    ax.set_ylabel("Point contamination (%)")
    ax.set_xticks(x, datasets)
    ax.set_ylim(0, 12.5)
    ax.legend(frameon=False, fontsize=8)
    _annotate_bars(ax, b1)
    _annotate_bars(ax, b2)
    ax.text(-0.42, 12.0, "(a) Static query-map cleanliness", fontsize=10, fontweight="bold")

    ax = axes[0, 1]
    retention = np.asarray([93.00, 75.53])
    bars = ax.bar(datasets, retention, color=[colors["accent"], colors["accent"]])
    ax.set_ylabel("Static retention (%)")
    ax.set_ylim(0, 105)
    _annotate_bars(ax, bars)
    ax.text(-0.46, 101, "(b) Static support retained", fontsize=10, fontweight="bold")
    ax.text(-0.46, -19, "Note: WebUOT mask is tracked-target bbox, not full fish instance GT.", fontsize=8, color="#4b5563")

    ax = axes[1, 0]
    methods = ["Raw", "R099"]
    metrics = ["Pose eval\nsuccess", "Input\nreg.", "Feature\ncontam.", "ATE\nRMSE"]
    raw_vals = [
        r099["raw"]["pose_eval_success"] * 100.0,
        r099["raw"]["input_registration_rate"] * 100.0,
        r099["raw"]["feature_contamination"] * 100.0,
        r099["raw"]["ate_rmse"] * 100.0,
    ]
    final_vals = [
        r099_final["pose_eval_success"] * 100.0,
        r099_final["input_registration_rate"] * 100.0,
        r099_final["feature_contamination"] * 100.0,
        r099_final["ate_rmse"] * 100.0,
    ]
    x = np.arange(len(metrics))
    b1 = ax.bar(x - width / 2, raw_vals, width, label=methods[0], color=colors["raw"])
    b2 = ax.bar(x + width / 2, final_vals, width, label=methods[1], color=colors["aqua"])
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Percent, except ATE x100")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    for bars in (b1, b2):
        _annotate_bars(ax, bars)
    ax.text(-0.55, 96, "(c) Tank Stress4 R099 selector", fontsize=10, fontweight="bold")

    ax = axes[1, 1]
    metrics = ["Input\nreg.", "Feature\ncontam.", "RPE\nx100", "ATE\nx100"]
    raw_vals = [
        r101["raw"]["input_registration_rate"] * 100.0,
        r101["raw"]["feature_contamination"] * 100.0,
        r101["raw"]["rpe_trans_rmse"] * 100.0,
        r101["raw"]["ate_rmse"] * 100.0,
    ]
    final_vals = [
        r101_final["input_registration_rate"] * 100.0,
        r101_final["feature_contamination"] * 100.0,
        r101_final["rpe_trans_rmse"] * 100.0,
        r101_final["ate_rmse"] * 100.0,
    ]
    x = np.arange(len(metrics))
    b1 = ax.bar(x - width / 2, raw_vals, width, label="Raw", color=colors["raw"])
    b2 = ax.bar(x + width / 2, final_vals, width, label="Adaptive", color=colors["aqua"])
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Percent, except errors x100")
    ax.set_ylim(0, 55)
    ax.legend(frameon=False, fontsize=8)
    for bars in (b1, b2):
        _annotate_bars(ax, bars)
    ax.text(-0.55, 52.5, "(d) All-stress seed42 trade-off", fontsize=10, fontweight="bold")

    fig.tight_layout(pad=1.0)
    png = out_dir / "aqua_claim_summary.png"
    pdf = out_dir / "aqua_claim_summary.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved {png}")
    print(f"Saved {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
