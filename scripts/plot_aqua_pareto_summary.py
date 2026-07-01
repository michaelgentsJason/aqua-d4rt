#!/usr/bin/env python3
"""Plot Aqua-D4RT contamination/registration/pose Pareto summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_r099(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for row in _read_csv_rows(path):
        policy = str(row["policy"])
        rows[policy] = {}
        for key, value in row.items():
            if key in {"policy", "selection_counts", "reason_counts"}:
                continue
            rows[policy][key] = float(value)
    return rows


def _read_summary_table(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for row in _read_csv_rows(path):
        variant = str(row.get("variant", ""))
        if not variant:
            continue
        rows[variant] = {}
        for key, value in row.items():
            if key == "variant" or value == "":
                continue
            try:
                rows[variant][key] = float(value)
            except ValueError:
                continue
    return rows


def _pretty(name: str) -> str:
    mapping = {
        "raw": "Raw",
        "raw_all_pixels": "Raw",
        "v3_pose_soft": "v3 pose-soft",
        "soft_raw_fallback_margin0.10_rawmin0.60": "R099 fallback",
        "temporal_rgb_static": "Temporal RGB",
        "temporal_rgb_prefilter_static": "Temporal RGB",
        "oracle_gt_static": "Oracle GT",
        "aqua_static_conf_ge_0p550": "Aqua hard",
        "aqua_static_conf_ge_0p550_full": "Aqua hard",
        "aqua_static_conf_ge_0p550_slam_retain": "Aqua+retain",
    }
    if name in mapping:
        return mapping[name]
    if name.endswith("_static"):
        return name[: -len("_static")].replace("_", " ")
    return name.replace("_", " ")


def _style(name: str) -> tuple[str, str]:
    low = name.lower()
    if "raw" in low:
        return "#6b7280", "o"
    if "oracle" in low:
        return "#7c3aed", "*"
    if "sam" in low:
        return "#db2777", "s"
    if "dino" in low or "grounding" in low:
        return "#ea580c", "D"
    if "temporal" in low:
        return "#2563eb", "^"
    if "fallback" in low or "r099" in low:
        return "#0f766e", "P"
    return "#059669", "o"


def _plot_points(ax: plt.Axes, points: list[dict[str, Any]], x_key: str, y_key: str, *, x_scale: float, y_scale: float) -> None:
    for point in points:
        color, marker = _style(str(point["name"]))
        x = float(point[x_key]) * x_scale
        y = float(point[y_key]) * y_scale
        ax.scatter([x], [y], s=70, color=color, marker=marker, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(
            _pretty(str(point["name"])),
            (x, y),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/aqua_pareto")
    parser.add_argument(
        "--r099-summary",
        default="tmp/aqua_adaptive_v3_t073_full_stress4_selector_v3_rawfallback/summary.csv",
    )
    parser.add_argument("--orb-summary", default="", help="Optional ORB proxy summary_table.csv with external baselines.")
    parser.add_argument("--static-summary", default="", help="Optional static-map summary_table.csv with external baselines.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "DejaVu Serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "figure.dpi": 160,
        }
    )

    panels = 1
    orb_rows: dict[str, dict[str, float]] = {}
    static_rows: dict[str, dict[str, float]] = {}
    if args.orb_summary:
        orb_path = Path(args.orb_summary)
        if orb_path.exists():
            orb_rows = _read_summary_table(orb_path)
            panels += 1
    if args.static_summary:
        static_path = Path(args.static_summary)
        if static_path.exists():
            static_rows = _read_summary_table(static_path)
            panels += 1

    fig, axes = plt.subplots(1, panels, figsize=(4.2 * panels, 3.2), squeeze=False)
    ax_list = list(axes[0])

    r099 = _read_r099(Path(args.r099_summary))
    r099_points = []
    for name in ("raw", "v3_pose_soft", "soft_raw_fallback_margin0.10_rawmin0.60"):
        if name in r099:
            r099_points.append({"name": name, **r099[name]})
    ax = ax_list.pop(0)
    _plot_points(
        ax,
        r099_points,
        x_key="feature_contamination",
        y_key="pose_eval_success",
        x_scale=100.0,
        y_scale=100.0,
    )
    ax.set_xlabel("Feature contamination (%)")
    ax.set_ylabel("Pose-eval success (%)")
    ax.grid(True, alpha=0.2, linewidth=0.6)

    if orb_rows and ax_list:
        ax = ax_list.pop(0)
        wanted = [
            "raw_all_pixels",
            "temporal_rgb_static",
            "aqua_static_conf_ge_0p550",
            "aqua_static_conf_ge_0p550_full",
            "groundingdino_box_static",
            "groundingdino_sam_static",
            "oracle_gt_static",
        ]
        points = [{"name": name, **orb_rows[name]} for name in wanted if name in orb_rows]
        if not points:
            points = [{"name": name, **metrics} for name, metrics in orb_rows.items()]
        _plot_points(
            ax,
            points,
            x_key="feature_contamination",
            y_key="essential_success_rate",
            x_scale=100.0,
            y_scale=100.0,
        )
        ax.set_xlabel("Feature contamination (%)")
        ax.set_ylabel("Essential-matrix success (%)")
        ax.grid(True, alpha=0.2, linewidth=0.6)

    if static_rows and ax_list:
        ax = ax_list.pop(0)
        wanted = [
            "all_d4rt_points",
            "temporal_rgb_prefilter_static",
            "aqua_static_conf_ge_0p550",
            "aqua_static_conf_ge_0p550_full",
            "groundingdino_box_static",
            "groundingdino_sam_static",
            "oracle_gt_static",
        ]
        points = [{"name": name, **static_rows[name]} for name in wanted if name in static_rows]
        if not points:
            points = [{"name": name, **metrics} for name, metrics in static_rows.items()]
        _plot_points(
            ax,
            points,
            x_key="point_contamination",
            y_key="point_static_retention",
            x_scale=100.0,
            y_scale=100.0,
        )
        ax.set_xlabel("Query contamination (%)")
        ax.set_ylabel("Static retention (%)")
        ax.grid(True, alpha=0.2, linewidth=0.6)

    fig.tight_layout(pad=0.8)
    png = out_dir / "aqua_pareto_summary.png"
    pdf = out_dir / "aqua_pareto_summary.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    summary = {
        "r099_summary": str(Path(args.r099_summary).resolve()),
        "orb_summary": str(Path(args.orb_summary).resolve()) if args.orb_summary else None,
        "static_summary": str(Path(args.static_summary).resolve()) if args.static_summary else None,
        "outputs": {"png": str(png.resolve()), "pdf": str(pdf.resolve())},
        "notes": [
            "R099 is a self-diagnostic multi-pass selector: raw and pose-soft reconstructions are both evaluated.",
            "Optional ORB/static panels are added when their summary_table.csv files are provided.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {png}")
    print(f"Saved {pdf}")
    print(f"Saved {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
