#!/usr/bin/env python3
"""Select and plot Aqua static-confidence operating points.

This script turns a static-threshold sweep into paper-facing evidence for two
different modes:

* clean-map mode: high threshold, low query/match contamination;
* front-end mode: lower threshold, better essential-matrix success.

It reads evaluator CSV files instead of hard-coding numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, default)
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _threshold_from_variant(name: str) -> float | None:
    match = re.search(r"ge_([0-9]+)p([0-9]+)", name)
    if not match:
        return None
    return float(f"{match.group(1)}.{match.group(2)}")


def _variant_for_threshold(threshold: float) -> str:
    return f"aqua_static_conf_ge_{threshold:.3f}".replace(".", "p").rstrip("0").rstrip("p")


def _read_orb_thresholds(path: Path) -> tuple[list[dict[str, float]], dict[str, dict[str, float]]]:
    baselines: dict[str, dict[str, float]] = {}
    thresholds: list[dict[str, float]] = []
    for row in _read_csv(path):
        variant = str(row.get("variant", ""))
        parsed = {
            "feature_contamination": _float(row, "feature_contamination"),
            "match_contamination": _float(row, "match_contamination_mean"),
            "essential_success_rate": _float(row, "essential_success_rate"),
            "features_per_frame": _float(row, "features_per_frame_mean"),
            "matches_per_pair": _float(row, "matches_per_pair_mean"),
            "essential_inlier_rate": _float(row, "essential_inlier_rate_mean"),
        }
        threshold = _threshold_from_variant(variant)
        if threshold is None:
            baselines[variant] = parsed
            continue
        parsed["threshold"] = threshold
        parsed["variant"] = variant
        thresholds.append(parsed)
    thresholds.sort(key=lambda item: item["threshold"])
    return thresholds, baselines


def _read_static_thresholds(path: Path | None) -> dict[float, dict[str, float]]:
    if path is None or not path.exists():
        return {}
    rows: dict[float, dict[str, float]] = {}
    for row in _read_csv(path):
        threshold = _float(row, "threshold")
        if not math.isfinite(threshold):
            threshold_from_variant = _threshold_from_variant(str(row.get("variant", "")))
            if threshold_from_variant is None:
                continue
            threshold = threshold_from_variant
        rows[round(threshold, 6)] = {
            "query_contamination": _float(row, "point_contamination"),
            "static_retention": _float(row, "point_static_retention"),
            "transient_rejection": _float(row, "point_transient_rejection"),
            "voxel_contamination": _float(row, "voxel_contamination_any"),
            "voxel_retention": _float(row, "voxel_static_support_retention"),
        }
    return rows


def _with_static(
    thresholds: list[dict[str, float]],
    static_rows: dict[float, dict[str, float]],
) -> list[dict[str, float]]:
    merged: list[dict[str, float]] = []
    for row in thresholds:
        out = dict(row)
        out.update(static_rows.get(round(float(row["threshold"]), 6), {}))
        merged.append(out)
    return merged


def _nearest(rows: list[dict[str, float]], threshold: float) -> dict[str, float]:
    return min(rows, key=lambda row: abs(float(row["threshold"]) - float(threshold)))


def _select_operating_points(
    rows: list[dict[str, float]],
    *,
    high_success_min: float,
    balanced_success_min: float,
    clean_threshold: float,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    def valid(metric: str) -> list[dict[str, float]]:
        return [row for row in rows if math.isfinite(float(row.get(metric, float("nan"))))]

    high_candidates = [
        row for row in valid("essential_success_rate") if row["essential_success_rate"] >= high_success_min
    ]
    if high_candidates:
        high = min(high_candidates, key=lambda row: (row["feature_contamination"], row["match_contamination"]))
    else:
        high = max(valid("essential_success_rate"), key=lambda row: row["essential_success_rate"])

    balanced_candidates = [
        row for row in valid("essential_success_rate") if row["essential_success_rate"] >= balanced_success_min
    ]
    if balanced_candidates:
        balanced = min(balanced_candidates, key=lambda row: row["match_contamination"])
    else:
        balanced = min(valid("match_contamination"), key=lambda row: row["match_contamination"])

    clean = _nearest(rows, clean_threshold)

    selections = [
        ("registration_first", high, f"minimum feature contamination with E-success >= {high_success_min:.2f}"),
        ("balanced", balanced, f"minimum match contamination with E-success >= {balanced_success_min:.2f}"),
        ("clean_map_default", clean, f"nearest threshold to clean-map default {clean_threshold:.2f}"),
    ]
    seen: set[tuple[str, float]] = set()
    output: list[dict[str, Any]] = []
    for name, row, rule in selections:
        key = (name, float(row["threshold"]))
        if key in seen:
            continue
        seen.add(key)
        merged = dict(row)
        merged["operating_point"] = name
        merged["selection_rule"] = rule
        output.append(merged)
    return output


def _pct(value: float) -> str:
    if not math.isfinite(float(value)):
        return "--"
    return f"{100.0 * float(value):.2f}"


def _num(value: float) -> str:
    if not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.1f}"


def _write_selected_csv(path: Path, rows: list[dict[str, Any]], baselines: dict[str, dict[str, float]]) -> None:
    fieldnames = [
        "operating_point",
        "threshold",
        "query_contamination_pct",
        "static_retention_pct",
        "feature_contamination_pct",
        "match_contamination_pct",
        "essential_success_pct",
        "features_per_frame",
        "matches_per_pair",
        "selection_rule",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "operating_point": row["operating_point"],
                    "threshold": f"{float(row['threshold']):.2f}",
                    "query_contamination_pct": _pct(float(row.get("query_contamination", float("nan")))),
                    "static_retention_pct": _pct(float(row.get("static_retention", float("nan")))),
                    "feature_contamination_pct": _pct(float(row.get("feature_contamination", float("nan")))),
                    "match_contamination_pct": _pct(float(row.get("match_contamination", float("nan")))),
                    "essential_success_pct": _pct(float(row.get("essential_success_rate", float("nan")))),
                    "features_per_frame": _num(float(row.get("features_per_frame", float("nan")))),
                    "matches_per_pair": _num(float(row.get("matches_per_pair", float("nan")))),
                    "selection_rule": row["selection_rule"],
                }
            )
        for name in ("raw_all_pixels", "aqua_pred_transient_filter", "groundingdino_box_static", "oracle_gt_static"):
            if name not in baselines:
                continue
            row = baselines[name]
            writer.writerow(
                {
                    "operating_point": name,
                    "threshold": "",
                    "query_contamination_pct": "",
                    "static_retention_pct": "",
                    "feature_contamination_pct": _pct(float(row.get("feature_contamination", float("nan")))),
                    "match_contamination_pct": _pct(float(row.get("match_contamination", float("nan")))),
                    "essential_success_pct": _pct(float(row.get("essential_success_rate", float("nan")))),
                    "features_per_frame": _num(float(row.get("features_per_frame", float("nan")))),
                    "matches_per_pair": _num(float(row.get("matches_per_pair", float("nan")))),
                    "selection_rule": "baseline",
                }
            )


def _write_tex(path: Path, rows: list[dict[str, Any]], baselines: dict[str, dict[str, float]]) -> None:
    labels = {
        "registration_first": "Aqua registration-first",
        "balanced": "Aqua balanced",
        "clean_map_default": "Aqua clean-map default",
        "raw_all_pixels": "Raw",
        "aqua_pred_transient_filter": "Aqua transient filter",
        "groundingdino_box_static": "GroundingDINO-box",
        "oracle_gt_static": "Oracle GT static",
    }
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & $\\tau_s$ & Query contam. & Static ret. & Feature contam. & Match contam. & E success \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{labels.get(str(row['operating_point']), str(row['operating_point']))} & "
            f"{float(row['threshold']):.2f} & "
            f"{_pct(float(row.get('query_contamination', float('nan'))))} & "
            f"{_pct(float(row.get('static_retention', float('nan'))))} & "
            f"{_pct(float(row.get('feature_contamination', float('nan'))))} & "
            f"{_pct(float(row.get('match_contamination', float('nan'))))} & "
            f"{_pct(float(row.get('essential_success_rate', float('nan'))))} \\\\"
        )
    lines.append("\\midrule")
    for name in ("raw_all_pixels", "aqua_pred_transient_filter", "groundingdino_box_static", "oracle_gt_static"):
        if name not in baselines:
            continue
        row = baselines[name]
        lines.append(
            f"{labels.get(name, name)} & -- & -- & -- & "
            f"{_pct(float(row.get('feature_contamination', float('nan'))))} & "
            f"{_pct(float(row.get('match_contamination', float('nan'))))} & "
            f"{_pct(float(row.get('essential_success_rate', float('nan'))))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(
    path_pdf: Path,
    path_png: Path,
    rows: list[dict[str, float]],
    selected: list[dict[str, Any]],
    baselines: dict[str, dict[str, float]],
) -> None:
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
    x = [100.0 * row["feature_contamination"] for row in rows]
    y = [100.0 * row["essential_success_rate"] for row in rows]
    c = [row["threshold"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    scatter = ax.scatter(x, y, c=c, cmap="viridis", s=38, edgecolor="white", linewidth=0.5, zorder=3)
    ax.plot(x, y, color="#9ca3af", linewidth=0.9, zorder=2)
    op_labels = {
        "registration_first": "Reg-first",
        "balanced": "Balanced",
        "clean_map_default": "Clean-map",
    }
    op_offsets = {
        "registration_first": (14, 13),
        "balanced": (14, -2),
        "clean_map_default": (14, -17),
    }
    for row in selected:
        sx = 100.0 * float(row["feature_contamination"])
        sy = 100.0 * float(row["essential_success_rate"])
        ax.scatter([sx], [sy], s=110, facecolor="none", edgecolor="#111827", linewidth=1.3, zorder=4)
        op_name = str(row["operating_point"])
        ax.annotate(
            op_labels.get(op_name, op_name.replace("_", " ")),
            (sx, sy),
            xytext=op_offsets.get(op_name, (8, 5)),
            textcoords="offset points",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.86},
            arrowprops={"arrowstyle": "-", "color": "#111827", "lw": 0.6, "alpha": 0.7},
        )
    baseline_labels = {
        "raw_all_pixels": "Raw",
        "groundingdino_box_static": "DINO-box",
        "oracle_gt_static": "Oracle GT",
    }
    baseline_offsets = {
        "raw_all_pixels": (9, 0),
        "groundingdino_box_static": (10, -4),
        "oracle_gt_static": (8, 9),
    }
    for name, marker, color in [
        ("raw_all_pixels", "o", "#6b7280"),
        ("groundingdino_box_static", "D", "#ea580c"),
        ("oracle_gt_static", "*", "#7c3aed"),
    ]:
        if name not in baselines:
            continue
        row = baselines[name]
        ax.scatter(
            [100.0 * float(row["feature_contamination"])],
            [100.0 * float(row["essential_success_rate"])],
            marker=marker,
            color=color,
            s=75,
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        ax.annotate(
            baseline_labels.get(name, name.replace("_", " ")),
            (100.0 * float(row["feature_contamination"]), 100.0 * float(row["essential_success_rate"])),
            xytext=baseline_offsets.get(name, (5, 3)),
            textcoords="offset points",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.82},
        )
    ax.set_xlabel("Feature contamination (%)")
    ax.set_ylabel("Essential-matrix success (%)")
    all_x = list(x)
    all_y = list(y)
    for name in ("raw_all_pixels", "groundingdino_box_static", "oracle_gt_static"):
        if name in baselines:
            all_x.append(100.0 * float(baselines[name]["feature_contamination"]))
            all_y.append(100.0 * float(baselines[name]["essential_success_rate"]))
    if all_x and all_y:
        x_lo = min(all_x)
        x_hi = max(all_x)
        y_lo = min(all_y)
        y_hi = max(all_y)
        ax.set_xlim(max(-0.5, x_lo - 0.08 * max(1.0, x_hi - x_lo)), x_hi + 0.18 * max(1.0, x_hi - x_lo))
        ax.set_ylim(max(0.0, y_lo - 0.10 * max(1.0, y_hi - y_lo)), min(100.0, y_hi + 0.10 * max(1.0, y_hi - y_lo)))
    ax.grid(True, alpha=0.22, linewidth=0.6)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Static threshold")

    ax = axes[1]
    thresholds = [row["threshold"] for row in rows]
    ax.plot(thresholds, [100.0 * row["match_contamination"] for row in rows], label="Match contamination", color="#059669", linewidth=1.8)
    ax.plot(thresholds, [100.0 * row["essential_success_rate"] for row in rows], label="E success", color="#2563eb", linewidth=1.8)
    if all(math.isfinite(float(row.get("static_retention", float("nan")))) for row in rows):
        ax.plot(thresholds, [100.0 * row["static_retention"] for row in rows], label="Static retention", color="#7c3aed", linewidth=1.5, linestyle="--")
    for row in selected:
        ax.axvline(float(row["threshold"]), color="#111827", alpha=0.18, linewidth=0.9)
    ax.set_xlabel("Static threshold")
    ax.set_ylabel("Rate (%)")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, fontsize=7, loc="best")

    fig.tight_layout(pad=0.5)
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orb-summary", required=True, help="ORB proxy summary_table.csv from a static-threshold sweep.")
    parser.add_argument("--static-threshold-curve", default="", help="Optional static-map threshold_curve.csv.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--high-success-min", type=float, default=0.83)
    parser.add_argument("--balanced-success-min", type=float, default=0.75)
    parser.add_argument("--clean-threshold", type=float, default=0.55)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds, baselines = _read_orb_thresholds(Path(args.orb_summary))
    static_path = Path(args.static_threshold_curve) if str(args.static_threshold_curve).strip() else None
    rows = _with_static(thresholds, _read_static_thresholds(static_path))
    selected = _select_operating_points(
        rows,
        high_success_min=float(args.high_success_min),
        balanced_success_min=float(args.balanced_success_min),
        clean_threshold=float(args.clean_threshold),
    )

    csv_path = out_dir / "operating_points.csv"
    tex_path = out_dir / "operating_points_table.tex"
    pdf_path = out_dir / "threshold_operating_points.pdf"
    png_path = out_dir / "threshold_operating_points.png"
    _write_selected_csv(csv_path, selected, baselines)
    _write_tex(tex_path, selected, baselines)
    _plot(pdf_path, png_path, rows, selected, baselines)

    summary = {
        "orb_summary": str(Path(args.orb_summary).resolve()),
        "static_threshold_curve": str(static_path.resolve()) if static_path is not None else None,
        "selection_config": {
            "high_success_min": float(args.high_success_min),
            "balanced_success_min": float(args.balanced_success_min),
            "clean_threshold": float(args.clean_threshold),
        },
        "selected": selected,
        "outputs": {
            "csv": str(csv_path.resolve()),
            "tex": str(tex_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "png": str(png_path.resolve()),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {csv_path}")
    print(f"Saved {tex_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    print(f"Saved {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
