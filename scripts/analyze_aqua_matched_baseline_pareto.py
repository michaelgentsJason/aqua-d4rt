#!/usr/bin/env python3
"""Matched baseline/Pareto audit for Aqua-D4RT paper claims.

The script reuses existing per-clip JSON artifacts.  It does not rerun D4RT,
GroundingDINO, SAM, ORB, or pyCOLMAP.  The goal is to compare Aqua and
image-level prefilters under matched operating conditions:

* matched static-query retention;
* matched essential-matrix success;
* frontend-friendly Aqua points with high E-success constraints.

Outputs include CSV/LaTeX tables, bootstrap CIs over clips, and compact Pareto
figures for paper/reviewer discussion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    static_path: Path
    orb_path: Path
    external_variants: tuple[str, ...]
    has_full_aqua_orb_sweep: bool = True


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="fish30_all30",
        label="WebUOT fish30 all30",
        static_path=Path("tmp/aqua_static_map_contamination/webuot238_fish30_external_dino_sam_all30/per_clip_metrics.json"),
        orb_path=Path("tmp/aqua_downstream_slam_proxy/webuot238_fish30_external_dino_sam_all30/per_clip_metrics.json"),
        external_variants=("groundingdino_box_static", "groundingdino_sam_static"),
        has_full_aqua_orb_sweep=False,
    ),
    DatasetSpec(
        key="fish30_val6",
        label="WebUOT fish30 val6",
        static_path=Path("tmp/aqua_static_map_contamination/webuot238_fish30_external_dino_sam_val6/per_clip_metrics.json"),
        orb_path=Path("tmp/aqua_downstream_slam_proxy/webuot238_fish30_external_dino_sam_val6/per_clip_metrics.json"),
        external_variants=("groundingdino_box_static", "groundingdino_sam_static"),
        has_full_aqua_orb_sweep=False,
    ),
    DatasetSpec(
        key="dynamic100_all100",
        label="WebUOT dynamic100 all100",
        static_path=Path("tmp/aqua_static_map_contamination/webuot238_dynamic100_external_dino_box_all100/per_clip_metrics.json"),
        orb_path=Path("tmp/aqua_downstream_slam_proxy/webuot238_dynamic100_aqua_threshold_sweep_dino_box_all100/per_clip_metrics.json"),
        external_variants=("groundingdino_box_static",),
    ),
    DatasetSpec(
        key="dynamic100_new70",
        label="WebUOT dynamic100 new70",
        static_path=Path("tmp/aqua_static_map_contamination/webuot238_dynamic100_external_dino_box_new70/per_clip_metrics.json"),
        orb_path=Path("tmp/aqua_downstream_slam_proxy/webuot238_dynamic100_aqua_threshold_sweep_dino_box_new70/per_clip_metrics.json"),
        external_variants=("groundingdino_box_static",),
    ),
    DatasetSpec(
        key="webuot_all238",
        label="WebUOT all238 Aqua sweep",
        static_path=Path("tmp/aqua_static_map_contamination/webuot238_all238_aqua_all_20260624/per_clip_metrics.json"),
        orb_path=Path("tmp/aqua_downstream_slam_proxy/webuot238_all238_aqua_threshold_sweep_20260624/per_clip_metrics.json"),
        external_variants=(),
    ),
    DatasetSpec(
        key="aqualoc_harbor07",
        label="AQUALOC harbor07 injected sanity",
        static_path=Path("tmp/aqua_static_map_contamination/aqualoc_harbor07_sanity_20260624/per_clip_metrics.json"),
        orb_path=Path("tmp/aqua_downstream_slam_proxy/aqualoc_harbor07_sanity_20260624/per_clip_metrics.json"),
        external_variants=(),
    ),
)


STATIC_TO_ORB = {
    "all_d4rt_points": "raw_all_pixels",
    "temporal_rgb_prefilter_static": "temporal_rgb_static",
}
ORB_TO_STATIC = {value: key for key, value in STATIC_TO_ORB.items()}


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def _clip_name(item: dict[str, Any]) -> str:
    return str(item.get("clip_name") or Path(str(item.get("manifest", ""))).stem)


def _safe_rate(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) else 0.0


def _threshold_from_variant(name: str) -> float | None:
    match = re.search(r"ge_([0-9]+)p([0-9]+)", name)
    if not match:
        return None
    return float(f"{match.group(1)}.{match.group(2)}")


def _is_aqua_threshold(name: str) -> bool:
    return name.startswith("aqua_static_conf_ge_") and _threshold_from_variant(name) is not None


def _static_variant_for_orb(name: str) -> str:
    return ORB_TO_STATIC.get(name, name)


def _orb_variant_for_static(name: str) -> str:
    return STATIC_TO_ORB.get(name, name)


def _aggregate_static(per_clip: list[dict[str, Any]], variant: str) -> dict[str, float]:
    point_keys = [
        "total_valid_points",
        "total_static_points",
        "total_transient_points",
        "kept_points",
        "kept_static_points",
        "kept_transient_points",
    ]
    voxel_keys = [
        "total_voxels",
        "kept_voxels",
        "kept_contaminated_voxels",
        "static_support_voxels",
        "kept_static_support_voxels",
        "clean_static_only_voxels",
        "kept_clean_static_only_voxels",
    ]
    rows = [clip["variant_metrics"][variant] for clip in per_clip if variant in clip.get("variant_metrics", {})]
    if not rows:
        return {}
    sums = {key: float(sum(float(row.get(key, 0.0)) for row in rows)) for key in point_keys + voxel_keys}
    return {
        "clips_static": float(len(rows)),
        **sums,
        "query_contamination": _safe_rate(sums["kept_transient_points"], sums["kept_points"]),
        "static_retention": _safe_rate(sums["kept_static_points"], sums["total_static_points"]),
        "transient_rejection": 1.0 - _safe_rate(sums["kept_transient_points"], sums["total_transient_points"]),
        "voxel_contamination": _safe_rate(sums["kept_contaminated_voxels"], sums["kept_voxels"]),
        "voxel_static_retention": _safe_rate(sums["kept_static_support_voxels"], sums["static_support_voxels"]),
    }


def _aggregate_orb(per_clip: list[dict[str, Any]], variant: str) -> dict[str, float]:
    summaries = [
        clip["variants"][variant]["summary"]
        for clip in per_clip
        if variant in clip.get("variants", {})
    ]
    if not summaries:
        return {}
    total_features = float(sum(float(row.get("total_features", 0.0)) for row in summaries))
    total_matches = float(sum(float(row.get("total_matches", 0.0)) for row in summaries))
    total_pairs = float(sum(float(row.get("pairs", 0.0)) for row in summaries))
    feature_contam_num = sum(
        float(row.get("feature_contamination", 0.0)) * float(row.get("total_features", 0.0))
        for row in summaries
    )
    success_pairs = sum(
        float(row.get("essential_success_rate", 0.0)) * float(row.get("pairs", 0.0))
        for row in summaries
    )
    return {
        "clips_orb": float(len(summaries)),
        "total_features": total_features,
        "features_per_frame": _mean([float(row.get("features_per_frame_mean", 0.0)) for row in summaries]),
        "feature_contamination": _safe_rate(feature_contam_num, total_features),
        "total_matches": total_matches,
        "matches_per_pair": _mean([float(row.get("matches_per_pair_mean", 0.0)) for row in summaries]),
        "match_contamination": _mean([float(row.get("match_contamination_mean", 0.0)) for row in summaries]),
        "essential_success": _safe_rate(success_pairs, total_pairs),
        "inliers_per_pair": _mean([float(row.get("essential_inliers_per_pair_mean", 0.0)) for row in summaries]),
        "inlier_rate": _mean([float(row.get("essential_inlier_rate_mean", 0.0)) for row in summaries]),
    }


def _mean(values: list[float]) -> float:
    return float(sum(values)) / float(len(values)) if values else 0.0


def _variants_static(per_clip: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for clip in per_clip:
        for variant in clip.get("variant_metrics", {}):
            if variant not in out:
                out.append(variant)
    return out


def _variants_orb(per_clip: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for clip in per_clip:
        for variant in clip.get("variants", {}):
            if variant not in out:
                out.append(variant)
    return out


def _metric_row(
    *,
    dataset: str,
    role: str,
    variant: str,
    static_metrics: dict[str, float] | None,
    orb_metrics: dict[str, float] | None,
    note: str = "",
) -> dict[str, Any]:
    static_metrics = static_metrics or {}
    orb_metrics = orb_metrics or {}
    threshold = _threshold_from_variant(variant)
    return {
        "dataset": dataset,
        "role": role,
        "variant": variant,
        "method": _pretty_variant(variant),
        "threshold": "" if threshold is None else f"{threshold:.2f}",
        "query_contamination": static_metrics.get("query_contamination", math.nan),
        "static_retention": static_metrics.get("static_retention", math.nan),
        "voxel_contamination": static_metrics.get("voxel_contamination", math.nan),
        "feature_contamination": orb_metrics.get("feature_contamination", math.nan),
        "match_contamination": orb_metrics.get("match_contamination", math.nan),
        "essential_success": orb_metrics.get("essential_success", math.nan),
        "features_per_frame": orb_metrics.get("features_per_frame", math.nan),
        "matches_per_pair": orb_metrics.get("matches_per_pair", math.nan),
        "note": note,
    }


def _combined_metrics(
    static_clips: list[dict[str, Any]],
    orb_clips: list[dict[str, Any]],
    variant: str,
) -> tuple[dict[str, float], dict[str, float]]:
    static_name = _static_variant_for_orb(variant)
    orb_name = _orb_variant_for_static(variant)
    return _aggregate_static(static_clips, static_name), _aggregate_orb(orb_clips, orb_name)


def _nearest_by_metric(rows: list[dict[str, Any]], metric: str, target: float) -> dict[str, Any] | None:
    valid = [row for row in rows if math.isfinite(float(row.get(metric, math.nan)))]
    if not valid:
        return None
    return min(valid, key=lambda row: abs(float(row[metric]) - float(target)))


def _select_min_contam_with_e(rows: list[dict[str, Any]], min_e: float) -> tuple[dict[str, Any] | None, bool]:
    valid = [
        row for row in rows
        if math.isfinite(float(row.get("essential_success", math.nan)))
        and float(row["essential_success"]) >= min_e
    ]
    if valid:
        return (
            min(
                valid,
                key=lambda row: (
                    float(row.get("feature_contamination", math.inf)),
                    float(row.get("match_contamination", math.inf)),
                ),
            ),
            True,
        )
    valid = [row for row in rows if math.isfinite(float(row.get("essential_success", math.nan)))]
    return (max(valid, key=lambda row: float(row["essential_success"])), False) if valid else (None, False)


def _pretty_variant(name: str) -> str:
    mapping = {
        "all_d4rt_points": "Raw D4RT",
        "raw_all_pixels": "Raw D4RT",
        "temporal_rgb_prefilter_static": "Temporal RGB",
        "temporal_rgb_static": "Temporal RGB",
        "aqua_pred_transient_filter": "Aqua transient filter",
        "aqua_static_conf_ge_0p150": "Aqua static >= 0.15",
        "aqua_static_conf_ge_0p200": "Aqua static >= 0.20",
        "aqua_static_conf_ge_0p550": "Aqua static >= 0.55",
        "groundingdino_box_static": "GroundingDINO-box",
        "groundingdino_sam_static": "GroundingDINO+SAM",
        "oracle_gt_static": "Oracle GT static",
    }
    if name in mapping:
        return mapping[name]
    threshold = _threshold_from_variant(name)
    if threshold is not None:
        return f"Aqua static >= {threshold:.2f}"
    return name.replace("_", " ")


def _plot_label(name: str) -> str:
    threshold = _threshold_from_variant(name)
    if threshold is not None:
        return f"Aqua .{int(round(threshold * 100)):02d}"
    mapping = {
        "all_d4rt_points": "Raw",
        "raw_all_pixels": "Raw",
        "temporal_rgb_prefilter_static": "Temp.",
        "temporal_rgb_static": "Temp.",
        "groundingdino_box_static": "DINO-box",
        "groundingdino_sam_static": "DINO+SAM",
        "oracle_gt_static": "Oracle",
    }
    return mapping.get(name, _pretty_variant(name))


def _label_offset(name: str) -> tuple[int, int, str]:
    threshold = _threshold_from_variant(name)
    if threshold is not None:
        if abs(threshold - 0.15) < 1e-6:
            return 5, 5, "left"
        if abs(threshold - 0.55) < 1e-6:
            return 5, -12, "left"
        return 4, 4, "left"
    mapping = {
        "all_d4rt_points": (-8, -14, "right"),
        "raw_all_pixels": (-8, -14, "right"),
        "temporal_rgb_prefilter_static": (-8, 7, "right"),
        "temporal_rgb_static": (-8, 7, "right"),
        "oracle_gt_static": (5, -14, "left"),
        "groundingdino_box_static": (5, 4, "left"),
        "groundingdino_sam_static": (5, -12, "left"),
    }
    return mapping.get(name, (4, 4, "left"))


def _pct(value: Any) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value_f):
        return ""
    return f"{100.0 * value_f:.2f}"


def _num(value: Any) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value_f):
        return ""
    return f"{value_f:.1f}"


def _fmt_table_pct(value: Any) -> str:
    out = _pct(value)
    return "--" if not out else out


def _fmt_threshold(value: Any) -> str:
    return str(value) if str(value).strip() else "--"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "comparison",
        "baseline_variant",
        "baseline_method",
        "aqua_variant",
        "aqua_method",
        "aqua_threshold",
        "match_metric",
        "baseline_query_contamination_pct",
        "baseline_static_retention_pct",
        "baseline_feature_contamination_pct",
        "baseline_match_contamination_pct",
        "baseline_essential_success_pct",
        "aqua_query_contamination_pct",
        "aqua_static_retention_pct",
        "aqua_feature_contamination_pct",
        "aqua_match_contamination_pct",
        "aqua_essential_success_pct",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_point_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "role",
        "variant",
        "method",
        "threshold",
        "query_contamination_pct",
        "static_retention_pct",
        "voxel_contamination_pct",
        "feature_contamination_pct",
        "match_contamination_pct",
        "essential_success_pct",
        "features_per_frame",
        "matches_per_pair",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "role": row["role"],
                    "variant": row["variant"],
                    "method": row["method"],
                    "threshold": row["threshold"],
                    "query_contamination_pct": _pct(row["query_contamination"]),
                    "static_retention_pct": _pct(row["static_retention"]),
                    "voxel_contamination_pct": _pct(row["voxel_contamination"]),
                    "feature_contamination_pct": _pct(row["feature_contamination"]),
                    "match_contamination_pct": _pct(row["match_contamination"]),
                    "essential_success_pct": _pct(row["essential_success"]),
                    "features_per_frame": _num(row["features_per_frame"]),
                    "matches_per_pair": _num(row["matches_per_pair"]),
                    "note": row["note"],
                }
            )


def _write_matched_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lllrrrrrrrr}",
        "\\toprule",
        "Dataset & Match & Baseline & B-Q & B-R & B-E & Aqua $\\tau_s$ & A-Q & A-R & A-E & Note \\\\",
        "\\midrule",
    ]
    for row in rows:
        note = str(row["note"]).replace("%", "\\%")
        lines.append(
            f"{row['dataset']} & {row['comparison']} & {row['baseline_method']} & "
            f"{row['baseline_query_contamination_pct']} & {row['baseline_static_retention_pct']} & "
            f"{row['baseline_essential_success_pct']} & {row['aqua_threshold']} & "
            f"{row['aqua_query_contamination_pct']} & {row['aqua_static_retention_pct']} & "
            f"{row['aqua_essential_success_pct']} & {note} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latex_includes(path: Path) -> None:
    lines = [
        "% Auto-generated by scripts/analyze_aqua_matched_baseline_pareto.py",
        "",
        "\\begin{figure}[t]",
        "    \\centering",
        "    \\includegraphics[width=0.95\\linewidth]{figures/aqua_matched_baselines_20260626/pareto_matched_baselines.pdf}",
        "    \\caption{Matched static-map and ORB-front-end Pareto audit. Aqua-D4RT provides a thresholded query-level operating curve, while detector/SAM prefilters act as image-level baselines. On WebUOT fish30 bbox labels, GroundingDINO-box is a strong low-contamination endpoint. On WebUOT dynamic100, it over-masks static structure and collapses essential-matrix success, whereas Aqua preserves a usable contamination/retention trade-off.}",
        "    \\label{fig:aqua_matched_pareto}",
        "\\end{figure}",
        "",
        "\\begin{figure}[t]",
        "    \\centering",
        "    \\includegraphics[width=0.95\\linewidth]{figures/aqua_matched_baselines_20260626/pareto_external_sanity.pdf}",
        "    \\caption{Held-out dynamic100 and AQUALOC external-background sanity Pareto curves. These plots are used as robustness and boundary evidence, not as broad underwater SLAM SOTA claims.}",
        "    \\label{fig:aqua_external_sanity_pareto}",
        "\\end{figure}",
        "",
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Matched baseline audit. B-Q/B-R/B-E denote baseline query contamination, static retention, and essential-matrix success; A-Q/A-R/A-E denote the matched Aqua operating point.}",
        "    \\label{tab:aqua_matched_baselines}",
        "    \\input{figures/aqua_matched_baselines_20260626/matched_baseline_table.tex}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _bootstrap_ci(
    static_map: dict[str, dict[str, Any]],
    orb_map: dict[str, dict[str, Any]],
    names: list[str],
    variant: str,
    *,
    num_samples: int,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    rng = random.Random(seed)
    static_variant = _static_variant_for_orb(variant)
    orb_variant = _orb_variant_for_static(variant)
    values: dict[str, list[float]] = {
        "query_contamination": [],
        "static_retention": [],
        "voxel_contamination": [],
        "feature_contamination": [],
        "match_contamination": [],
        "essential_success": [],
    }
    if not names:
        return {key: (math.nan, math.nan, math.nan) for key in values}
    for _ in range(num_samples):
        sample_names = [rng.choice(names) for _ in names]
        static_sample = [static_map[name] for name in sample_names if name in static_map]
        orb_sample = [orb_map[name] for name in sample_names if name in orb_map]
        static_metrics = _aggregate_static(static_sample, static_variant)
        orb_metrics = _aggregate_orb(orb_sample, orb_variant)
        if static_metrics:
            values["query_contamination"].append(static_metrics["query_contamination"])
            values["static_retention"].append(static_metrics["static_retention"])
            values["voxel_contamination"].append(static_metrics["voxel_contamination"])
        if orb_metrics:
            values["feature_contamination"].append(orb_metrics["feature_contamination"])
            values["match_contamination"].append(orb_metrics["match_contamination"])
            values["essential_success"].append(orb_metrics["essential_success"])
    out: dict[str, tuple[float, float, float]] = {}
    for key, vals in values.items():
        out[key] = _ci(vals)
    return out


def _ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    vals = sorted(float(v) for v in values)
    n = len(vals)
    lo = vals[max(0, int(math.floor(0.025 * (n - 1))))]
    hi = vals[min(n - 1, int(math.ceil(0.975 * (n - 1))))]
    return _mean(vals), lo, hi


def _write_ci_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "variant",
        "method",
        "metric",
        "mean_pct",
        "ci_low_pct",
        "ci_high_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _style_variant(name: str) -> tuple[str, str, float]:
    low = name.lower()
    if "raw" in low or "d4rt" in low or name == "all_d4rt_points":
        return "#6b7280", "o", 58.0
    if "oracle" in low:
        return "#7c3aed", "*", 90.0
    if "sam" in low:
        return "#db2777", "s", 62.0
    if "dino" in low or "grounding" in low:
        return "#ea580c", "D", 62.0
    if "temporal" in low:
        return "#2563eb", "^", 58.0
    return "#059669", "o", 38.0


def _plot_static_panel(ax: plt.Axes, rows: list[dict[str, Any]], *, label: str) -> None:
    aqua = [row for row in rows if _is_aqua_threshold(str(row["variant"]))]
    aqua.sort(key=lambda row: float(row["threshold"] or 0.0))
    if aqua:
        ax.plot(
            [100.0 * float(row["query_contamination"]) for row in aqua],
            [100.0 * float(row["static_retention"]) for row in aqua],
            color="#059669",
            linewidth=1.6,
            alpha=0.9,
            label="Aqua sweep",
        )
    for row in rows:
        if _is_aqua_threshold(str(row["variant"])) and str(row["threshold"]) not in {"0.15", "0.55"}:
            continue
        color, marker, size = _style_variant(str(row["variant"]))
        x = 100.0 * float(row["query_contamination"])
        y = 100.0 * float(row["static_retention"])
        ax.scatter([x], [y], color=color, marker=marker, s=size, edgecolor="white", linewidth=0.7, zorder=3)
        if not _is_aqua_threshold(str(row["variant"])) or str(row["threshold"]) in {"0.15", "0.55"}:
            dx, dy, ha = _label_offset(str(row["variant"]))
            ax.annotate(
                _plot_label(str(row["variant"])),
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7,
                ha=ha,
            )
    ax.set_xlabel("Query contamination (%)")
    ax.set_ylabel("Static retention (%)")
    finite_x = [100.0 * float(row["query_contamination"]) for row in rows if math.isfinite(float(row.get("query_contamination", math.nan)))]
    if finite_x:
        ax.set_xlim(left=min(-0.5, min(finite_x) - 0.5), right=max(finite_x) * 1.10 + 0.8)
    ax.set_ylim(-4.0, 105.0)
    ax.grid(True, alpha=0.22, linewidth=0.55)
    ax.text(0.02, 0.96, label, transform=ax.transAxes, ha="left", va="top", fontsize=8)


def _plot_orb_panel(ax: plt.Axes, rows: list[dict[str, Any]], *, label: str) -> None:
    aqua = [row for row in rows if _is_aqua_threshold(str(row["variant"]))]
    aqua.sort(key=lambda row: float(row["threshold"] or 0.0))
    if aqua:
        ax.plot(
            [100.0 * float(row["feature_contamination"]) for row in aqua],
            [100.0 * float(row["essential_success"]) for row in aqua],
            color="#059669",
            linewidth=1.6,
            alpha=0.9,
            label="Aqua sweep",
        )
    for row in rows:
        if _is_aqua_threshold(str(row["variant"])) and str(row["threshold"]) not in {"0.15", "0.55"}:
            continue
        if not math.isfinite(float(row.get("feature_contamination", math.nan))):
            continue
        color, marker, size = _style_variant(str(row["variant"]))
        x = 100.0 * float(row["feature_contamination"])
        y = 100.0 * float(row["essential_success"])
        ax.scatter([x], [y], color=color, marker=marker, s=size, edgecolor="white", linewidth=0.7, zorder=3)
        if not _is_aqua_threshold(str(row["variant"])) or str(row["threshold"]) in {"0.15", "0.55"}:
            dx, dy, ha = _label_offset(str(row["variant"]))
            ax.annotate(
                _plot_label(str(row["variant"])),
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7,
                ha=ha,
            )
    ax.set_xlabel("Feature contamination (%)")
    ax.set_ylabel("E-matrix success (%)")
    finite_x = [100.0 * float(row["feature_contamination"]) for row in rows if math.isfinite(float(row.get("feature_contamination", math.nan)))]
    finite_y = [100.0 * float(row["essential_success"]) for row in rows if math.isfinite(float(row.get("essential_success", math.nan)))]
    if finite_x:
        ax.set_xlim(left=min(-0.5, min(finite_x) - 0.5), right=max(finite_x) * 1.10 + 0.8)
    if finite_y:
        ax.set_ylim(max(-3.0, min(finite_y) - 8.0), min(105.0, max(finite_y) * 1.08 + 2.0))
    ax.grid(True, alpha=0.22, linewidth=0.55)
    ax.text(0.02, 0.96, label, transform=ax.transAxes, ha="left", va="top", fontsize=8)


def _plot_figures(out_dir: Path, dataset_rows: dict[str, list[dict[str, Any]]]) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "font.family": "DejaVu Serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "figure.dpi": 180,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.2))
    _plot_static_panel(axes[0, 0], dataset_rows["fish30_all30"], label="(a) WebUOT fish30 static map")
    _plot_static_panel(axes[0, 1], dataset_rows["dynamic100_all100"], label="(b) Dynamic100 static map")
    _plot_orb_panel(axes[1, 0], dataset_rows["dynamic100_all100"], label="(c) Dynamic100 ORB front-end")
    _plot_orb_panel(axes[1, 1], dataset_rows["webuot_all238"], label="(d) WebUOT all238 ORB front-end")
    fig.tight_layout(pad=0.85)
    fig.savefig(out_dir / "pareto_matched_baselines.png", bbox_inches="tight")
    fig.savefig(out_dir / "pareto_matched_baselines.pdf", bbox_inches="tight")
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 2, figsize=(7.3, 2.7))
    _plot_orb_panel(axes2[0], dataset_rows["dynamic100_new70"], label="(a) Held-out dynamic100 new70")
    _plot_orb_panel(axes2[1], dataset_rows["aqualoc_harbor07"], label="(b) AQUALOC injected sanity")
    fig2.tight_layout(pad=0.85)
    fig2.savefig(out_dir / "pareto_external_sanity.png", bbox_inches="tight")
    fig2.savefig(out_dir / "pareto_external_sanity.pdf", bbox_inches="tight")
    plt.close(fig2)


def _build_dataset_rows(
    spec: DatasetSpec,
    static_clips: list[dict[str, Any]],
    orb_clips: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_variants = ["all_d4rt_points", "temporal_rgb_prefilter_static"]
    aqua_variants = [v for v in _variants_static(static_clips) if _is_aqua_threshold(v)]
    aqua_variants.sort(key=lambda name: _threshold_from_variant(name) or 0.0)
    selected = base_variants + aqua_variants + list(spec.external_variants) + ["oracle_gt_static"]
    seen: set[str] = set()
    for variant in selected:
        if variant in seen:
            continue
        seen.add(variant)
        static_metrics, orb_metrics = _combined_metrics(static_clips, orb_clips, variant)
        if not static_metrics and not orb_metrics:
            continue
        role = "aqua_threshold" if _is_aqua_threshold(variant) else "baseline"
        rows.append(
            _metric_row(
                dataset=spec.label,
                role=role,
                variant=variant,
                static_metrics=static_metrics,
                orb_metrics=orb_metrics,
            )
        )
    return rows


def _matched_rows(spec: DatasetSpec, point_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    aqua_static = [
        row for row in point_rows
        if _is_aqua_threshold(str(row["variant"])) and math.isfinite(float(row.get("static_retention", math.nan)))
    ]
    aqua_orb = [
        row for row in point_rows
        if _is_aqua_threshold(str(row["variant"])) and math.isfinite(float(row.get("essential_success", math.nan)))
    ]
    for baseline_variant in spec.external_variants:
        baseline = next((row for row in point_rows if row["variant"] == baseline_variant), None)
        if baseline is None:
            continue
        target_retention = float(baseline.get("static_retention", math.nan))
        matched = _nearest_by_metric(aqua_static, "static_retention", target_retention)
        if matched is not None:
            out.append(_format_match_row(spec, "retention_matched", baseline, matched, "static retention"))
        target_e = float(baseline.get("essential_success", math.nan))
        matched_e = _nearest_by_metric(aqua_orb, "essential_success", target_e)
        if matched_e is not None:
            note = "limited Aqua ORB sweep" if not spec.has_full_aqua_orb_sweep else "matched E success"
            out.append(_format_match_row(spec, "e_success_matched", baseline, matched_e, note))

    target, target_met = _select_min_contam_with_e(aqua_orb, 0.80)
    raw = next((row for row in point_rows if row["variant"] == "all_d4rt_points"), None)
    if target is not None and raw is not None:
        if target_met:
            out.append(
                _format_match_row(
                    spec,
                    "frontend_target_E80",
                    raw,
                    target,
                    "Aqua min feature contamination with E>=80%",
                )
            )
        else:
            out.append(
                _format_match_row(
                    spec,
                    "best_available_frontend",
                    raw,
                    target,
                    "No Aqua point reaches E>=80%; selected highest-E Aqua point",
                )
            )
    return out


def _format_match_row(
    spec: DatasetSpec,
    comparison: str,
    baseline: dict[str, Any],
    aqua: dict[str, Any],
    note: str,
) -> dict[str, str]:
    return {
        "dataset": spec.label,
        "comparison": comparison,
        "baseline_variant": str(baseline["variant"]),
        "baseline_method": str(baseline["method"]),
        "aqua_variant": str(aqua["variant"]),
        "aqua_method": str(aqua["method"]),
        "aqua_threshold": _fmt_threshold(aqua["threshold"]),
        "match_metric": note,
        "baseline_query_contamination_pct": _fmt_table_pct(baseline.get("query_contamination", math.nan)),
        "baseline_static_retention_pct": _fmt_table_pct(baseline.get("static_retention", math.nan)),
        "baseline_feature_contamination_pct": _fmt_table_pct(baseline.get("feature_contamination", math.nan)),
        "baseline_match_contamination_pct": _fmt_table_pct(baseline.get("match_contamination", math.nan)),
        "baseline_essential_success_pct": _fmt_table_pct(baseline.get("essential_success", math.nan)),
        "aqua_query_contamination_pct": _fmt_table_pct(aqua.get("query_contamination", math.nan)),
        "aqua_static_retention_pct": _fmt_table_pct(aqua.get("static_retention", math.nan)),
        "aqua_feature_contamination_pct": _fmt_table_pct(aqua.get("feature_contamination", math.nan)),
        "aqua_match_contamination_pct": _fmt_table_pct(aqua.get("match_contamination", math.nan)),
        "aqua_essential_success_pct": _fmt_table_pct(aqua.get("essential_success", math.nan)),
        "note": note,
    }


def _bootstrap_rows(
    spec: DatasetSpec,
    static_clips: list[dict[str, Any]],
    orb_clips: list[dict[str, Any]],
    point_rows: list[dict[str, Any]],
    *,
    num_samples: int,
    seed: int,
) -> list[dict[str, str]]:
    static_map = {_clip_name(item): item for item in static_clips}
    orb_map = {_clip_name(item): item for item in orb_clips}
    names = sorted(set(static_map) & set(orb_map))
    selected_variants: list[str] = []
    for variant in (
        "all_d4rt_points",
        "temporal_rgb_prefilter_static",
        *spec.external_variants,
        "oracle_gt_static",
    ):
        if any(row["variant"] == variant for row in point_rows):
            selected_variants.append(variant)
    for row in point_rows:
        variant = str(row["variant"])
        if _is_aqua_threshold(variant) and variant not in selected_variants:
            selected_variants.append(variant)
    out: list[dict[str, str]] = []
    for variant in selected_variants:
        cis = _bootstrap_ci(
            static_map,
            orb_map,
            names,
            variant,
            num_samples=num_samples,
            seed=seed + _stable_hash(f"{spec.key}:{variant}") % 100000,
        )
        for metric, (mean, lo, hi) in cis.items():
            if not math.isfinite(mean):
                continue
            out.append(
                {
                    "dataset": spec.label,
                    "variant": variant,
                    "method": _pretty_variant(variant),
                    "metric": metric,
                    "mean_pct": _pct(mean),
                    "ci_low_pct": _pct(lo),
                    "ci_high_pct": _pct(hi),
                }
            )
    return out


def _stable_hash(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) % (2**32)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/aqua_matched_baselines_20260626")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260626)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_point_rows: list[dict[str, Any]] = []
    all_match_rows: list[dict[str, str]] = []
    all_ci_rows: list[dict[str, str]] = []
    dataset_rows: dict[str, list[dict[str, Any]]] = {}
    sources: list[dict[str, str]] = []

    for spec in DATASETS:
        if not spec.static_path.exists() or not spec.orb_path.exists():
            print(f"[skip] {spec.label}: missing static/orb JSON")
            continue
        static_clips = _load_json(spec.static_path)
        orb_clips = _load_json(spec.orb_path)
        rows = _build_dataset_rows(spec, static_clips, orb_clips)
        dataset_rows[spec.key] = rows
        all_point_rows.extend(rows)
        all_match_rows.extend(_matched_rows(spec, rows))
        all_ci_rows.extend(
            _bootstrap_rows(
                spec,
                static_clips,
                orb_clips,
                rows,
                num_samples=args.bootstrap_samples,
                seed=args.seed,
            )
        )
        sources.append(
            {
                "dataset": spec.label,
                "static_path": str(spec.static_path.resolve()),
                "orb_path": str(spec.orb_path.resolve()),
            }
        )

    _write_point_csv(out_dir / "operating_point_table.csv", all_point_rows)
    _write_csv(out_dir / "matched_baseline_table.csv", all_match_rows)
    _write_matched_tex(out_dir / "matched_baseline_table.tex", all_match_rows)
    _write_latex_includes(out_dir / "latex_includes.tex")
    _write_ci_csv(out_dir / "bootstrap_ci_table.csv", all_ci_rows)

    required_for_plot = {"fish30_all30", "dynamic100_all100", "dynamic100_new70", "webuot_all238", "aqualoc_harbor07"}
    if required_for_plot.issubset(dataset_rows):
        _plot_figures(out_dir, dataset_rows)

    summary = {
        "sources": sources,
        "outputs": {
            "operating_points": str((out_dir / "operating_point_table.csv").resolve()),
            "matched_baselines": str((out_dir / "matched_baseline_table.csv").resolve()),
            "matched_baselines_tex": str((out_dir / "matched_baseline_table.tex").resolve()),
            "latex_includes": str((out_dir / "latex_includes.tex").resolve()),
            "bootstrap_ci": str((out_dir / "bootstrap_ci_table.csv").resolve()),
            "pareto_png": str((out_dir / "pareto_matched_baselines.png").resolve()),
            "pareto_pdf": str((out_dir / "pareto_matched_baselines.pdf").resolve()),
            "external_sanity_png": str((out_dir / "pareto_external_sanity.png").resolve()),
            "external_sanity_pdf": str((out_dir / "pareto_external_sanity.pdf").resolve()),
        },
        "claim_gate": [
            "On WebUOT fish30 bbox masks, GroundingDINO-box is a stronger image-level prefilter than Aqua at bbox-oriented static/ORB metrics.",
            "On WebUOT dynamic100, GroundingDINO-box is low-contamination but over-masks static structure and collapses E-matrix success; Aqua provides a better frontend retention/success trade-off.",
            "Use these tables as a fairness audit, not as a universal SOTA claim.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved {out_dir / 'operating_point_table.csv'}")
    print(f"Saved {out_dir / 'matched_baseline_table.csv'}")
    print(f"Saved {out_dir / 'bootstrap_ci_table.csv'}")
    print(f"Saved {out_dir / 'pareto_matched_baselines.pdf'}")
    print(f"Saved {out_dir / 'pareto_external_sanity.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
