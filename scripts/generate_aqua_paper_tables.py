#!/usr/bin/env python3
"""Generate paper-ready Aqua-D4RT tables and the R099 sensitivity figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_policy_csv(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            policy = str(row["policy"])
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"policy"}:
                    continue
                if key in {"selection_counts", "reason_counts"}:
                    parsed[key] = json.loads(value)
                else:
                    parsed[key] = float(value)
            rows[policy] = parsed
    return rows


def _finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def _range(values: list[float], *, scale: float = 1.0, digits: int = 2) -> str:
    valid = _finite(values)
    if not valid:
        return "--"
    lo = min(valid) * scale
    hi = max(valid) * scale
    if abs(lo - hi) < 10 ** (-(digits + 1)):
        return f"{lo:.{digits}f}"
    return f"{lo:.{digits}f}--{hi:.{digits}f}"


def _fmt_num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    value = float(value)
    if not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "--"
    value = float(value)
    if not math.isfinite(value):
        return "--"
    return f"{value * 100.0:.{digits}f}"


def _escape_latex(text: Any) -> str:
    s = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Saved {path}")


def _latex_table(
    *,
    env: str,
    caption: str,
    label: str,
    colspec: str,
    headers: list[str],
    rows: list[list[str]],
    note: str | None = None,
) -> str:
    lines = [
        f"\\begin{{{env}}}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(headers) + r" \\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if note:
        lines.extend(["\\vspace{0.25em}", f"\\footnotesize{{{note}}}"])
    lines.append(f"\\end{{{env}}}")
    lines.append("")
    return "\n".join(lines)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    clean_headers = [h.replace("\\%", "%").replace("\\uparrow", "↑").replace("\\downarrow", "↓") for h in headers]
    clean_rows = [[cell.replace("\\textbf{", "").replace("}", "") for cell in row] for row in rows]
    lines = ["| " + " | ".join(clean_headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in clean_rows)
    return "\n".join(lines) + "\n"


def _policy_label(policy: str) -> str:
    labels = {
        "raw": "Raw",
        "aqua_inpaint": "Aqua hard mask",
        "v3_hard": "v3 hard retention",
        "v3_pose_soft": "v3 pose-soft",
        "soft_raw_fallback_margin0.10_rawmin0.60": r"\textbf{R099 pose-soft + raw fallback}",
    }
    return labels.get(policy, policy)


def _selection_summary(row: dict[str, Any]) -> str:
    counts = row.get("selection_counts", {})
    if not counts:
        return "--"
    names = {
        "raw": "raw",
        "aqua_pose_soft_t0p73": "soft",
        "aqua_inpaint": "hard-mask",
        "aqua_learned_retain_t0p73_inpaint": "hard",
    }
    return ", ".join(f"{names.get(k, k)} {v}" for k, v in counts.items())


def _prefilter_rows(source: dict[str, Any], dino_box: Path, dino_sam: Path) -> list[dict[str, Any]]:
    rows = list(source["prefilter_baselines_static_rows"])
    for method, input_name, path in [
        ("GroundingDINO-box", "RGB + text prompt", dino_box),
        ("GroundingDINO+SAM-base", "Predicted box prompt", dino_sam),
    ]:
        data = _read_json(path)
        metrics = data["metrics_vs_transient"]
        rows.append(
            {
                "method": method,
                "input": input_name,
                "oracle": False,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "iou": float(data["iou_vs_transient"]),
                "pred_coverage": float(metrics["pred_positive_rate"]),
                "gt_coverage": float(metrics["gt_positive_rate"]),
                "boxes_per_frame": float(data.get("detection", {}).get("mean_boxes_per_frame", float("nan"))),
            }
        )
    order = {
        "Temporal RGB pseudo-mask": 0,
        "GroundingDINO-box": 1,
        "GroundingDINO+SAM-base": 2,
        "GT-box GrabCut": 3,
        "GT-box SAM-base": 4,
    }
    return sorted(rows, key=lambda row: order[str(row["method"])])


def _make_static_table(source: dict[str, Any]) -> tuple[str, str]:
    headers = [
        "Dataset",
        "Method",
        "Point contam. $\\downarrow$",
        "Static ret. $\\uparrow$",
        "Voxel contam. $\\downarrow$",
        "Voxel ret. $\\uparrow$",
    ]
    rows: list[list[str]] = []
    for dataset in source["static_query_map"]:
        first = True
        for item in dataset["rows"]:
            method = _escape_latex(item["method"])
            if str(item["method"]).startswith("Aqua"):
                method = r"\textbf{" + method + "}"
            rows.append(
                [
                    _escape_latex(dataset["dataset"]) if first else "",
                    method,
                    _fmt_pct(item["point_contamination"]),
                    _fmt_pct(item["static_retention"]),
                    _fmt_pct(item["voxel_contamination"]),
                    _fmt_pct(item["voxel_retention"]),
                ]
            )
            first = False
    tex = _latex_table(
        env="table*",
        caption=(
            "Static query-map contamination. Aqua keeps a high-confidence static query set "
            "without using ground-truth masks at inference."
        ),
        label="tab:aqua_static_query_map",
        colspec="llrrrr",
        headers=headers,
        rows=rows,
        note=(
            "Numbers are percentages. WebUOT labels are tracked-target bounding boxes, "
            "not complete fish-instance masks."
        ),
    )
    return tex, _markdown_table(headers, rows)


def _make_head_ablation_table(source: dict[str, Any]) -> tuple[str, str]:
    headers = ["Setting", "Dataset / model", "Dynamic F1", "Particle F1", "Point contam. $\\downarrow$", "Static ret. $\\uparrow$"]
    rows: list[list[str]] = []
    for item in source["head_ablation_training"]:
        label = _escape_latex(item["model"])
        if str(item["model"]).startswith("Full"):
            label = r"\textbf{" + label + "}"
        rows.append(
            [
                "Training",
                label,
                _fmt_num(item["dynamic_best_f1"], 3),
                _fmt_num(item["particle_best_f1"], 3),
                _fmt_pct(item["point_contamination"]),
                _fmt_pct(item["static_retention"]),
            ]
        )
    for item in source["head_ablation_inference"]:
        label = _escape_latex(f"{item['dataset']}: {item['variant']}")
        if str(item["variant"]).startswith("Full"):
            label = r"\textbf{" + label + "}"
        rows.append(
            [
                "Score term",
                label,
                "--",
                "--",
                _fmt_pct(item["point_contamination"]),
                _fmt_pct(item["static_retention"]),
            ]
        )
    tex = _latex_table(
        env="table*",
        caption="Dynamic and particle head ablations. Both transient heads are needed for the clean static-map result.",
        label="tab:aqua_head_ablation",
        colspec="llrrrr",
        headers=headers,
        rows=rows,
    )
    return tex, _markdown_table(headers, rows)


def _make_prefilter_table(source: dict[str, Any], dino_box: Path, dino_sam: Path) -> tuple[str, str]:
    headers = ["Method", "Input", "Oracle?", "Prec.", "Rec.", "F1", "IoU", "Pred. cov.", "Boxes/frame"]
    rows: list[list[str]] = []
    for item in _prefilter_rows(source, dino_box, dino_sam):
        boxes = item.get("boxes_per_frame")
        rows.append(
            [
                _escape_latex(item["method"]),
                _escape_latex(item["input"]),
                "yes" if item["oracle"] else "no",
                _fmt_num(item["precision"], 3),
                _fmt_num(item["recall"], 3),
                _fmt_num(item["f1"], 3),
                _fmt_num(item["iou"], 3),
                _fmt_pct(item["pred_coverage"]),
                _fmt_num(boxes, 2) if boxes is not None else "--",
            ]
        )
    tex = _latex_table(
        env="table*",
        caption=(
            "WebUOT fish30 2D prefilter baselines. Detector/SAM rows are non-oracle; "
            "GT-box rows use WebUOT boxes and are oracle-style references."
        ),
        label="tab:aqua_prefilter_baselines",
        colspec="llcrrrrrr",
        headers=headers,
        rows=rows,
        note=(
            "These are mask-only metrics under WebUOT tracked-box labels. "
            "They are not query-map or downstream SfM metrics."
        ),
    )
    return tex, _markdown_table(headers, rows)


def _make_policy_table(summary: dict[str, dict[str, Any]], *, caption: str, label: str) -> tuple[str, str]:
    headers = [
        "Method",
        "Pose succ. $\\uparrow$",
        "Input reg. $\\uparrow$",
        "ATE $\\downarrow$",
        "RPE $\\downarrow$",
        "Feat. contam. $\\downarrow$",
        "Match contam. $\\downarrow$",
        "Selection",
    ]
    policy_order = ["raw", "aqua_inpaint", "v3_hard", "v3_pose_soft", "soft_raw_fallback_margin0.10_rawmin0.60"]
    rows: list[list[str]] = []
    for policy in policy_order:
        if policy not in summary:
            continue
        item = summary[policy]
        rows.append(
            [
                _policy_label(policy),
                _fmt_pct(item["pose_eval_success"]),
                _fmt_pct(item["input_registration_rate"]),
                _fmt_num(item["ate_rmse"], 4),
                _fmt_num(item["rpe_trans_rmse"], 4),
                _fmt_pct(item["feature_contamination"]),
                _fmt_pct(item["match_contamination_mean"]),
                _escape_latex(_selection_summary(item)),
            ]
        )
    tex = _latex_table(
        env="table*",
        caption=caption,
        label=label,
        colspec="lrrrrrrl",
        headers=headers,
        rows=rows,
    )
    return tex, _markdown_table(headers, rows)


def _make_sensitivity_table(summary: dict[str, dict[str, Any]]) -> tuple[str, str]:
    fallback = {k: v for k, v in summary.items() if k.startswith("soft_raw_fallback")}
    default = summary["soft_raw_fallback_margin0.10_rawmin0.60"]
    raw_counts = [float(row.get("selection_counts", {}).get("raw", 0)) for row in fallback.values()]
    headers = [
        "Policy set",
        "Pose succ. $\\uparrow$",
        "Input reg. $\\uparrow$",
        "ATE $\\downarrow$",
        "RPE $\\downarrow$",
        "Feat. contam. $\\downarrow$",
        "Raw fallbacks",
    ]
    rows = [
        [
            "R100 grid, 12 policies",
            _range([row["pose_eval_success"] for row in fallback.values()], scale=100.0),
            _range([row["input_registration_rate"] for row in fallback.values()], scale=100.0),
            _range([row["ate_rmse"] for row in fallback.values()], digits=4),
            _range([row["rpe_trans_rmse"] for row in fallback.values()], digits=4),
            _range([row["feature_contamination"] for row in fallback.values()], scale=100.0),
            _range(raw_counts, digits=0),
        ],
        [
            r"\textbf{Default R099}",
            _fmt_pct(default["pose_eval_success"]),
            _fmt_pct(default["input_registration_rate"]),
            _fmt_num(default["ate_rmse"], 4),
            _fmt_num(default["rpe_trans_rmse"], 4),
            _fmt_pct(default["feature_contamination"]),
            str(int(default.get("selection_counts", {}).get("raw", 0))),
        ],
    ]
    tex = _latex_table(
        env="table",
        caption="R099 raw-fallback selector sensitivity over neighboring thresholds.",
        label="tab:aqua_r099_sensitivity",
        colspec="lrrrrrr",
        headers=headers,
        rows=rows,
    )
    return tex, _markdown_table(headers, rows)


def _parse_policy_grid(policy: str) -> tuple[float, float]:
    match = re.search(r"margin([0-9.]+)_rawmin([0-9.]+)$", policy)
    if not match:
        raise ValueError(f"Cannot parse fallback policy: {policy}")
    return float(match.group(2)), float(match.group(1))


def _plot_sensitivity(summary: dict[str, dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on local plotting env
        print(f"Skipping sensitivity figure; matplotlib unavailable: {exc}")
        return

    fallback = {k: v for k, v in summary.items() if k.startswith("soft_raw_fallback")}
    raw_mins = sorted({_parse_policy_grid(k)[0] for k in fallback})
    margins = sorted({_parse_policy_grid(k)[1] for k in fallback})
    metrics = [
        ("input_registration_rate", 100.0, "Input registration (%)", "{:.1f}"),
        ("ate_rmse", 1.0, "ATE RMSE", "{:.3f}"),
        ("feature_contamination", 100.0, "Feature contamination (%)", "{:.1f}"),
    ]

    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "DejaVu Serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.2), constrained_layout=True)
    for ax, (metric, scale, cbar_label, fmt) in zip(axes, metrics):
        matrix = np.full((len(raw_mins), len(margins)), np.nan, dtype=float)
        for policy, row in fallback.items():
            raw_min, margin = _parse_policy_grid(policy)
            i = raw_mins.index(raw_min)
            j = margins.index(margin)
            matrix[i, j] = float(row[metric]) * scale
        im = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(margins)), [f"{v:.2f}" for v in margins])
        ax.set_yticks(range(len(raw_mins)), [f"{v:.2f}" for v in raw_mins])
        ax.set_xlabel("Raw margin")
        if ax is axes[0]:
            ax.set_ylabel("Raw min. reg.")
        else:
            ax.set_ylabel("")
        finite = matrix[np.isfinite(matrix)]
        lo = float(np.min(finite)) if finite.size else 0.0
        hi = float(np.max(finite)) if finite.size else 1.0
        span = max(hi - lo, 1e-9)
        for i in range(len(raw_mins)):
            for j in range(len(margins)):
                value = matrix[i, j]
                norm = (float(value) - lo) / span
                text_color = "black" if norm > 0.62 else "white"
                ax.text(j, i, fmt.format(value), ha="center", va="center", color=text_color, fontsize=7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(cbar_label)
    pdf = output_dir / "fig_r099_sensitivity.pdf"
    png = output_dir / "fig_r099_sensitivity.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"Saved {pdf}")
    print(f"Saved {png}")


def _make_latex_includes(output_dir: Path) -> str:
    return "\n".join(
        [
            "% Auto-generated by scripts/generate_aqua_paper_tables.py",
            r"\input{figures/aqua_paper_tables/table_static_query_map.tex}",
            r"\input{figures/aqua_paper_tables/table_head_ablation.tex}",
            r"\input{figures/aqua_paper_tables/table_tank_stress4_r099.tex}",
            r"\input{figures/aqua_paper_tables/table_r099_sensitivity.tex}",
            r"\input{figures/aqua_paper_tables/table_allstress_r101.tex}",
            r"\input{figures/aqua_paper_tables/table_prefilter_baselines.tex}",
            "",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=\linewidth]{figures/aqua_paper_tables/fig_r099_sensitivity.pdf}",
            r"\caption{Sensitivity of the R099 raw-fallback selector over neighboring raw-registration thresholds and margins. The pose-evaluation success stays fixed at 90.28\% across all twelve settings; the figure shows the remaining registration, ATE, and contamination variation.}",
            r"\label{fig:aqua_r099_sensitivity}",
            r"\end{figure}",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(root / "figures/aqua_paper_tables"))
    parser.add_argument("--source-json", default=str(root / "figures/aqua_paper_tables/table_source_notes.json"))
    parser.add_argument("--r099-summary", default=str(root / "tmp/aqua_adaptive_v3_t073_full_stress4_selector_v3_rawfallback/summary.csv"))
    parser.add_argument("--r100-summary", default=str(root / "tmp/aqua_adaptive_v3_t073_full_stress4_selector_v3_sensitivity/summary.csv"))
    parser.add_argument("--r101-summary", default=str(root / "tmp/aqua_adaptive_v3_t073_allstress_seed42_selector_v3_rawfallback/summary.csv"))
    parser.add_argument("--dino-box-json", default=str(root / "tmp/aqua_prefilter_masks/webuot238_fish30_groundingdino_box_all30/aggregate_metrics.json"))
    parser.add_argument("--dino-sam-json", default=str(root / "tmp/aqua_prefilter_masks/webuot238_fish30_groundingdino_sam_all30/aggregate_metrics.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = _read_json(Path(args.source_json))
    r099 = _read_policy_csv(Path(args.r099_summary))
    r100 = _read_policy_csv(Path(args.r100_summary))
    r101 = _read_policy_csv(Path(args.r101_summary))

    artifacts: list[tuple[str, str]] = []
    static_tex, static_md = _make_static_table(source)
    artifacts.append(("table_static_query_map.tex", static_tex))
    head_tex, head_md = _make_head_ablation_table(source)
    artifacts.append(("table_head_ablation.tex", head_tex))
    prefilter_tex, prefilter_md = _make_prefilter_table(source, Path(args.dino_box_json), Path(args.dino_sam_json))
    artifacts.append(("table_prefilter_baselines.tex", prefilter_tex))
    r099_tex, r099_md = _make_policy_table(
        r099,
        caption=(
            "Expanded Tank stress4 downstream result. R099 is a self-diagnostic multi-pass selector: "
            "it runs raw and v3 pose-soft candidates, then falls back to raw only when raw registration is substantially higher."
        ),
        label="tab:aqua_tank_stress4_r099",
    )
    artifacts.append(("table_tank_stress4_r099.tex", r099_tex))
    sens_tex, sens_md = _make_sensitivity_table(r100)
    artifacts.append(("table_r099_sensitivity.tex", sens_tex))
    r101_tex, r101_md = _make_policy_table(
        r101,
        caption=(
            "All-stress seed42 sanity check over all eight Tank stress variants. "
            "This supports a contamination/registration Pareto but not a uniform pose win."
        ),
        label="tab:aqua_allstress_r101",
    )
    artifacts.append(("table_allstress_r101.tex", r101_tex))
    artifacts.append(("latex_includes.tex", _make_latex_includes(output_dir)))

    for filename, text in artifacts:
        _write(output_dir / filename, text)

    summary_md = "\n".join(
        [
            "# Aqua-D4RT Paper Tables",
            "",
            "Generated by `scripts/generate_aqua_paper_tables.py`.",
            "",
            "## Static Query-Map",
            "",
            static_md,
            "## Head Ablation",
            "",
            head_md,
            "## Tank Stress4 R099",
            "",
            r099_md,
            "## R099 Sensitivity",
            "",
            sens_md,
            "## All-Stress Seed42 R101",
            "",
            r101_md,
            "## Prefilter Baselines",
            "",
            prefilter_md,
            "## Source Notes",
            "",
            "\n".join(f"- {note}" for note in source.get("notes", [])),
            "",
        ]
    )
    _write(output_dir / "paper_tables_summary.md", summary_md)
    _plot_sensitivity(r100, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
