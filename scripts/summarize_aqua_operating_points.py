#!/usr/bin/env python3
"""Combine Aqua operating-point CSV files into one paper table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_inputs(values: list[str] | None) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Input must be SPLIT=CSV, got {value!r}")
        name, path = value.split("=", 1)
        out.append((name.strip(), Path(path.strip())))
    if not out:
        raise ValueError("Provide at least one --input SPLIT=CSV.")
    return out


def _parse_static_summaries(values: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Static summary must be SPLIT=CSV, got {value!r}")
        name, path = value.split("=", 1)
        out[name.strip()] = Path(path.strip())
    return out


def _read_static_summary(path: Path) -> dict[str, dict[str, str]]:
    return {row.get("variant", ""): row for row in _read_rows(path)}


def _keep_row(row: dict[str, str]) -> bool:
    return row.get("operating_point", "") in {
        "raw_all_pixels",
        "registration_first",
        "balanced",
        "clean_map_default",
        "groundingdino_box_static",
        "oracle_gt_static",
    }


def _method_label(name: str) -> str:
    return {
        "raw_all_pixels": "Raw",
        "registration_first": "Aqua reg-first",
        "balanced": "Aqua balanced",
        "clean_map_default": "Aqua clean-map",
        "groundingdino_box_static": "DINO-box",
        "oracle_gt_static": "Oracle GT",
    }.get(name, name)


def _fmt(value: str) -> str:
    return value if str(value).strip() else "--"


def _pct_str(value: str) -> str:
    if not str(value).strip():
        return ""
    try:
        return f"{100.0 * float(value):.2f}"
    except ValueError:
        return ""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "method",
        "threshold",
        "query_contamination_pct",
        "static_retention_pct",
        "feature_contamination_pct",
        "match_contamination_pct",
        "essential_success_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Split & Method & $\\tau_s$ & Query contam. & Static ret. & Feature contam. & Match contam. & E success \\\\",
        "\\midrule",
    ]
    last_split = None
    for row in rows:
        split = str(row["split"]) if row["split"] != last_split else ""
        last_split = row["split"]
        lines.append(
            f"{split} & {row['method']} & {_fmt(row['threshold'])} & "
            f"{_fmt(row['query_contamination_pct'])} & {_fmt(row['static_retention_pct'])} & "
            f"{_fmt(row['feature_contamination_pct'])} & {_fmt(row['match_contamination_pct'])} & "
            f"{_fmt(row['essential_success_pct'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=None, metavar="SPLIT=CSV")
    parser.add_argument(
        "--static-summary",
        action="append",
        default=None,
        metavar="SPLIT=CSV",
        help="Optional static-map summary_table.csv used to fill baseline query contamination/retention.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    order = {
        "raw_all_pixels": 0,
        "registration_first": 1,
        "balanced": 2,
        "clean_map_default": 3,
        "groundingdino_box_static": 4,
        "oracle_gt_static": 5,
    }
    sources = _parse_inputs(args.input)
    static_sources = _parse_static_summaries(args.static_summary)
    static_rows_by_split = {
        split: _read_static_summary(path) for split, path in static_sources.items() if path.exists()
    }
    static_variant = {
        "raw_all_pixels": "all_d4rt_points",
        "groundingdino_box_static": "groundingdino_box_static",
        "oracle_gt_static": "oracle_gt_static",
    }
    for split, path in sources:
        split_rows = [row for row in _read_rows(path) if _keep_row(row)]
        split_rows.sort(key=lambda row: order.get(row.get("operating_point", ""), 99))
        for row in split_rows:
            static_row = static_rows_by_split.get(split, {}).get(static_variant.get(row["operating_point"], ""))
            query_contam = row.get("query_contamination_pct", "")
            static_ret = row.get("static_retention_pct", "")
            if static_row:
                query_contam = query_contam or _pct_str(static_row.get("point_contamination", ""))
                static_ret = static_ret or _pct_str(static_row.get("point_static_retention", ""))
            rows.append(
                {
                    "split": split,
                    "method": _method_label(row["operating_point"]),
                    "threshold": row.get("threshold", ""),
                    "query_contamination_pct": query_contam,
                    "static_retention_pct": static_ret,
                    "feature_contamination_pct": row.get("feature_contamination_pct", ""),
                    "match_contamination_pct": row.get("match_contamination_pct", ""),
                    "essential_success_pct": row.get("essential_success_pct", ""),
                }
            )

    csv_path = out_dir / "combined_operating_points.csv"
    tex_path = out_dir / "combined_operating_points_table.tex"
    _write_csv(csv_path, rows)
    _write_tex(tex_path, rows)
    summary = {
        "sources": [{"split": split, "path": str(path.resolve())} for split, path in sources],
        "static_sources": [
            {"split": split, "path": str(path.resolve())} for split, path in static_sources.items()
        ],
        "outputs": {"csv": str(csv_path.resolve()), "tex": str(tex_path.resolve())},
        "claim_gate": [
            "Use Aqua registration-first/balanced thresholds for downstream front-end evidence, not only the clean-map threshold 0.55.",
            "DINO-box is a strong low-contamination baseline but can over-mask static structure and collapse E-success on broad WebUOT dynamic100.",
            "WebUOT labels remain tracked-target bounding boxes, not full dynamic-scene masks.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {csv_path}")
    print(f"Saved {tex_path}")
    print(f"Saved {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
