#!/usr/bin/env python3
"""Generate a runtime/memory table from Aqua-D4RT JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _nested_get(payload: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _read_runtime(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    runtime = _nested_get(payload, ["metadata", "runtime"])
    if runtime is None:
        runtime = _nested_get(payload, ["runtime"])
    if not isinstance(runtime, dict):
        return None
    out: dict[str, float] = {}
    for key in ("wall_seconds", "clips_per_second", "frames_per_second", "peak_vram_gb"):
        value = runtime.get(key)
        if value is not None:
            out[key] = float(value)
    return out


def _parse_artifacts(values: list[str] | None) -> list[tuple[str, Path, str]]:
    rows: list[tuple[str, Path, str]] = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Artifact must be NAME=PATH[:NOTE], got {value!r}")
        name, rest = value.split("=", 1)
        note = ""
        path_raw = rest
        if "::" in rest:
            path_raw, note = rest.split("::", 1)
        rows.append((name.strip(), Path(path_raw.strip()), note.strip()))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/aqua_runtime")
    parser.add_argument(
        "--artifact",
        action="append",
        default=None,
        metavar="NAME=JSON[::NOTE]",
        help="Read runtime from JSON. Supports aggregate_metrics.json with metadata.runtime or top-level runtime.",
    )
    parser.add_argument(
        "--include-r099-note",
        action="store_true",
        help="Append a manual caveat row for the R099 multi-pass selector.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _parse_artifacts(args.artifact)
    rows: list[dict[str, Any]] = []
    for name, path, note in artifacts:
        runtime = _read_runtime(path)
        if runtime is None:
            rows.append(
                {
                    "stage": name,
                    "source": str(path),
                    "wall_seconds": "",
                    "frames_per_second": "",
                    "clips_per_second": "",
                    "peak_vram_gb": "",
                    "note": f"runtime metadata missing. {note}".strip(),
                }
            )
            continue
        rows.append(
            {
                "stage": name,
                "source": str(path.resolve()),
                "wall_seconds": runtime.get("wall_seconds", ""),
                "frames_per_second": runtime.get("frames_per_second", ""),
                "clips_per_second": runtime.get("clips_per_second", ""),
                "peak_vram_gb": runtime.get("peak_vram_gb", ""),
                "note": note,
            }
        )
    if bool(args.include_r099_note):
        rows.append(
            {
                "stage": "R099 selector",
                "source": "tmp/aqua_adaptive_v3_t073_full_stress4_selector_v3_rawfallback/summary.csv",
                "wall_seconds": "",
                "frames_per_second": "",
                "clips_per_second": "",
                "peak_vram_gb": "",
                "note": "Multi-pass policy: raw and pose-soft candidate reconstructions are both computed before selection.",
            }
        )

    csv_path = out_dir / "runtime_table.csv"
    fieldnames = [
        "stage",
        "wall_seconds",
        "frames_per_second",
        "clips_per_second",
        "peak_vram_gb",
        "note",
        "source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tex_path = out_dir / "runtime_table.tex"
    lines = [
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        "Stage & Wall s & FPS & Clips/s & VRAM GB & Note \\\\",
        "\\midrule",
    ]
    for row in rows:
        def fmt(value: Any) -> str:
            if value == "":
                return "--"
            try:
                return f"{float(value):.2f}"
            except (TypeError, ValueError):
                return str(value)

        stage = str(row["stage"]).replace("_", "\\_")
        note = str(row["note"]).replace("_", "\\_")[:82]
        lines.append(
            f"{stage} & {fmt(row['wall_seconds'])} & "
            f"{fmt(row['frames_per_second'])} & {fmt(row['clips_per_second'])} & "
            f"{fmt(row['peak_vram_gb'])} & {note} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "artifacts": [{"name": name, "path": str(path), "note": note} for name, path, note in artifacts],
        "rows": rows,
        "outputs": {"csv": str(csv_path.resolve()), "tex": str(tex_path.resolve())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {csv_path}")
    print(f"Saved {tex_path}")
    print(f"Saved {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
