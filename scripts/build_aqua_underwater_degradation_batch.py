#!/usr/bin/env python3
"""Build lightweight underwater degradation batches for Aqua-D4RT.

Unlike the R116 smoke visualizer, this batch builder avoids videos/contact
sheets by default. It is intended for R117/R118 style metric sweeps where many
degraded WebUOT/AQUALOC/synthetic manifests are needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_aqua_underwater_degradation_smoke import (  # noqa: E402
    DegradationVariant,
    VARIANTS,
    _alignment_report,
    _apply_variant,
    _load_clip,
    _metric_summary,
)


def _read_manifest_list(path: str | Path) -> list[str]:
    items: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            items.append(item)
    return items


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _write_rgb(path: Path, rgb: np.ndarray, *, image_format: str, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    params: list[int] = []
    if image_format.lower() in {"jpg", "jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    ok = cv2.imwrite(str(path), bgr, params)
    if not ok:
        raise RuntimeError(f"Failed to write frame: {path}")


def _write_manifest(
    *,
    output_path: Path,
    source_manifest_path: Path,
    source_manifest: dict[str, Any],
    variant_name: str,
    variant_label: str,
    variant_severity: float,
    frame_paths: list[Path],
    labels_npz: Path,
    source_clip_name: str,
) -> None:
    payload = dict(source_manifest)
    payload.update(
        {
            "name": f"{source_clip_name}_{variant_name}",
            "source_manifest": str(source_manifest_path.resolve()),
            "degradation_variant": {
                "name": variant_name,
                "label": variant_label,
                "severity": float(variant_severity),
                "geometry_preserving": True,
            },
            "frames": [str(path.resolve()) for path in frame_paths],
            "frames_dir": str(frame_paths[0].parent.resolve()) if frame_paths else "",
            "labels_npz": str(labels_npz.resolve()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "notes": [
                "Batch underwater degradation is appearance-only; original masks are reused.",
                "Use for metric sweeps, not as new manually-labeled data.",
            ],
        }
    )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=None)
    parser.add_argument("--manifest-list", action="append", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--variants", default=",".join(v.name for v in VARIANTS))
    parser.add_argument(
        "--severity-values",
        default="",
        help="Optional comma-separated severity values to sweep for each selected variant family.",
    )
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--image-format", default="jpg", choices=("jpg", "png"))
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_manifests(args: argparse.Namespace) -> list[Path]:
    items: list[str] = []
    if args.manifest:
        for value in args.manifest:
            items.extend(part.strip() for part in str(value).split(",") if part.strip())
    if args.manifest_list:
        for value in args.manifest_list:
            items.extend(_read_manifest_list(value))
    out: list[Path] = []
    seen: set[str] = set()
    for item in items:
        path = Path(item)
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            out.append(path)
    if int(args.max_clips) > 0:
        out = out[: int(args.max_clips)]
    if not out:
        raise ValueError("Provide --manifest or --manifest-list.")
    return out


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_lookup = {variant.name: variant for variant in VARIANTS}
    variant_names = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    missing = [name for name in variant_names if name not in variant_lookup]
    if missing:
        raise ValueError(f"Unknown degradation variants: {missing}")
    severity_values = [float(item) for item in str(args.severity_values).split(",") if item.strip()]
    if severity_values:
        if any(value <= 0 for value in severity_values):
            raise ValueError(f"--severity-values must be positive, got: {severity_values}")
        variant_specs = [
            {
                "base_variant": variant_lookup[name],
                "output_name": _safe_stem(f"{name}_s{severity:.2f}"),
                "label": f"{variant_lookup[name].label} (s={severity:.2f})",
                "severity": float(severity),
            }
            for name in variant_names
            for severity in severity_values
        ]
    else:
        variant_specs = [
            {
                "base_variant": variant_lookup[name],
                "output_name": name,
                "label": variant_lookup[name].label,
                "severity": float(variant_lookup[name].severity),
            }
            for name in variant_names
        ]

    manifests = _resolve_manifests(args)
    manifest_lists: dict[str, list[str]] = {str(spec["output_name"]): [] for spec in variant_specs}
    all_degraded_manifests: list[str] = []
    per_clip: list[dict[str, Any]] = []

    ext = "jpg" if args.image_format == "jpg" else "png"
    for clip_idx, manifest_path in enumerate(manifests):
        manifest, frames, masks = _load_clip(manifest_path, max_frames=int(args.max_frames))
        clip_name = _safe_stem(str(manifest.get("name", manifest_path.parent.name or manifest_path.stem)))
        clip_dir = output_dir / clip_name
        clip_dir.mkdir(parents=True, exist_ok=True)
        labels_npz = clip_dir / "labels_reused_masks.npz"
        if bool(args.overwrite) or not labels_npz.exists():
            np.savez_compressed(
                labels_npz,
                dynamic_object_mask=masks["dynamic"],
                particle_mask=masks["particle"],
                transient_mask=masks["transient"],
            )

        clip_summary: dict[str, Any] = {
            "source_manifest": str(manifest_path.resolve()),
            "clip_name": clip_name,
            "num_frames": len(frames),
            "alignment": _alignment_report(frames, masks),
            "original_appearance_metrics": _metric_summary(frames),
            "variants": {},
        }
        print(f"[{clip_idx + 1}/{len(manifests)}] {clip_name}", flush=True)
        for variant_idx, spec in enumerate(variant_specs):
            base_variant = spec["base_variant"]
            variant_name = str(spec["output_name"])
            variant_label = str(spec["label"])
            variant_severity = float(spec["severity"])
            variant_dir = clip_dir / variant_name
            frames_dir = variant_dir / "frames"
            manifest_out = variant_dir / "manifest.json"
            if manifest_out.exists() and not bool(args.overwrite):
                manifest_lists[variant_name].append(str(manifest_out.resolve()))
                all_degraded_manifests.append(str(manifest_out.resolve()))
                clip_summary["variants"][variant_name] = {
                    "manifest": str(manifest_out.resolve()),
                    "skipped_existing": True,
                    "degradation_variant": {
                        "name": variant_name,
                        "label": variant_label,
                        "severity": float(variant_severity),
                    },
                }
                continue

            rng = np.random.default_rng(int(args.seed) + 1000003 * clip_idx + 10007 * (variant_idx + 1))
            degraded: list[np.ndarray] = []
            frame_paths: list[Path] = []
            for frame_idx, rgb in enumerate(frames):
                out = _apply_variant(rgb, base_variant, frame_idx=frame_idx, rng=rng)
                degraded.append(out)
                frame_path = frames_dir / f"frame_{frame_idx:06d}.{ext}"
                _write_rgb(
                    frame_path,
                    out,
                    image_format=str(args.image_format),
                    jpeg_quality=int(args.jpeg_quality),
                )
                frame_paths.append(frame_path)

            _write_manifest(
                output_path=manifest_out,
                source_manifest_path=manifest_path,
                source_manifest=manifest,
                variant_name=variant_name,
                variant_label=variant_label,
                variant_severity=variant_severity,
                frame_paths=frame_paths,
                labels_npz=labels_npz,
                source_clip_name=clip_name,
            )
            manifest_lists[variant_name].append(str(manifest_out.resolve()))
            all_degraded_manifests.append(str(manifest_out.resolve()))
            clip_summary["variants"][variant_name] = {
                "manifest": str(manifest_out.resolve()),
                "frames_dir": str(frames_dir.resolve()),
                "appearance_metrics": _metric_summary(degraded),
                "degradation_variant": {
                    "name": variant_name,
                    "label": variant_label,
                    "severity": float(variant_severity),
                },
            }
        per_clip.append(clip_summary)

    lists_dir = output_dir / "splits"
    lists_dir.mkdir(parents=True, exist_ok=True)
    all_list_path = lists_dir / "all_degraded_manifests.txt"
    all_list_path.write_text("\n".join(all_degraded_manifests) + "\n", encoding="utf-8")
    variant_list_paths: dict[str, str] = {}
    for name, paths in manifest_lists.items():
        path = lists_dir / f"{name}_manifests.txt"
        path.write_text("\n".join(paths) + "\n", encoding="utf-8")
        variant_list_paths[name] = str(path.resolve())

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifests": [str(path.resolve()) for path in manifests],
        "num_source_clips": len(manifests),
        "num_degraded_manifests": len(all_degraded_manifests),
        "variants": [str(spec["output_name"]) for spec in variant_specs],
        "severity_values": severity_values,
        "max_frames": int(args.max_frames),
        "image_format": str(args.image_format),
        "jpeg_quality": int(args.jpeg_quality) if args.image_format == "jpg" else None,
        "all_degraded_manifest_list": str(all_list_path.resolve()),
        "variant_manifest_lists": variant_list_paths,
        "per_clip": per_clip,
        "claim_gate": [
            "Geometry-preserving appearance degradation only; original masks are reused.",
            "Use downstream metric scripts before making robustness claims.",
        ],
    }
    summary_path = output_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {summary_path}")
    print(f"Saved {all_list_path} ({len(all_degraded_manifests)} manifests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
