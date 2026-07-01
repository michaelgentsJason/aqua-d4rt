#!/usr/bin/env python3
"""Run Tank stress-v2 high-stress pyCOLMAP evaluation in resumable shards."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/media/data/u24conda/envs/longlive/bin/python")
DEFAULT_MANIFEST_LIST = Path("data/real_underwater/tank_pose_stress_v2/manifests_stress4.txt")
DEFAULT_OUTPUT_ROOT = Path("tmp/aqua_tank_pose_stress_v2_stress4_multiseed_t090_shards")
DEFAULT_CKPT = Path("output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt")
DEFAULT_SCORER = Path("tmp/aqua_retention_scorer/webuot_synth_tank_mix_train_v2/retention_scorer.pt")
STRESS_VARIANTS = ("fish-high", "fish-extreme", "snow-high", "mixed-fish-snow")


def _threshold_tag(threshold: float) -> str:
    return f"t{float(threshold):.2f}".replace(".", "p")


def _read_manifest_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _write_shard_manifest(output_root: Path, stress_variant: str, manifests: list[str]) -> Path:
    selected = [item for item in manifests if f"/{stress_variant}/manifest.json" in item]
    if not selected:
        raise RuntimeError(f"No manifests found for stress variant: {stress_variant}")
    shard_dir = output_root / stress_variant
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_manifest = shard_dir / "manifests.txt"
    shard_manifest.write_text("\n".join(selected) + "\n", encoding="utf-8")
    return shard_manifest


def _is_complete(shard_output: Path) -> bool:
    return (shard_output / "aggregate_metrics.json").exists() and (shard_output / "summary_seed_stability.csv").exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-list", default=str(DEFAULT_MANIFEST_LIST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", default=str(DEFAULT_CKPT))
    parser.add_argument("--retention-scorer-path", default=str(DEFAULT_SCORER))
    parser.add_argument("--stress-variants", default=",".join(STRESS_VARIANTS))
    parser.add_argument("--pycolmap-random-seeds", default="42,43,44")
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--aqua-window-size", type=int, default=32)
    parser.add_argument("--aqua-grid-stride", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--max-runtime-seconds", type=float, default=45.0)
    parser.add_argument("--max-num-features", type=int, default=4096)
    parser.add_argument("--retention-score-threshold", type=float, default=0.90)
    parser.add_argument("--retention-patch-radius", type=int, default=5)
    parser.add_argument("--retention-min-inlier-support", type=int, default=1)
    parser.add_argument("--retention-max-features-per-frame", type=int, default=300)
    parser.add_argument("--retention-max-fraction", type=float, default=0.18)
    parser.add_argument("--enable-adaptive-retention", action="store_true")
    parser.add_argument("--adaptive-particle-coverage-high", type=float, default=0.025)
    parser.add_argument("--adaptive-rejected-fraction-high", type=float, default=0.42)
    parser.add_argument("--adaptive-retained-fraction-low", type=float, default=0.015)
    parser.add_argument("--adaptive-model-pair-rate-low", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_list = Path(args.manifest_list)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = _read_manifest_list(manifest_list)
    stress_variants = [item.strip() for item in str(args.stress_variants).split(",") if item.strip()]
    if not stress_variants:
        raise ValueError("--stress-variants cannot be empty")
    retention_tag = _threshold_tag(float(args.retention_score_threshold))

    for stress_variant in stress_variants:
        shard_manifest = _write_shard_manifest(output_root, stress_variant, manifests)
        shard_output = output_root / stress_variant / "eval"
        log_path = output_root / stress_variant / "eval.log"
        if _is_complete(shard_output) and not bool(args.force):
            print(f"[skip] {stress_variant}: complete at {shard_output}", flush=True)
            continue

        cmd = [
            str(PYTHON),
            "scripts/eval_aqua_pose_gt_validation.py",
            "--manifest-list",
            str(shard_manifest),
            "--model-config",
            str(args.model_config),
            "--ckpt-path",
            str(args.ckpt_path),
            "--retention-scorer-path",
            str(args.retention_scorer_path),
            "--output-dir",
            str(shard_output),
            "--device",
            str(args.device),
            "--max-frames",
            str(int(args.max_frames)),
            "--frame-stride",
            str(int(args.frame_stride)),
            "--aqua-window-size",
            str(int(args.aqua_window_size)),
            "--aqua-grid-stride",
            str(int(args.aqua_grid_stride)),
            "--query-chunk-size",
            str(int(args.query_chunk_size)),
            "--max-runtime-seconds",
            str(float(args.max_runtime_seconds)),
            "--max-num-features",
            str(int(args.max_num_features)),
            "--no-temporal-rgb",
            "--no-oracle",
            "--enable-learned-retention",
            "--enable-pose-aware-soft-retention",
            "--retention-score-threshold",
            str(float(args.retention_score_threshold)),
            "--retention-patch-radius",
            str(int(args.retention_patch_radius)),
            "--retention-min-inlier-support",
            str(int(args.retention_min_inlier_support)),
            "--retention-max-features-per-frame",
            str(int(args.retention_max_features_per_frame)),
            "--retention-max-fraction",
            str(float(args.retention_max_fraction)),
            "--adaptive-particle-coverage-high",
            str(float(args.adaptive_particle_coverage_high)),
            "--adaptive-rejected-fraction-high",
            str(float(args.adaptive_rejected_fraction_high)),
            "--adaptive-retained-fraction-low",
            str(float(args.adaptive_retained_fraction_low)),
            "--adaptive-model-pair-rate-low",
            str(float(args.adaptive_model_pair_rate_low)),
            "--pycolmap-random-seeds",
            str(args.pycolmap_random_seeds),
            "--fixed-initial-pair",
            "auto",
            "--variant-filter",
            "raw",
            "--variant-filter",
            "aqua_inpaint",
            "--variant-filter",
            f"aqua_learned_retain_{retention_tag}_inpaint",
            "--variant-filter",
            f"aqua_pose_soft_{retention_tag}",
            "--seed",
            str(int(args.seed)),
        ]
        if bool(args.enable_adaptive_retention):
            cmd.extend(
                [
                    "--enable-adaptive-retention",
                    "--variant-filter",
                    f"aqua_adaptive_retain_{retention_tag}",
                ]
            )
        print(f"[run] {stress_variant}: {' '.join(cmd)}", flush=True)
        print(f"[log] {log_path}", flush=True)
        if bool(args.dry_run):
            continue
        with log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if proc.returncode != 0:
            print(f"[fail] {stress_variant}: exit code {proc.returncode}; see {log_path}", file=sys.stderr)
            return int(proc.returncode)
        print(f"[done] {stress_variant}: {shard_output}", flush=True)
    print(f"All requested shards finished under {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
