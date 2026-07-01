#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export EXP_NAME="${EXP_NAME:-aqua_d4rt_phase_a_ninemix_clip48}"
export EXP_OUTPUT="${EXP_OUTPUT:-aqua_d4rt_phase_a_ninemix_clip48}"
export OUT_ROOT="${OUT_ROOT:-output/exp_aqua_d4rt}"
export AUTO_EVAL_WORLDTRACK_ENABLED="${AUTO_EVAL_WORLDTRACK_ENABLED:-false}"
export TOTAL_STEPS="${TOTAL_STEPS:-8000}"
export PEAK_LR="${PEAK_LR:-4.0e-6}"
export FINAL_LR="${FINAL_LR:-4.0e-7}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE:-1}"

bash scripts/train_worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_8gpu.sh \
  --override "augmentation.underwater_transient.enabled=true" \
  --override "augmentation.underwater_transient.train_only=true" \
  --override "augmentation.underwater_transient.apply_probability=${AQUA_APPLY_PROBABILITY:-1.0}" \
  --override "loss.transient.enabled=true" \
  --override "loss.transient.geometry_masking.enabled=true" \
  --override "loss.transient.dynamic_object.enabled=true" \
  --override "loss.transient.dynamic_object.weight_lambda=${AQUA_DYNAMIC_LOSS_WEIGHT:-0.1}" \
  --override "loss.transient.particle.enabled=true" \
  --override "loss.transient.particle.weight_lambda=${AQUA_PARTICLE_LOSS_WEIGHT:-0.1}" \
  --override "fine_tuning.freeze_encoder=true" \
  --override "fine_tuning.freeze_memory_proj=false" \
  --override "fine_tuning.freeze_query_embedder=false" \
  --override "fine_tuning.freeze_decoder=false" \
  --override "fine_tuning.freeze_heads=false" \
  "$@"
