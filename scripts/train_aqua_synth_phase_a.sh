#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-/media/data/u24conda/envs/longlive/bin/python}"
MODEL_CONFIG="${MODEL_CONFIG:-checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_aqua_synth_phase_a.yaml}"
INIT_MODEL="${INIT_MODEL:-checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt}"

"${PYTHON_BIN}" train.py \
  --model-config "${MODEL_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --init-model "${INIT_MODEL}" \
  "$@"
