# Aqua-D4RT Data Pipeline

This note records the reproducible data path used for the first Aqua-D4RT prototype.

## Prepared Data

```text
data/aqua_smoke/underwater_caves_sonar_32/
  frames/                 # 32 symlinked underwater background frames
  manifest.json
  preview.mp4

data/watermask_uiis/
  README.md
  annotations/train.json
  annotations/val.json
  train/*.jpg             # downloaded fish subset images
  fish_subset_train_0120.json

data/aqua_synth/watermask_caves_32/
  frames/                 # corrupted frames, D4RT smoke input
  frames_clean/
  masks/dynamic_object/
  masks/particle/
  masks/transient/
  labels/transient_masks.npz
  manifest.json
  preview_corrupt.mp4
  preview_masks.mp4

data/real_underwater/tank_short_test/
  short_test.yaml
  extracted/short_test/        # IMG_L/IMG_R + gt/depth/imu/dvl csv
  clip_img_r_32/               # 32-frame right-camera clip for D4RT/Aqua sanity
```

The Tank `short_test.zip` archive was removed after extraction to avoid keeping a duplicate copy; `extracted/short_test/` remains the source used by scripts.

## Commands

Prepare a small underwater background clip:

```bash
/media/data/u24conda/envs/longlive/bin/python scripts/prepare_aqua_smoke_data.py \
  --num-frames 32 \
  --stride 2 \
  --output-dir data/aqua_smoke/underwater_caves_sonar_32
```

Download a compact WaterMask/UIIS fish subset:

```bash
/media/data/u24conda/envs/longlive/bin/python scripts/download_watermask_fish_subset.py \
  --output-root data/watermask_uiis \
  --split train \
  --category fish \
  --limit 120 \
  --min-ann-area 96 \
  --selection largest
```

Build the synthetic transient clip:

```bash
/media/data/u24conda/envs/longlive/bin/python scripts/build_aqua_synthetic_transients.py \
  --background data/aqua_smoke/underwater_caves_sonar_32 \
  --fish-manifest data/watermask_uiis/fish_subset_train_0120.json \
  --output-dir data/aqua_synth/watermask_caves_32 \
  --num-frames 32 \
  --fish-tracks 5 \
  --particles-min 80 \
  --particles-max 180 \
  --seed 20260610
```

Run OpenD4RT smoke inference on the synthetic corrupted clip:

```bash
CUDA_VISIBLE_DEVICES=0 /media/data/u24conda/envs/longlive/bin/python scripts/run_d4rt_smoke.py \
  --input data/aqua_synth/watermask_caves_32 \
  --device cuda \
  --max-frames 32 \
  --query-cols 4 \
  --query-rows 4 \
  --max-queries 16 \
  --query-chunk-size 16 \
  --output-dir tmp/aqua_smoke/d4rt_synth_watermask_caves_16q \
  --save-overlay
```

## Current Smoke Results

- Synthetic mask coverage:
  - dynamic object: `0.0696`
  - particle: `0.0397`
  - transient: `0.1068`
- D4RT synthetic smoke:
  - `finite_xyz_ratio=1.0000`
  - `visible_ratio=0.0312`
  - `mean_static_confidence=0.3095`
  - CUDA peak memory: about `4940 MB`

Note: use `/media/data/u24conda/envs/longlive` for RTX 5090. The current
`d4rt` conda environment has `torch 2.6.0+cu124`, which cannot execute CUDA
kernels on `sm_120`.

## Real Tank Sample

See:

```text
docs/aqua_real_tank_sanity.md
```

This sample is useful for real-domain smoke testing, but it is mostly static and
does not validate real fish/particle removal.
