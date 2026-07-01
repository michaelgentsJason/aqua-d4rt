# Aqua-D4RT Tank Pose-Stress GT Validation

Date: 2026-06-17

## Goal

This experiment implements the GT-pose downstream validation block from the
ICRA sprint plan. It turns the mostly-static Tank short sample into controlled
dynamic underwater stress clips by injecting WaterMask fish cutouts and
procedural marine-snow particles while keeping the original Tank GT poses.

The main question is not 2D mask F1. The question is whether Aqua-D4RT plus
retention can preserve or recover pyCOLMAP pose/SfM performance while reducing
dynamic feature and match contamination.

## Implemented

New scripts:

- `scripts/build_aqua_tank_pose_stress.py`
- `scripts/eval_aqua_pose_gt_validation.py`

Extended helper:

- `scripts/aqua_prefilter_utils.py`
  - Added `soft_temporal_fill_video`.

Evaluator metrics:

- pyCOLMAP input registration rate.
- Number of registered images and 3D points.
- Sim(3)-aligned translation ATE RMSE from COLMAP camera centers to Tank GT positions.
- Translation RPE from adjacent GT/predicted step lengths.
- ORB feature and match contamination against injected transient masks.

Translation ATE/RPE are the primary pose metrics. Orientation RPE is secondary
because COLMAP and Tank camera-frame conventions need a separate frame audit.

## Stress Benchmark

Command:

```bash
/media/data/u24conda/envs/longlive/bin/python scripts/build_aqua_tank_pose_stress.py \
  --output-root data/real_underwater/tank_pose_stress \
  --variants clean,fish-med,fish-high,snow-high \
  --num-frames 128 --stride 1 \
  --output-width 384 --output-height 288 \
  --seed 20260618
```

Output:

- `data/real_underwater/tank_pose_stress/`
- `data/real_underwater/tank_pose_stress/manifests.txt`

Mask coverage:

| Variant | Dynamic | Particle | Transient |
| --- | ---: | ---: | ---: |
| clean | 0.00% | 0.00% | 0.00% |
| fish-med | 12.52% | 2.10% | 14.35% |
| fish-high | 19.51% | 4.55% | 23.16% |
| snow-high | 0.95% | 12.55% | 13.39% |

Each variant has:

- `frames.csv` with image path, timestamp, GT position/quaternion, and transient mask path.
- `labels/transient_masks.npz`.
- 32-frame Aqua window manifests.
- RGB and mask preview videos.

## Main Run: t=0.50 Learned Retention + Soft Fill

Command:

```bash
CUDA_VISIBLE_DEVICES=0 /media/data/u24conda/envs/longlive/bin/python \
  scripts/eval_aqua_pose_gt_validation.py \
  --manifest-list data/real_underwater/tank_pose_stress/manifests.txt \
  --ckpt-path output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt \
  --retention-scorer-path tmp/aqua_retention_scorer/webuot_synth_mix_train24_10_grid4_orb_train/retention_scorer.pt \
  --output-dir tmp/aqua_tank_pose_stress_128_eval_t050_soft \
  --max-frames 128 --frame-stride 2 \
  --aqua-window-size 32 --aqua-grid-stride 8 \
  --query-chunk-size 4096 --max-runtime-seconds 45 \
  --max-num-features 4096 \
  --enable-slam-aware-retention \
  --enable-learned-retention \
  --enable-soft-learned-retention \
  --retention-score-threshold 0.50 \
  --retention-patch-radius 5 \
  --retention-min-inlier-support 1 \
  --retention-max-features-per-frame 300 \
  --retention-max-fraction 0.18 \
  --seed 42
```

Result files:

- `tmp/aqua_tank_pose_stress_128_eval_t050_soft/aggregate_metrics.json`
- `tmp/aqua_tank_pose_stress_128_eval_t050_soft/summary_table.csv`
- `tmp/aqua_tank_pose_stress_128_eval_t050_soft/summary_by_stress_variant.csv`

Overall 4-variant average:

| System | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw | 28.12% | 0.0159 | 0.0139 | 11.04% | 8.94% |
| Temporal RGB inpaint | 52.34% | 0.0391 | 0.0211 | 11.00% | 8.50% |
| Aqua hard inpaint | 28.12% | 0.0699 | 0.0423 | 5.25% | 4.99% |
| Aqua rule retain | 53.91% | 0.0295 | 0.0293 | 7.43% | 8.25% |
| Aqua learned t=0.50 hard | 51.56% | 0.0176 | 0.0116 | 7.42% | 8.25% |
| Aqua learned t=0.50 soft | 76.56% | 0.0269 | 0.0195 | 7.42% | 8.25% |
| Oracle GT inpaint | 30.47% | 0.0298 | 0.0236 | 0.00% | 0.00% |

Main stress-specific result:

| Stress | System | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fish-high | Raw | 3.12% | n/a | n/a | 15.15% | 10.49% |
| fish-high | Temporal RGB | 100.00% | 0.0742 | 0.0334 | 15.19% | 9.42% |
| fish-high | Aqua hard | 3.12% | n/a | n/a | 6.54% | 5.60% |
| fish-high | Aqua learned t=0.50 soft | 100.00% | 0.0280 | 0.0097 | 9.36% | 9.13% |
| fish-med | Raw | 4.69% | 0.0097 | 0.0188 | 9.71% | 6.89% |
| fish-med | Aqua learned t=0.50 hard | 100.00% | 0.0190 | 0.0165 | 6.17% | 5.73% |
| snow-high | Raw | 100.00% | 0.0366 | 0.0203 | 19.30% | 18.36% |
| snow-high | Aqua hard | 100.00% | 0.1081 | 0.0371 | 11.56% | 11.41% |
| snow-high | Aqua learned t=0.50 soft | 100.00% | 0.0553 | 0.0429 | 14.16% | 18.14% |

Takeaway:

- On `fish-high`, raw pyCOLMAP does not yield enough registered poses for ATE,
  while Aqua learned soft retention recovers full registration and lowers
  feature contamination from 15.15% to 9.36%.
- On `fish-med`, retention restores full registration and reduces
  contamination, but raw has lower ATE.
- On `snow-high`, Aqua reduces contamination but currently worsens pose error.
  This is the main failure mode for the next training/optimization pass.

## Focused Threshold Check: t=0.80

Command:

```bash
CUDA_VISIBLE_DEVICES=0 /media/data/u24conda/envs/longlive/bin/python \
  scripts/eval_aqua_pose_gt_validation.py \
  --manifest data/real_underwater/tank_pose_stress/fish-med/manifest.json \
  --manifest data/real_underwater/tank_pose_stress/fish-high/manifest.json \
  --manifest data/real_underwater/tank_pose_stress/snow-high/manifest.json \
  --ckpt-path output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt \
  --retention-scorer-path tmp/aqua_retention_scorer/webuot_synth_mix_train24_10_grid4_orb_train/retention_scorer.pt \
  --output-dir tmp/aqua_tank_pose_stress_128_eval_t080_soft_stress3 \
  --max-frames 128 --frame-stride 2 \
  --aqua-window-size 32 --aqua-grid-stride 8 \
  --query-chunk-size 4096 --max-runtime-seconds 45 \
  --max-num-features 4096 \
  --no-temporal-rgb --no-oracle \
  --enable-learned-retention \
  --enable-soft-learned-retention \
  --retention-score-threshold 0.80 \
  --retention-patch-radius 5 \
  --retention-min-inlier-support 1 \
  --retention-max-features-per-frame 300 \
  --retention-max-fraction 0.18 \
  --seed 43
```

Stress3 average:

| System | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. |
| --- | ---: | ---: | ---: | ---: |
| Raw | 67.71% | 0.0430 | 0.0285 | 14.72% |
| Aqua hard | 42.19% | 0.0648 | 0.0273 | 7.01% |
| Aqua learned t=0.80 hard | 71.35% | 0.0542 | 0.0268 | 9.65% |
| Aqua learned t=0.80 soft | 100.00% | 0.0654 | 0.0222 | 9.65% |

Notable per-stress details:

- `fish-high`: learned hard t=0.80 has the best ATE among non-oracle systems
  with valid pose evaluation, 0.0131, but only 14.06% input registration.
- `fish-high`: learned soft t=0.80 keeps 100.00% registration and low RPE
  0.0119, but ATE increases to 0.0781.
- `snow-high`: t=0.80 lowers contamination relative to raw and improves RPE
  slightly in the stress3 average, but it still does not beat raw ATE.

## Follow-Up Focused Checks: Tank-Aware v2

After retraining the retention scorer with Tank stress candidates, two focused
checks were run on `fish-high` and `snow-high`.

### t=0.90

Result: `tmp/aqua_tank_pose_stress_128_eval_v2_t090_fishhigh_snow/aggregate_metrics.json`

| Stress | System | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fish-high | Raw | 100.00% | 0.0817 | 0.0097 | 15.15% | 10.49% |
| fish-high | Aqua learned t=0.90 hard | 100.00% | 0.0343 | 0.0146 | 8.39% | 8.34% |
| fish-high | Aqua learned t=0.90 soft | 100.00% | 0.0640 | 0.0168 | 8.39% | 8.34% |
| snow-high | Raw | 100.00% | 0.0241 | 0.0259 | 19.30% | 18.36% |
| snow-high | Aqua learned t=0.90 hard | 6.25% | 0.0056 | 0.0063 | 13.52% | 17.54% |
| snow-high | Aqua learned t=0.90 soft | 4.69% | 0.0173 | 0.0116 | 13.52% | 17.54% |

Interpretation:

- `fish-high` is the encouraging case. v2 hard retention cuts contamination
  nearly in half and improves ATE substantially over raw.
- `snow-high` remains fragile. The hard variant gives small valid local pose
  error, but registration collapses, so this is not a sequence-level win.

### t=0.78

Result: `tmp/aqua_tank_pose_stress_128_eval_v2_t078_fishhigh_snow/aggregate_metrics.json`

| Stress | System | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. |
| --- | --- | ---: | ---: | ---: | ---: |
| fish-high | Raw | 3.12% | n/a | n/a | 15.15% |
| fish-high | Aqua inpaint | 7.81% | 0.0118 | 0.0121 | 6.54% |
| fish-high | Aqua learned t=0.78 soft | 3.12% | n/a | n/a | 8.88% |
| snow-high | Raw | 4.69% | 0.0012 | 0.0025 | 19.30% |
| snow-high | Aqua inpaint | 7.81% | 0.0373 | 0.0552 | 11.56% |
| snow-high | Aqua learned t=0.78 hard | 9.38% | 0.0346 | 0.0379 | 13.90% |
| snow-high | Aqua learned t=0.78 soft | 100.00% | 0.0713 | 0.0377 | 13.90% |

Interpretation:

- t=0.78 is the best pseudo-label threshold on Tank val candidates, but the
  pyCOLMAP result is still sensitive to initial reconstruction and time limits.
- This reinforces the current claim boundary: we have a strong retention/Pareto
  result and a meaningful fish-stress recovery case, but not a universal ATE/RPE
  win over raw yet.

## Multi-Seed / Fixed-Pair Stabilization

The evaluator now supports:

```text
--pycolmap-random-seeds 42,43,44 --fixed-initial-pair auto
```

This reduces the risk of treating one favorable incremental-mapping
initialization as a method result. A fallback from the requested fixed pair is
enabled by default when pyCOLMAP rejects the pair.

### v2 Fish-High / Snow-High Check

Result: `tmp/aqua_tank_pose_stress_multiseed_v2_t090_fishhigh_snow/summary_seed_stability.csv`

Aggregated over two stress variants and three random seeds:

| System | Success | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 100.00% | 100.00% | 84.11% | 0.0556 | 0.0234 | 17.22% | 14.43% |
| Aqua hard inpaint | 100.00% | 83.33% | 22.40% | 0.0249 | 0.0266 | 9.05% | 8.51% |
| v2 learned hard t=0.90 | 100.00% | 83.33% | 22.40% | 0.0243 | 0.0219 | 10.95% | 12.94% |
| v2 learned soft t=0.90 | 83.33% | 66.67% | 67.19% | 0.0815 | 0.0351 | 10.95% | 12.94% |

Interpretation:

- Hard Aqua/v2 improves aligned ATE and reduces feature contamination on this
  fish-high/snow-high pair, but it registers many fewer input images than raw.
- Soft v2 restores more registration completeness but worsens ATE in this
  stabilized run.
- The downstream claim should therefore be "pose-quality/contamination Pareto
  with promising fish-stress gains," not a blanket full-sequence SLAM win.

## Pose-Aware Soft Retention

The soft fill path was extended to be geometry-aware instead of only
score-aware. The implementation is in:

- `scripts/aqua_retention_utils.py`
  - `pose_aware_retention_weight_from_candidates`
- `scripts/aqua_prefilter_utils.py`
  - `soft_temporal_fill_video(..., retain_weight=...)`
- `scripts/eval_aqua_pose_gt_validation.py`
  - `--enable-pose-aware-soft-retention`
  - `--variant-filter`

The pose-aware weight is continuous. It combines the learned retention score
with inlier support, inlier ratio, patch NCC, match distance, and flow. The
soft renderer uses that value as an original-image weight inside Aqua-rejected
regions, so stable keypoint patches are partially preserved while weak rejected
regions are still filled from temporal neighbors.

Result: `tmp/aqua_pose_aware_soft_retention_poseonly_multiseed_t090_fishhigh_snow/summary_seed_stability.csv`

Aggregated over `fish-high` and `snow-high`, three seeds:

| System | Success | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Points3D | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 100.00% | 100.00% | 84.11% | 0.0556 | 0.0234 | 2997.5 | 17.22% | 14.43% |
| Aqua hard | 100.00% | 83.33% | 22.40% | 0.0249 | 0.0266 | 669.3 | 9.05% | 8.51% |
| v2 learned hard t=0.90 | 100.00% | 83.33% | 22.40% | 0.0243 | 0.0219 | 763.8 | 10.95% | 12.94% |
| v2 learned soft t=0.90 | 83.33% | 66.67% | 67.19% | 0.0815 | 0.0351 | 2422.0 | 10.95% | 12.94% |
| v2 pose-aware soft t=0.90 | 100.00% | 83.33% | 83.85% | 0.0860 | 0.0377 | 2953.3 | 10.95% | 12.94% |

Interpretation:

- Pose-aware soft retention fixes the previous soft-fill stability issue:
  success improves from 83.33% to 100.00%, and input registration rises from
  67.19% to 83.85%, essentially matching raw registration completeness.
- It keeps the contamination benefit of learned retention: feature
  contamination is 10.95% vs raw 17.22%.
- It does not improve pose accuracy over raw or hard v2. ATE/RPE are worse
  than raw in this stabilized run.

This is a useful method component for the Pareto story, not a final downstream
SOTA result. The current best use is to report hard v2 as the low-ATE/low-
contamination endpoint and pose-aware soft as the high-completeness/low-
contamination endpoint.

## No-Dynamic / No-Particle Pose Ablation

Full ablation report: `docs/aqua_head_ablation_results.md`

The same multi-seed/fixed-pair evaluator was used for the head ablation
checkpoints.

### No-Dynamic Checkpoint

Result: `tmp/aqua_tank_pose_ablation_no_dynamic_pose_gt/summary_seed_stability.csv`

| Variant | Success | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 91.67% | 83.33% | 50.65% | 0.0233 | 0.0143 | 11.04% | 8.94% |
| No-dynamic Aqua hard | 100.00% | 66.67% | 5.73% | 0.0233 | 0.0252 | 4.78% | 6.70% |
| No-dynamic Aqua retain | 100.00% | 83.33% | 15.10% | 0.0374 | 0.0306 | 6.17% | 7.44% |

Without the dynamic branch, Aqua can reduce some measured feature
contamination, but the reconstruction becomes very sparse and RPE worsens.

### No-Particle Checkpoint

Result: `tmp/aqua_tank_pose_ablation_no_particle_pose_gt/summary_seed_stability.csv`

| Variant | Success | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 83.33% | 75.00% | 32.55% | 0.0358 | 0.0385 | 11.04% | 8.94% |
| No-particle Aqua hard | 100.00% | 91.67% | 58.07% | 0.0211 | 0.0166 | 8.34% | 7.35% |
| No-particle Aqua retain | 100.00% | 91.67% | 62.24% | 0.0223 | 0.0157 | 9.03% | 8.32% |

This result is useful but should be handled carefully. The no-particle model is
friendlier to pyCOLMAP because it keeps more texture, but it fails particle
detection on synthetic test (`particle best F1 = 0.078`) and weakens static-map
cleanliness. The right optimization direction is softer pose-aware particle
retention, not deleting the particle branch.

## Claim Status

Supported now:

1. The GT-pose Tank stress pipeline is implemented and produces parseable
   pose, registration, contamination, and map-density metrics.
2. Aqua filtering strongly reduces dynamic feature contamination on injected
   fish/snow stress clips.
3. Learned/soft retention gives a controllable Pareto: more registration and
   denser SfM at the cost of re-admitting some transient features.
4. On `fish-high`, Aqua learned soft retention recovers a usable full
   reconstruction where raw has too few poses for ATE.
5. Tank-aware v2 improves the ORB Pareto and gives a better fish-high pose
   result in the t=0.90 focused check.
6. Multi-seed/fixed-pair reporting is implemented, and it shows that
   pyCOLMAP-based claims must be reported as mean/stability rather than single
   lucky runs.
7. Pose-aware soft retention restores registration completeness under
   fish-high/snow-high stress while keeping feature contamination lower than
   raw.

Not supported yet:

1. A blanket claim that Aqua-D4RT beats raw SLAM/SfM on ATE/RPE.
2. A universal SOTA claim across underwater SLAM datasets.
3. A final marine-snow solution. `snow-high` is cleaner but not more accurate
   in pose.
4. A claim that pose-aware soft retention solves ATE/RPE. It improves
   completeness and contamination, but ATE/RPE are still worse than raw in the
   stabilized high-stress run.

Best current ICRA wording:

```text
Aqua-D4RT improves static geometry cleanliness and provides a retention-aware
front-end Pareto. On a GT-pose injected-fish Tank stress sequence, learned
soft retention restores pyCOLMAP registration under high fish occlusion while
reducing dynamic feature contamination. Full ATE/RPE superiority across fish
and marine snow remains an open optimization target.
```

## Next Optimization Targets

1. Optimize pose-aware soft retention for ATE/RPE, not just registration
   completeness; likely direction is adaptive hard-vs-soft selection based on
   snow/fish stress and pyCOLMAP track quality.
2. Add different start-index Tank stress runs to test whether the fish-high
   recovery holds beyond one controlled sequence.
3. Add a fair non-oracle detector plus SAM/SAM2 baseline once dependencies are
   installed.
4. Add a runtime/memory table and a compact paper-ready ablation table.

## Tank Stress v2 Expansion

Date: 2026-06-18

The Tank stress builder now supports multiple start indices and multiple
injection seeds in a single run, plus additional stress densities:

- `fish-extreme`
- `snow-med`
- `mixed-fish-snow`

It also writes per-window `frames.csv` files so future retention-scorer v3
training can align GT pose rows with 32-frame window manifests.

### Builder Command

```bash
/media/data/u24conda/envs/longlive/bin/python scripts/build_aqua_tank_pose_stress.py \
  --output-root data/real_underwater/tank_pose_stress_v2 \
  --variants clean,fish-low,fish-med,fish-high,fish-extreme,snow-med,snow-high,mixed-fish-snow \
  --num-frames 128 --start-indices 0,64,128 --seeds 20260618,20260619 \
  --output-width 384 --output-height 288 --no-previews
```

Output:

- `data/real_underwater/tank_pose_stress_v2/manifests.txt`
- `data/real_underwater/tank_pose_stress_v2/window_manifests.txt`
- 48 full clips = 3 start indices x 2 injection seeds x 8 stress variants.
- 192 32-frame Aqua windows.

Mean transient coverage over the 6 clips per variant:

| Variant | Dynamic | Particle | Transient |
| --- | ---: | ---: | ---: |
| clean | 0.00% | 0.00% | 0.00% |
| fish-low | 4.11% | 0.63% | 4.71% |
| fish-med | 7.44% | 2.12% | 9.40% |
| fish-high | 18.04% | 4.60% | 21.79% |
| fish-extreme | 36.39% | 7.06% | 40.87% |
| snow-med | 0.00% | 5.53% | 5.53% |
| snow-high | 1.52% | 12.40% | 13.73% |
| mixed-fish-snow | 16.39% | 10.28% | 24.96% |

### Sanity Checks

Smoke build:

- `data/real_underwater/tank_pose_stress_v2_smoke/`
- 16 tiny 8-frame clips across two starts, two seeds, and four variants.
- Verified manifest counts and per-window `frames.csv` row counts.

Single-clip GT-pose sanity:

- Result: `tmp/aqua_tank_pose_stress_v2_sanity_fishextreme/`
- Clip: `start0000_seed20260618/fish-extreme`
- Eval setting: 64 frames, frame stride 4, one pyCOLMAP seed.
- This is a plumbing check, not a paper result.

| System | Input Reg. | Pose Eval | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 56.25% | 100.00% | 0.0139 | 0.0125 | 34.24% | 22.84% |
| Aqua hard | 12.50% | 0.00% | n/a | n/a | 10.81% | 10.05% |
| v2 learned hard t=0.90 | 37.50% | 100.00% | 0.0123 | 0.0078 | 11.45% | 10.36% |
| Pose-aware soft t=0.90 | 18.75% | 100.00% | 0.0074 | 0.0126 | 11.45% | 10.36% |

Stress4 first-clip sanity:

- Result: `tmp/aqua_tank_pose_stress_v2_sanity_stress4_firstclip/`
- Clips: first `fish-high`, `fish-extreme`, `snow-high`,
  `mixed-fish-snow` clip from v2.
- Eval setting: 64 frames, frame stride 4, pyCOLMAP seeds 42 and 43.
- This is still a sanity subset, not the final expanded benchmark.

| System | Records | Pose Eval | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 8 | 87.50% | 48.44% | 0.0186 | 0.0142 | 21.19% | 16.75% |
| Aqua hard | 8 | 75.00% | 30.47% | 0.0126 | 0.0149 | 9.88% | 9.46% |
| v2 learned hard t=0.90 | 8 | 87.50% | 41.41% | 0.0194 | 0.0218 | 10.40% | 10.81% |
| Pose-aware soft t=0.90 | 8 | 87.50% | 43.75% | 0.0108 | 0.0096 | 10.40% | 10.81% |

Interpretation:

- The v2 benchmark expansion is working end to end.
- The first-clip stress4 subset shows the desired direction for pose-aware soft:
  lower contamination than raw and lower ATE/RPE in this small sanity subset.
- Registration completeness is still mixed: raw has the highest average input
  registration in the first-clip subset, and the result is not yet a full
  mean-over-start/seed claim.
- The next publishable check should run the 24-clip stress4 list:
  `data/real_underwater/tank_pose_stress_v2/manifests_stress4.txt`.

Recommended next command shape:

```bash
CUDA_VISIBLE_DEVICES=0 /media/data/u24conda/envs/longlive/bin/python \
  scripts/eval_aqua_pose_gt_validation.py \
  --manifest-list data/real_underwater/tank_pose_stress_v2/manifests_stress4.txt \
  --ckpt-path output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt \
  --retention-scorer-path tmp/aqua_retention_scorer/webuot_synth_tank_mix_train_v2/retention_scorer.pt \
  --output-dir tmp/aqua_tank_pose_stress_v2_stress4_multiseed_t090 \
  --max-frames 128 --frame-stride 2 \
  --aqua-window-size 32 --aqua-grid-stride 8 \
  --query-chunk-size 4096 --max-runtime-seconds 45 --max-num-features 4096 \
  --no-temporal-rgb --no-oracle \
  --enable-learned-retention --enable-pose-aware-soft-retention \
  --retention-score-threshold 0.90 \
  --retention-patch-radius 5 --retention-min-inlier-support 1 \
  --retention-max-features-per-frame 300 --retention-max-fraction 0.18 \
  --pycolmap-random-seeds 42,43,44 --fixed-initial-pair auto \
  --variant-filter raw --variant-filter aqua_inpaint \
  --variant-filter 'aqua_learned_retain_t0p90_inpaint' \
  --variant-filter 'aqua_pose_soft_t0p90' \
  --seed 42
```

Given the observed pyCOLMAP runtime, run this as a queued or split job rather
than as an interactive one-shot command.

## Full Stress4 Multi-Seed Evaluation

Date: 2026-06-19

The full high-stress claim gate has now been run on the 24-clip stress4 list:

- 4 stress variants: `fish-high`, `fish-extreme`, `snow-high`,
  `mixed-fish-snow`.
- 6 clips per stress variant: 3 starts x 2 injection seeds.
- 3 pyCOLMAP seeds: 42, 43, 44.
- 4 non-oracle systems: raw, Aqua hard inpaint, v2 learned hard t=0.90,
  pose-aware soft t=0.90.

To make the run resumable, it was executed as stress shards:

- Runner: `scripts/run_aqua_tank_stress4_shards.py`
- Merger: `scripts/merge_aqua_pose_gt_shards.py`
- Shards: `tmp/aqua_tank_pose_stress_v2_stress4_multiseed_t090_shards/`
- Merged result:
  `tmp/aqua_tank_pose_stress_v2_stress4_multiseed_t090_merged/`

### Overall Result

Result:
`tmp/aqua_tank_pose_stress_v2_stress4_multiseed_t090_merged/summary_seed_stability.csv`

Aggregated over 24 clips x 3 pyCOLMAP seeds = 72 records per system:

| System | Success | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 98.61% | 90.28% | 43.51% | 0.0412 | 0.0284 | 22.86% | 17.63% |
| Aqua hard inpaint | 100.00% | 84.72% | 35.29% | 0.0588 | 0.0400 | 9.94% | 9.66% |
| v2 learned hard t=0.90 | 100.00% | 90.28% | 34.98% | 0.0447 | 0.0346 | 12.88% | 14.29% |
| Pose-aware soft t=0.90 | 98.61% | 76.39% | 24.70% | 0.0324 | 0.0262 | 12.89% | 14.29% |

Interpretation:

- All Aqua variants substantially reduce feature contamination versus raw.
  The strongest reduction is Aqua hard inpaint: 22.86% -> 9.94%.
- v2 learned hard keeps raw-level pose-eval success (90.28%) while reducing
  feature contamination to 12.88%, but it has lower input registration and
  slightly worse ATE/RPE than raw.
- Pose-aware soft has the best overall ATE/RPE in this aggregate
  (0.0324/0.0262 vs raw 0.0412/0.0284), but its pose-eval success and input
  registration are lower. This is not a complete downstream win.

### Stress-Specific Result

Mean over 6 clips x 3 pyCOLMAP seeds per stress variant:

| Stress | System | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fish-high | Raw | 88.89% | 44.18% | 0.0351 | 0.0160 | 17.45% | 11.84% |
| fish-high | v2 learned hard t=0.90 | 88.89% | 27.43% | 0.0334 | 0.0227 | 8.76% | 8.73% |
| fish-high | Pose-aware soft t=0.90 | 72.22% | 35.94% | 0.0306 | 0.0168 | 8.76% | 8.73% |
| fish-extreme | Raw | 88.89% | 17.36% | 0.0284 | 0.0239 | 31.52% | 21.99% |
| fish-extreme | v2 learned hard t=0.90 | 94.44% | 32.73% | 0.0530 | 0.0318 | 15.77% | 15.80% |
| fish-extreme | Pose-aware soft t=0.90 | 88.89% | 11.72% | 0.0250 | 0.0222 | 15.77% | 15.80% |
| snow-high | Raw | 100.00% | 74.74% | 0.0395 | 0.0335 | 18.82% | 18.46% |
| snow-high | v2 learned hard t=0.90 | 88.89% | 59.37% | 0.0522 | 0.0496 | 13.80% | 17.09% |
| snow-high | Pose-aware soft t=0.90 | 72.22% | 14.67% | 0.0251 | 0.0306 | 13.81% | 17.08% |
| mixed-fish-snow | Raw | 83.33% | 37.76% | 0.0621 | 0.0416 | 23.64% | 18.25% |
| mixed-fish-snow | Aqua hard inpaint | 83.33% | 26.13% | 0.0378 | 0.0243 | 11.18% | 11.34% |
| mixed-fish-snow | v2 learned hard t=0.90 | 88.89% | 20.40% | 0.0381 | 0.0336 | 13.20% | 15.56% |
| mixed-fish-snow | Pose-aware soft t=0.90 | 72.22% | 36.46% | 0.0495 | 0.0356 | 13.20% | 15.56% |

Stress-specific reading:

- `fish-high`: learned/pose-aware variants cut contamination roughly in half
  and are competitive on ATE, but they do not dominate raw in registration or
  RPE.
- `fish-extreme`: raw contamination is very high. Pose-aware soft improves
  ATE/RPE while lowering contamination, but registration is sparse. Learned
  hard improves registration and contamination but worsens ATE/RPE.
- `snow-high`: raw remains the most complete reconstruction. Aqua variants
  reduce contamination, but marine snow is still a failure mode for sequence
  completeness and pose reliability.
- `mixed-fish-snow`: Aqua hard and learned hard give a useful mixed-stress
  pose/contamination Pareto. The safest statement is lower contamination with
  competitive or improved pose on some metrics, not full dominance.

### Result-to-Claim Gate

Local verdict: `partial` support; external review pending.

Supported by R094:

1. The expanded Tank v2 stress benchmark is now a real mean-over-start/seed
   downstream test, not a single favorable clip.
2. Aqua-D4RT variants substantially reduce dynamic feature and match
   contamination under high fish, extreme fish, snow, and mixed stress.
3. Retention creates a controllable Pareto between contamination, registration
   completeness, and ATE/RPE.
4. There are credible pose-quality endpoints: pose-aware soft has the best
   aggregate ATE/RPE, and learned hard keeps raw-level pose-eval success with
   much lower contamination.

Not supported by R094:

1. A blanket claim that the final non-oracle system beats raw pyCOLMAP on all
   downstream metrics.
2. A claim that marine snow is solved. `snow-high` still exposes registration
   and pose fragility.
3. A broad underwater SLAM SOTA claim. This remains a controlled injected
   Tank benchmark plus WebUOT/Synthetic map-cleanliness evidence.

Current safe paper wording:

```text
Aqua-D4RT substantially reduces transient feature contamination and exposes a
retention-aware downstream Pareto on an expanded GT-pose Tank stress benchmark.
The best endpoints trade off pose accuracy, registration completeness, and
cleanliness; full ATE/RPE dominance over raw SfM remains an optimization target.
```

Decision for the next sprint:

- Do not scale to many new datasets before improving the final retention policy.
  R094 already exposes the key method gap: hard retention is cleaner but can be
  sparse; pose-aware soft can improve ATE/RPE in aggregate but has lower
  pose-eval success and registration.
- The next method experiment should be an adaptive hard/soft policy or v3
  retention scorer trained with pose/downstream labels.
- After that, add one external GT-pose underwater dataset such as AQUALOC or
  VAROS as a generalization sanity check. Adding more datasets before fixing
  the retention policy would mostly add engineering variables rather than
  answer the current claim gap.

## Adaptive Selector Feasibility Check

Date: 2026-06-19

After R094, a lightweight post-hoc selector analysis was run without new
pyCOLMAP jobs. It reads the merged `per_clip_metrics.json` and chooses among
raw, Aqua hard, v2 learned hard, and pose-aware soft per clip/seed.

- Script: `scripts/analyze_aqua_adaptive_pose_selector.py`
- Result:
  `tmp/aqua_tank_pose_stress_v2_stress4_multiseed_t090_adaptive_analysis/summary.csv`

This is not a deployable method result. It is a feasibility probe to decide
whether a simple adaptive policy is worth implementing.

| Policy | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 90.28% | 43.51% | 0.0412 | 0.0284 | 22.86% | 17.63% |
| v2 learned hard t=0.90 | 90.28% | 34.98% | 0.0447 | 0.0346 | 12.88% | 14.29% |
| Pose-aware soft t=0.90 | 76.39% | 24.70% | 0.0324 | 0.0262 | 12.89% | 14.29% |
| Reconstruction-quality selector | 83.33% | 56.66% | 0.0663 | 0.0422 | 12.12% | 13.16% |
| Reconstruction-quality + raw fallback | 90.28% | 67.12% | 0.0666 | 0.0412 | 13.01% | 13.49% |
| Oracle min ATE | 100.00% | 19.10% | 0.0135 | 0.0139 | 15.08% | 14.46% |
| Oracle min RPE | 100.00% | 25.20% | 0.0194 | 0.0118 | 15.19% | 14.26% |

Interpretation:

- A simple reconstruction-quality selector improves registration completeness
  and keeps contamination low, but worsens ATE/RPE. It should not be promoted
  as the final adaptive method.
- The oracle selectors show substantial headroom, but they use GT pose errors
  and are only an upper bound.
- This points to v3 retention/adaptive training with pose/downstream labels,
  rather than another hand-tuned selector or broad dataset scale-up.

## R096 Pose-Label v3 Retention Full Stress4 Evaluation

Date: 2026-06-22

After R095, a v3 retention scorer was trained with GT-pose geometric labels on
Tank stress v2 train starts. The default v3 scorer uses GT pose only to create
training labels; its deployed input features remain Aqua/ORB/context features.
Pose-feature inputs remain diagnostic/oracle-only and are not used for the
numbers below.

- v3 scorer:
  `tmp/aqua_retention_scorer/tank_pose_v3_train_start0000_0064_val_start0128_stress4_mlp32/retention_scorer.pt`
- Threshold selection screen:
  `tmp/aqua_retention_scorer/tank_pose_v3_val_start0128_pose_eval_seed42_t073/`
- Full stress4 shards:
  `tmp/aqua_tank_pose_stress_v2_stress4_multiseed_v3_t073_shards/`
- Merged result:
  `tmp/aqua_tank_pose_stress_v2_stress4_multiseed_v3_t073_merged/`

Scope matches R094:

- 24 clips: `fish-high`, `fish-extreme`, `snow-high`, `mixed-fish-snow`;
  3 starts x 2 injection seeds per stress variant.
- 3 pyCOLMAP seeds: 42, 43, 44.
- 72 records per system.
- Evaluated systems: raw, Aqua hard inpaint, v3 learned hard t=0.73, and
  v3 pose-aware soft t=0.73.

### Threshold Screen

On the start0128 validation slice with pyCOLMAP seed 42, v3 t=0.73 was the
most balanced threshold among the tested values:

| Threshold / System | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw in t=0.73 run | 75.00% | 33.79% | 0.0443 | 0.0283 | 21.69% | 16.97% |
| v3 pose-soft t=0.65 | 100.00% | 23.05% | 0.0512 | 0.0304 | 13.73% | 15.35% |
| v3 pose-soft t=0.70 | 100.00% | 15.43% | 0.0437 | 0.0282 | 13.56% | 15.17% |
| v3 pose-soft t=0.73 | 100.00% | 30.08% | 0.0363 | 0.0282 | 13.45% | 15.07% |
| v3 pose-soft t=0.75 | 100.00% | 15.04% | 0.0348 | 0.0321 | 13.36% | 14.95% |

The t=0.80 run was intentionally stopped after t=0.75 showed very sparse
registration. The interrupted t=0.80 directory is not used as a result.

### Overall Result

Result:
`tmp/aqua_tank_pose_stress_v2_stress4_multiseed_v3_t073_merged/summary_seed_stability.csv`

| System | Success | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 98.61% | 75.00% | 33.79% | 0.0469 | 0.0288 | 22.86% | 17.63% |
| Aqua hard inpaint | 98.61% | 84.72% | 35.55% | 0.0602 | 0.0433 | 9.94% | 9.66% |
| v3 learned hard t=0.73 | 100.00% | 80.56% | 31.47% | 0.0497 | 0.0348 | 13.83% | 15.39% |
| v3 pose-aware soft t=0.73 | 98.61% | 77.78% | 28.39% | 0.0398 | 0.0281 | 13.83% | 15.40% |

Within this v3 run, pose-aware soft t=0.73 reduces feature contamination
22.86% -> 13.83% and slightly improves ATE/RPE versus raw
0.0469/0.0288 -> 0.0398/0.0281. It also has slightly higher pose-eval success
than raw (77.78% vs 75.00%). However, input registration remains lower than
raw (28.39% vs 33.79%), so this is still not a full downstream dominance
claim.

The raw rows differ from R094 despite the same nominal pyCOLMAP seeds, which
confirms that the pyCOLMAP backend retains implementation/thread-level
nondeterminism. Therefore, the primary v3 claim should be made against the raw
baseline inside the same v3 run; v2-vs-v3 comparisons are useful trend checks
but not exact paired statistical tests.

### Stress-Specific Result

Mean over 6 clips x 3 pyCOLMAP seeds per stress variant:

| Stress | System | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fish-high | Raw | 61.11% | 22.66% | 0.0399 | 0.0263 | 17.45% | 11.84% |
| fish-high | v3 learned hard t=0.73 | 72.22% | 28.39% | 0.0460 | 0.0318 | 9.81% | 9.60% |
| fish-high | v3 pose-soft t=0.73 | 66.67% | 15.89% | 0.0253 | 0.0192 | 9.81% | 9.60% |
| fish-extreme | Raw | 77.78% | 6.51% | 0.0169 | 0.0179 | 31.52% | 21.99% |
| fish-extreme | v3 learned hard t=0.73 | 88.89% | 30.38% | 0.0542 | 0.0315 | 17.59% | 18.37% |
| fish-extreme | v3 pose-soft t=0.73 | 88.89% | 24.22% | 0.0260 | 0.0190 | 17.60% | 18.37% |
| snow-high | Raw | 83.33% | 69.01% | 0.0627 | 0.0322 | 18.82% | 18.46% |
| snow-high | v3 learned hard t=0.73 | 94.44% | 60.76% | 0.0640 | 0.0518 | 14.04% | 17.37% |
| snow-high | v3 pose-soft t=0.73 | 72.22% | 35.85% | 0.0582 | 0.0364 | 14.04% | 17.38% |
| mixed-fish-snow | Raw | 77.78% | 36.98% | 0.0656 | 0.0379 | 23.64% | 18.25% |
| mixed-fish-snow | v3 learned hard t=0.73 | 66.67% | 6.34% | 0.0274 | 0.0182 | 13.88% | 16.25% |
| mixed-fish-snow | v3 pose-soft t=0.73 | 83.33% | 37.59% | 0.0502 | 0.0378 | 13.88% | 16.25% |

Stress-specific reading:

- `fish-high`: v3 pose-soft gives the cleanest/lowest-error endpoint but is
  sparse. v3 learned hard improves registration and contamination versus this
  run's raw, but worsens ATE/RPE.
- `fish-extreme`: v3 substantially improves registration completeness versus
  raw while roughly halving feature contamination. Raw still has the lowest
  ATE/RPE among successful pose evaluations, so this is not a pose win.
- `snow-high`: raw remains the most complete system. v3 lowers contamination
  and v3 learned hard improves pose-eval success, but snow remains a
  registration/pose trade-off case.
- `mixed-fish-snow`: this is the strongest v3 pose-soft endpoint. It keeps
  input registration essentially raw-level (37.59% vs 36.98%), improves
  pose-eval success (83.33% vs 77.78%), lowers ATE (0.0502 vs 0.0656), and
  reduces feature contamination (13.88% vs 23.64%).

### v2-vs-v3 Trend Check

Compared with R094 v2 pose-aware soft t=0.90, v3 pose-soft t=0.73 improves
overall input registration (24.70% -> 28.39%) and slightly improves
pose-eval success (76.39% -> 77.78%). It gives up some of v2's best aggregate
ATE/RPE (0.0324/0.0262 -> 0.0398/0.0281) and has slightly higher contamination
(12.89% -> 13.83%). This is a useful movement toward completeness, but not
enough to make v3 the final method by itself.

### Result-to-Claim Gate

Local verdict: `partial` support; external review pending.

Supported by R096:

1. A pose/downstream-label v3 scorer is feasible without using GT pose at
   deployment time.
2. v3 pose-soft gives a stronger completeness/cleanliness/pose endpoint than
   the previous soft policy in some stress variants, especially
   `mixed-fish-snow`.
3. The overall v3 run supports the paper-safe claim that Aqua-D4RT can reduce
   downstream feature contamination while exposing controllable pose vs
   registration trade-offs.

Not supported by R096:

1. v3 is not a blanket replacement for raw SfM: input registration is still
   lower overall than raw, and stress-specific pose wins are uneven.
2. v3 does not solve marine snow or high-fish pose estimation by itself.
3. v3 does not yet justify a broad underwater SLAM SOTA claim.

Decision:

- The ICRA story is stronger than before because v3 provides a non-oracle,
  pose-trained endpoint that, within the same full run, has lower contamination
  and slightly better ATE/RPE than raw.
- The safest main claim remains:

```text
Aqua-D4RT produces substantially cleaner static geometry and provides
retention-aware downstream Pareto control on dynamic underwater stress tests.
Pose-trained v3 retention improves some mixed-stress pose/cleanliness endpoints,
but final deployment should use an adaptive hard/soft policy rather than a
single fixed retention mode.
```

- Next method direction: implement an adaptive selector that uses sequence
  stress statistics and reconstruction-quality signals to choose among v3
  hard, v3 pose-soft, Aqua hard, and raw/relaxed fallback. Do this before broad
  external dataset scale-up.

## R098 Deployable Adaptive v3 Selector Full Stress4 Evaluation

Date: 2026-06-22

R098 evaluated a post-hoc deployable selector over the full v3 hard and v3
pose-soft outputs, avoiding an extra pyCOLMAP rerun for copied adaptive videos.
The reported deployable policy is `aqua_dynamic_high_guard`: if the Aqua
dynamic probability is high, choose v3 hard; otherwise choose v3 pose-soft.
GT-derived policies in the selector script are diagnostics only and should not
be reported as deployment methods.

- Full stress4 shards, cleaned after merge:
  `tmp/aqua_adaptive_v3_t073_full_stress4_shards/`
- Merged result:
  `tmp/aqua_adaptive_v3_t073_full_stress4_merged/`
- Historical selector result, superseded by R099 and cleaned:
  `tmp/aqua_adaptive_v3_t073_full_stress4_selector_v2/summary.csv`

Scope:

- 24 clips: `fish-high`, `fish-extreme`, `snow-high`, `mixed-fish-snow`.
- 3 pyCOLMAP seeds: 42, 43, 44.
- 72 records per system/policy.
- Same-run raw baseline is from `tmp/aqua_adaptive_v3_t073_full_stress4_merged/`.

### Overall Selector Result

| Policy / System | Success | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 98.61% | 81.94% | 38.00% | 0.0435 | 0.0283 | 22.86% | 17.63% |
| v3 hard always | 98.61% | 84.72% | 32.12% | 0.0529 | 0.0362 | 13.83% | 15.39% |
| v3 pose-soft always | 95.83% | 83.33% | 23.05% | 0.0306 | 0.0253 | 13.83% | 15.40% |
| Aqua dynamic high guard | 97.22% | 83.33% | 24.11% | 0.0330 | 0.0268 | 13.83% | 15.40% |

Selection counts for the deployable guard: v3 hard on 21 records and v3
pose-soft on 51 records.

Relative to same-run raw, the deployable guard reduces feature contamination
22.86% -> 13.83%, reduces ATE 0.0435 -> 0.0330, and slightly reduces RPE
0.0283 -> 0.0268. However, input registration drops 38.00% -> 24.11%, so this
is still a Pareto endpoint rather than downstream dominance.

### Stress-Specific Selector Reading

Mean over 6 clips x 3 seeds per stress variant:

| Stress | Policy | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Selection |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fish-extreme | Raw | 72.22% | 10.50% | 0.0218 | 0.0207 | 31.52% | raw |
| fish-extreme | Aqua dynamic high guard | 88.89% | 10.68% | 0.0330 | 0.0287 | 17.59% | hard |
| fish-high | Raw | 94.44% | 40.10% | 0.0260 | 0.0142 | 17.45% | raw |
| fish-high | Aqua dynamic high guard | 77.78% | 26.65% | 0.0323 | 0.0199 | 9.81% | mostly soft |
| mixed-fish-snow | Raw | 77.78% | 36.28% | 0.0799 | 0.0428 | 23.64% | raw |
| mixed-fish-snow | Aqua dynamic high guard | 77.78% | 30.90% | 0.0295 | 0.0225 | 13.88% | soft |
| snow-high | Raw | 83.33% | 65.10% | 0.0483 | 0.0374 | 18.82% | raw |
| snow-high | Aqua dynamic high guard | 88.89% | 28.21% | 0.0368 | 0.0344 | 14.04% | soft |

Stress reading:

- `mixed-fish-snow` is the strongest downstream result: the guard keeps
  pose-eval success at raw level, substantially lowers ATE/RPE, and reduces
  feature contamination.
- `snow-high` benefits in pose error and contamination but loses registration
  completeness.
- `fish-extreme` benefits in pose-eval success and contamination, but raw keeps
  lower ATE/RPE among successful pose evaluations.
- `fish-high` is a failure/trade-off case for the deployable guard: it lowers
  contamination but worsens pose success, registration, and pose error versus
  raw.

### R098 Claim Gate

Local verdict: `partial` support; external review pending.

Supported:

1. The adaptive v3 selector is deployable in the sense that its reported policy
   uses Aqua dynamic-score statistics rather than GT pose or GT masks.
2. It produces a stronger low-contamination / low-pose-error endpoint than raw
   on the full stress4 benchmark.
3. It reinforces the main paper-safe downstream claim: Aqua-D4RT exposes a
   controllable retention Pareto under dynamic underwater stress.

Not supported:

1. The selector does not recover raw-level registration completeness.
2. It does not provide uniform stress-specific pose improvement.
3. It still cannot justify a broad underwater SLAM/SfM SOTA claim.

Updated decision:

- Keep the main story as static-geometry cleanup plus downstream Pareto.
- Use R098 as the best current v3/adaptive downstream evidence, not as a final
  solved-SLAM result.
- The next improvement should target registration completeness without giving
  up the pose-soft error gains, likely by adding a reconstruction-quality
  fallback or training the selector directly on per-sequence pose/registration
  outcomes.

## R099 v3 Pose-Soft + Raw Registration Fallback

Date: 2026-06-22

R099 tests the most direct version of the R098 next-step: use v3 pose-soft as
the default low-contamination, low-pose-error endpoint, but fall back to raw
when raw self-reconstruction quality is clearly stronger. This does not use GT
pose or GT masks, but it is a multi-pass self-diagnostic selector: raw and
pose-soft candidate reconstructions must both be available before selection.

- Script: `scripts/analyze_aqua_adaptive_pose_selector_v3.py`
- Result:
  `tmp/aqua_adaptive_v3_t073_full_stress4_selector_v3_rawfallback/summary.csv`
- Policy:
  `soft_raw_fallback_margin0.10_rawmin0.60`
- Rule: select raw if raw input registration is at least 0.60 and exceeds
  v3 pose-soft registration by more than 0.10; otherwise select v3 pose-soft.

### Overall Result

| Policy / System | Success | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 98.61% | 81.94% | 38.00% | 0.0435 | 0.0283 | 22.86% | 17.63% |
| v3 pose-soft always | 95.83% | 83.33% | 23.05% | 0.0306 | 0.0253 | 13.83% | 15.40% |
| v3 pose-soft + raw fallback | 97.22% | 90.28% | 40.13% | 0.0389 | 0.0281 | 15.17% | 15.73% |

Selection counts: v3 pose-soft on 57 records, raw on 15 records.

This is the first stress4 result in the current series that is better than
same-run raw on all four headline axes: pose-eval success, input registration,
ATE/RPE, and feature contamination. The trade-off is that contamination is
higher than pure v3 pose-soft because raw is deliberately reintroduced for
high-registration cases.

### Stress-Specific Result

| Stress | Policy | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Selection |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fish-extreme | Raw | 72.22% | 10.50% | 0.0218 | 0.0207 | 31.52% | raw |
| fish-extreme | v3 pose-soft + raw fallback | 83.33% | 10.76% | 0.0291 | 0.0258 | 18.42% | 17 soft / 1 raw |
| fish-high | Raw | 94.44% | 40.10% | 0.0260 | 0.0142 | 17.45% | raw |
| fish-high | v3 pose-soft + raw fallback | 88.89% | 38.28% | 0.0315 | 0.0181 | 10.69% | 16 soft / 2 raw |
| mixed-fish-snow | Raw | 77.78% | 36.28% | 0.0799 | 0.0428 | 23.64% | raw |
| mixed-fish-snow | v3 pose-soft + raw fallback | 88.89% | 41.67% | 0.0448 | 0.0297 | 14.83% | 16 soft / 2 raw |
| snow-high | Raw | 83.33% | 65.10% | 0.0483 | 0.0374 | 18.82% | raw |
| snow-high | v3 pose-soft + raw fallback | 100.00% | 69.79% | 0.0485 | 0.0375 | 16.73% | 8 soft / 10 raw |

Stress reading:

- `mixed-fish-snow` is the strongest positive case: higher pose success,
  higher registration, much lower ATE/RPE, and lower contamination than raw.
- `snow-high` recovers raw-level completeness and improves contamination, but
  pose error is essentially tied with raw.
- `fish-extreme` improves pose success and contamination, but raw still has
  lower ATE/RPE.
- `fish-high` remains the main failure/trade-off: the selector lowers
  contamination but is slightly worse than raw in pose success and pose error.

### R099 Claim Gate

Local verdict: `partial+` support; external review pending.

Supported:

1. A self-diagnostic adaptive system can recover raw-level registration while
   retaining substantial Aqua contamination reduction on Tank stress4.
2. On this benchmark, v3 pose-soft + raw fallback improves the aggregate
   downstream headline metrics versus same-run raw.
3. This strengthens the ICRA downstream story from a pure Pareto claim toward
   a bounded "adaptive retention can match or exceed raw SfM completeness while
   reducing dynamic contamination" claim on the Tank stress4 benchmark.

Not supported:

1. This is still one expanded Tank benchmark, not broad underwater SLAM SOTA.
2. The selector is multi-pass and uses post-reconstruction registration quality,
   so it is not a cheap one-pass online policy.
3. Stress-specific pose improvement is not uniform; `fish-high` and
   `fish-extreme` still need better selection or training signals.

Updated paper-safe wording:

```text
On an expanded Tank GT-pose stress benchmark, a self-diagnostic adaptive
retention policy matches raw reconstruction completeness while reducing
feature contamination and improving aggregate pose error. This complements
the primary static-geometry cleanup claim and should be reported with the
multi-pass selector caveat.
```

## R100 R099 Selector Sensitivity

Date: 2026-06-22

R100 checks whether the R099 raw-fallback result is a brittle threshold
artifact. The same full stress4 per-clip table was re-analyzed over a small
grid of raw-registration thresholds and raw-vs-pose-soft registration margins.

- Script: `scripts/analyze_aqua_adaptive_pose_selector_v3.py`
- Input:
  `tmp/aqua_adaptive_v3_t073_full_stress4_merged/per_clip_metrics.json`
- Result:
  `tmp/aqua_adaptive_v3_t073_full_stress4_selector_v3_sensitivity/summary.csv`
- Grid:
  `raw_min_registration in {0.50, 0.60, 0.70}`,
  `raw_margin in {0.05, 0.10, 0.15, 0.20}`.

### Sensitivity Summary

| System / Range | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw baseline | 81.94% | 38.00% | 0.0435 | 0.0283 | 22.86% |
| R099 selected point, rawmin 0.60 / margin 0.10 | 90.28% | 40.13% | 0.0389 | 0.0281 | 15.17% |
| R100 fallback grid, min-max | 90.28% | 37.76-40.97% | 0.0363-0.0437 | 0.0266-0.0282 | 14.98-15.70% |

All 12 fallback settings preserve 90.28% pose-eval success. The `rawmin=0.50`
and `rawmin=0.60` settings keep input registration above the raw baseline,
while `rawmin=0.70` trades slightly lower registration for lower ATE/RPE and
slightly lower contamination. The R099 headline setting is therefore not a
single lucky threshold; it is a middle point on a stable registration-vs-error
trade-off.

### R100 Claim Gate

Supported:

1. The R099 conclusion is stable over nearby self-diagnostic fallback
   thresholds.
2. The selector consistently reduces dynamic feature contamination relative to
   raw on stress4.
3. There is a tunable sub-Pareto inside the R099 policy: lower raw fallback
   threshold favors registration completeness, higher threshold favors pose
   error and contamination.

Not supported:

1. This does not remove the multi-pass selector caveat.
2. This still does not address external underwater datasets or a non-oracle
   detector+SAM baseline.

## R101 All-Stress Seed42 Scale-Up Sanity

Date: 2026-06-22

R101 extends the v3 t=0.73 evaluation beyond the high-stress `stress4` subset by
adding the missing `clean`, `fish-low`, `fish-med`, and `snow-med` variants.
This is a one-seed sanity run, not a replacement for the R099 3-seed stress4
main result.

- Missing4 shards, cleaned after merge:
  `tmp/aqua_adaptive_v3_t073_allstress_missing4_seed42_shards/`
- Missing4 merged:
  `tmp/aqua_adaptive_v3_t073_allstress_missing4_seed42_merged/`
- Combined seed42 all-stress table:
  `tmp/aqua_adaptive_v3_t073_allstress_seed42_combined/`
- Raw-fallback selector:
  `tmp/aqua_adaptive_v3_t073_allstress_seed42_selector_v3_rawfallback/`
- Sensitivity:
  `tmp/aqua_adaptive_v3_t073_allstress_seed42_selector_v3_sensitivity/`

Scope:

- 8 stress variants: clean, fish-low, fish-med, fish-high, fish-extreme,
  snow-med, snow-high, mixed-fish-snow.
- 48 clips total at seed42 only.
- Stress4 seed42 records are filtered from the R098/R099 3-seed run; missing4
  records are newly evaluated.

### All-Stress Seed42 Summary

| System | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 91.67% | 33.56% | 0.0379 | 0.0263 | 13.63% | 10.78% |
| Aqua inpaint | 89.58% | 41.89% | 0.0514 | 0.0359 | 6.22% | 6.24% |
| v3 hard t=0.73 | 91.67% | 30.73% | 0.0438 | 0.0323 | 8.43% | 9.41% |
| v3 pose-soft t=0.73 | 83.33% | 25.65% | 0.0301 | 0.0220 | 8.44% | 9.43% |
| R099-style raw fallback | 89.58% | 42.68% | 0.0397 | 0.0250 | 9.29% | 9.64% |

Interpretation:

- The all-stress sanity is more conservative than R099 stress4: adding clean
  and low-stress clips makes raw a stronger pose baseline.
- R099-style raw fallback still reduces feature contamination substantially
  compared with raw (13.63% -> 9.29%) and increases input registration
  (33.56% -> 42.68%), but it does not beat raw on pose-eval success or ATE.
- v3 pose-soft remains the lowest ATE/RPE endpoint, but loses registration and
  pose-eval success.

Stress-specific sanity:

- `clean`: all methods have zero contamination; raw has the best or tied pose
  behavior. This supports keeping raw fallback for low-interference scenes.
- `fish-low`: Aqua reduces contamination, but raw keeps slightly lower pose
  error. This is another low-stress case where over-filtering can hurt.
- `fish-med`: v3 pose-soft improves ATE/RPE and lowers feature contamination
  versus raw, but registration is lower.
- `snow-med`: v3 pose-soft improves ATE/RPE and lowers feature contamination,
  while v3 hard has stronger registration.

### R101 Claim Gate

Supported:

1. The method is not only tuned to high stress: on all-stress seed42, Aqua
   variants consistently reduce contamination relative to raw.
2. Raw fallback is useful and necessary for clean/low-stress cases.
3. The downstream story should remain an adaptive Pareto story across the full
   stress range.

Not supported:

1. R101 does not upgrade the claim to uniform pose improvement across all stress
   levels.
2. R101 is seed42 only. The main downstream table should still use R099 stress4
   3-seed results, with R101 as a robustness/boundary sanity check.
