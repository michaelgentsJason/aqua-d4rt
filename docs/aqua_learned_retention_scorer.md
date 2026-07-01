# Aqua-D4RT Learned Retention Scorer

Date: 2026-06-16

## Goal

The previous SLAM-aware retention baseline used a hand-tuned rule: re-admit
patches around Aqua-rejected keypoints if they appear in adjacent-frame
essential-matrix RANSAC inliers. That recovered ORB front-end success, but also
re-admitted dynamic features.

This experiment turns retention into a lightweight trainable scoring module.
It still starts from Aqua's rejected region, but each candidate keypoint gets a
score from Aqua probabilities, ORB response, RANSAC support, and temporal
appearance consistency.

Ground-truth masks are used only for training labels and evaluation metrics.
WebUOT labels are tracked-target bbox masks, not complete fish instance masks.

## Implemented

New scripts:

- `scripts/aqua_retention_utils.py`
- `scripts/build_aqua_retention_training_data.py`
- `scripts/train_aqua_retention_scorer.py`
- `scripts/eval_aqua_retention_pareto.py`

Extended script:

- `scripts/eval_aqua_pycolmap_validation.py`
  - Added `--retention-scorer-path`
  - Added `--retention-score-threshold`

## Training Data

Training set:

- WebUOT fish30 train24
- Aqua checkpoint: `output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt`
- Aqua grid stride: 4
- ORB max features: 1200
- Static threshold: 0.55

Result:

- `tmp/aqua_retention_scorer/webuot_train24_grid4_orb/retention_candidates.npz`
- 159,779 candidate keypoints in Aqua-rejected regions
- Positive pseudo-label rate: 31.60%
- GT transient rate among candidates: 21.51%
- Geometry-stable rate: 37.90%

Positive label definition:

```text
GT-static keypoint inside the Aqua-rejected region
and adjacent-frame essential-matrix inlier support >= 1
```

## Scorer

Model:

- Small MLP, 17 input features, hidden dim 32
- Features include Aqua dynamic/particle/static scores, ORB response/rank,
  match support, inlier support, inlier ratio, match distance, flow, and patch NCC.

Result:

- `tmp/aqua_retention_scorer/webuot_train24_grid4_orb_train/retention_scorer.pt`
- Internal clip-split val best pseudo-label F1: 0.960 at threshold 0.65

This pseudo-label F1 is not a paper metric. The meaningful metrics are the
downstream contamination and registration trade-offs below.

## ORB Proxy: WebUOT Fish30 All30

Result: `tmp/aqua_retention_scorer/webuot_all30_grid4_orb_pareto/aggregate_metrics.json`

| Variant | Feature Contam. | Match Contam. | E Success | Feat/Frame |
| --- | ---: | ---: | ---: | ---: |
| Raw | 20.97% | 25.42% | 96.67% | 582.8 |
| Aqua clean 0.55 | 3.75% | 4.84% | 63.75% | 235.0 |
| Rule SLAM retain | 10.38% | 21.88% | 95.00% | 311.3 |
| Learned retain t=0.50 | 10.09% | 21.68% | 94.58% | 310.1 |
| Learned retain t=0.65 | 9.55% | 20.66% | 92.71% | 307.5 |
| Learned retain t=0.80 | 8.12% | 18.45% | 91.67% | 301.9 |
| Temporal RGB | 16.32% | 23.22% | 92.50% | 484.5 |
| Oracle GT static | 0.00% | 0.00% | 84.58% | 472.6 |

Takeaway: learned retention creates a useful Pareto curve. Compared with
rule-only retention, threshold 0.80 reduces feature contamination by 2.26
points and match contamination by 3.43 points, while keeping E success above
91%.

## ORB Proxy: WebUOT Fish30 Val6

Result: `tmp/aqua_retention_scorer/webuot_val6_grid4_orb_pareto/aggregate_metrics.json`

| Variant | Feature Contam. | Match Contam. | E Success |
| --- | ---: | ---: | ---: |
| Raw | 49.01% | 39.06% | 98.96% |
| Aqua clean 0.55 | 8.24% | 14.26% | 47.92% |
| Rule SLAM retain | 22.17% | 38.11% | 98.96% |
| Learned retain t=0.65 | 21.60% | 37.83% | 97.92% |
| Learned retain t=0.80 | 17.90% | 35.48% | 95.83% |

Val6 is harder than all30 by tracked-bbox coverage. The learned scorer still
trades a small amount of E success for lower contamination.

## Synthetic Test5 Cross-Domain Check

Result: `tmp/aqua_retention_scorer/synth_test5_grid4_orb_pareto/aggregate_metrics.json`

| Variant | Feature Contam. | Match Contam. | E Success |
| --- | ---: | ---: | ---: |
| Raw | 84.07% | 69.12% | 67.50% |
| Aqua clean 0.55 | 50.18% | 28.92% | 8.75% |
| Rule SLAM retain | 60.74% | 59.61% | 63.75% |
| Learned retain t=0.50 | 58.38% | 55.68% | 58.75% |
| Learned retain t=0.80 | 51.31% | 41.14% | 18.75% |

Takeaway: a WebUOT-trained scorer does not transfer cleanly to synthetic
composited fish/particles. It still exposes a contamination/success curve, but
high thresholds become too conservative.

## pyCOLMAP Val6

Conservative learned retention was not enough to recover pyCOLMAP registration.

Aggressive learned result:

- `tmp/aqua_retention_scorer/pycolmap_webuot_val6_learned_t050_aggressive/aggregate_metrics.json`

| Variant | Success | Reg. Rate | Points3D | Reproj. |
| --- | ---: | ---: | ---: | ---: |
| Raw | 100.00% | 100.00% | 29.0 | 0.175 |
| Temporal RGB inpaint | 100.00% | 100.00% | 127.5 | 0.328 |
| Aqua inpaint | 50.00% | 50.00% | 87.3 | 0.114 |
| Rule aggressive retain | 83.33% | 83.33% | 88.0 | 0.171 |
| Learned aggressive retain t=0.50 | 66.67% | 66.67% | 79.8 | 0.194 |
| Oracle GT inpaint | 66.67% | 66.67% | 184.5 | 0.165 |

Takeaway: learned retention is promising in ORB proxy metrics, but the current
image inpainting/pyCOLMAP path still favors the simpler aggressive rule. This
should be reported as a limitation, not as a solved SLAM result.

## Follow-Up Optimization: WebUOT + Synthetic Mix Scorer

The first learned scorer was trained only on WebUOT train24. To improve
robustness, a second scorer mixes WebUOT train24 candidates with 10 synthetic
training clips.

Additional training data:

- `tmp/aqua_retention_scorer/synth_train10_grid4_orb/retention_candidates.npz`
- 65,286 synthetic candidates
- Positive pseudo-label rate: 0.32%
- GT transient rate among candidates: 91.42%

Mixed scorer:

- `tmp/aqua_retention_scorer/webuot_synth_mix_train24_10_grid4_orb_train/retention_scorer.pt`
- Internal mixed clip-split val pseudo-label F1: 0.940 at threshold 0.60

### ORB Proxy: WebUOT Fish30 All30

Result: `tmp/aqua_retention_scorer/webuot_all30_grid4_orb_pareto_mix_scorer/aggregate_metrics.json`

| Variant | Feature Contam. | Match Contam. | E Success | Feat/Frame |
| --- | ---: | ---: | ---: | ---: |
| Raw | 20.97% | 25.42% | 96.67% | 582.8 |
| Aqua clean 0.55 | 3.75% | 4.84% | 63.75% | 235.0 |
| Rule SLAM retain | 10.38% | 21.88% | 95.00% | 311.3 |
| Mix learned t=0.50 | 9.79% | 21.03% | 93.96% | 306.4 |
| Mix learned t=0.80 | 8.35% | 17.00% | 91.25% | 300.4 |

Compared with the WebUOT-only learned scorer, the mixed scorer slightly improves
match contamination on WebUOT all30 at high threshold: 18.45% -> 17.00%.

### Synthetic Test5

Result: `tmp/aqua_retention_scorer/synth_test5_grid4_orb_pareto_mix_scorer/aggregate_metrics.json`

| Variant | Feature Contam. | Match Contam. | E Success |
| --- | ---: | ---: | ---: |
| Rule SLAM retain | 60.74% | 59.61% | 63.75% |
| Mix learned t=0.30 | 59.39% | 57.78% | 61.25% |
| Mix learned t=0.50 | 52.89% | 48.64% | 32.50% |
| Mix learned t=0.80 | 49.86% | 30.30% | 15.00% |

The mixed scorer is still conservative on synthetic test clips. It improves
contamination, but high thresholds remove too many useful features.

### pyCOLMAP Val6

Result: `tmp/aqua_retention_scorer/pycolmap_webuot_val6_mix_t050_aggressive/aggregate_metrics.json`

| Variant | Success | Reg. Rate | Points3D | Reproj. |
| --- | ---: | ---: | ---: | ---: |
| Raw | 100.00% | 100.00% | 77.8 | 0.291 |
| Aqua inpaint | 50.00% | 50.00% | 36.7 | 0.009 |
| Rule aggressive retain | 83.33% | 83.33% | 79.3 | 0.159 |
| Mix learned aggressive t=0.50 | 83.33% | 83.33% | 174.3 | 0.167 |
| Oracle GT inpaint | 66.67% | 66.67% | 159.8 | 0.157 |

This is the strongest learned-retention downstream result so far: the mixed
scorer matches rule aggressive retention success and yields higher sparse-point
density on successful val6 clips. The raw pipeline still has the highest
registration success, but it also keeps dynamic contamination.

## Follow-Up Optimization: Tank-Aware v2 Scorer

After the GT-pose Tank stress benchmark was available, the scorer was retrained
with Tank stress candidate windows in addition to WebUOT and synthetic clips.

Additional Tank split files:

- `data/real_underwater/tank_pose_stress/splits/train_windows.txt`
- `data/real_underwater/tank_pose_stress/splits/val_windows.txt`

Tank candidate data:

| Split | Candidates | Positive Rate | GT Transient Rate | Stable Rate |
| --- | ---: | ---: | ---: | ---: |
| Tank stress train windows | 53,352 | 18.63% | 36.10% | 22.88% |
| Tank stress val windows | 16,864 | 11.78% | 44.95% | 16.05% |

Scorer:

- Train NPZs: WebUOT train24 + synthetic train10 + Tank stress train windows.
- External validation: Tank stress val windows.
- Output: `tmp/aqua_retention_scorer/webuot_synth_tank_mix_train_v2/retention_scorer.pt`
- Train candidates: 278,417, positive rate 21.78%.
- Tank-val best pseudo-label F1: 0.891 at threshold 0.78.

### Tank Stress Val ORB Pareto

Result: `tmp/aqua_retention_scorer/tank_val_windows_v2_pareto/aggregate_metrics.json`

| Variant | Feature Contam. | Match Contam. | E Success | Feat/Frame |
| --- | ---: | ---: | ---: | ---: |
| Raw | 16.62% | 12.41% | 100.00% | 1052.2 |
| Aqua clean 0.55 | 4.12% | 4.11% | 100.00% | 897.8 |
| Rule SLAM retain | 7.65% | 9.65% | 100.00% | 935.7 |
| v2 learned t=0.78 | 6.49% | 8.59% | 100.00% | 930.6 |
| v2 learned t=0.90 | 5.95% | 7.72% | 100.00% | 927.8 |
| Temporal RGB | 16.52% | 11.47% | 100.00% | 1022.4 |

Takeaway: v2 is a clear Pareto improvement on Tank validation windows. It keeps
essential-matrix success at 100% while reducing the dynamic re-admission cost of
rule retention. At t=0.90, feature contamination drops from 7.65% to 5.95% and
match contamination from 9.65% to 7.72% relative to rule retention.

### Tank Focused pyCOLMAP Checks

Focused checks on `fish-high` and `snow-high` are in:

- `tmp/aqua_tank_pose_stress_128_eval_v2_t078_fishhigh_snow/`
- `tmp/aqua_tank_pose_stress_128_eval_v2_t090_fishhigh_snow/`

The t=0.90 hard-retention check is the most encouraging point:

| Stress | Variant | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. |
| --- | --- | ---: | ---: | ---: | ---: |
| fish-high | Raw | 100.00% | 0.0817 | 0.0097 | 15.15% |
| fish-high | v2 learned hard t=0.90 | 100.00% | 0.0343 | 0.0146 | 8.39% |
| snow-high | Raw | 100.00% | 0.0241 | 0.0259 | 19.30% |
| snow-high | v2 learned hard t=0.90 | 6.25% | 0.0056 | 0.0063 | 13.52% |

This suggests v2 can improve pose quality under fish stress while reducing
contamination. However, the snow-high t=0.90 result registers only 4/64 input
frames, so it should be described as a small local reconstruction, not a full
sequence win. The t=0.78 check also shows pyCOLMAP sensitivity to initial
reconstruction and time limits. This motivated the multi-seed/fixed-pairing
validation summarized below before making a strong downstream pose claim.

## Follow-Up: Multi-Seed / Fixed-Pair Validation

The pyCOLMAP evaluator was extended with:

```text
--pycolmap-random-seeds 42,43,44 --fixed-initial-pair auto
```

Result: `tmp/aqua_tank_pose_stress_multiseed_v2_t090_fishhigh_snow/summary_seed_stability.csv`

Aggregated over `fish-high` and `snow-high`:

| System | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. |
| --- | ---: | ---: | ---: | ---: |
| Raw | 84.11% | 0.0556 | 0.0234 | 17.22% |
| Aqua hard | 22.40% | 0.0249 | 0.0266 | 9.05% |
| v2 learned hard t=0.90 | 22.40% | 0.0243 | 0.0219 | 10.95% |
| v2 learned soft t=0.90 | 67.19% | 0.0815 | 0.0351 | 10.95% |

This stabilizes the interpretation. The hard Aqua/v2 path can produce cleaner
and lower-ATE reconstructions on the high-stress pair, but with much lower
registration completeness. The soft path recovers more input frames but is not
yet pose-accurate enough. The current paper claim should remain Pareto-based.

## Follow-Up: Pose-Aware Soft Retention

The previous soft path used the learned retained mask as a binary copy-back
region. It recovered more frames than hard filtering, but the stabilized run
still had worse pose accuracy than raw. The new pose-aware soft path turns
retention into a continuous original-image weight:

```text
retention_weight = f(learned score, inlier support, inlier ratio,
                     patch NCC, match distance, flow)
```

This weight is used by `soft_temporal_fill_video` to partially preserve
geometrically stable rejected pixels while temporally filling weaker rejected
regions.

Implementation:

- `scripts/aqua_retention_utils.py`
  - `pose_aware_retention_weight_from_candidates`
- `scripts/aqua_prefilter_utils.py`
  - `soft_temporal_fill_video(..., retain_weight=...)`
- `scripts/eval_aqua_pose_gt_validation.py`
  - `--enable-pose-aware-soft-retention`
- `scripts/eval_aqua_pycolmap_validation.py`
  - `--enable-pose-aware-soft-retention`

Result: `tmp/aqua_pose_aware_soft_retention_poseonly_multiseed_t090_fishhigh_snow/summary_seed_stability.csv`

Aggregated over Tank `fish-high` and `snow-high`, three seeds:

| System | Success | Input Reg. | ATE RMSE | RPE Trans. | Points3D | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 100.00% | 84.11% | 0.0556 | 0.0234 | 2997.5 | 17.22% | 14.43% |
| v2 learned hard t=0.90 | 100.00% | 22.40% | 0.0243 | 0.0219 | 763.8 | 10.95% | 12.94% |
| v2 learned soft t=0.90 | 83.33% | 67.19% | 0.0815 | 0.0351 | 2422.0 | 10.95% | 12.94% |
| v2 pose-aware soft t=0.90 | 100.00% | 83.85% | 0.0860 | 0.0377 | 2953.3 | 10.95% | 12.94% |

Takeaway: pose-aware soft retention solves the soft path's registration
completeness problem and keeps contamination far below raw, but it does not
solve pose accuracy. It should be presented as the high-completeness endpoint
of the retention Pareto, while hard v2 remains the low-ATE endpoint.

## Current Verdict

The learned retention scorer is a good next method step:

1. It turns retention from a fixed rule into an optimized score.
2. It produces a controllable contamination vs. E-success Pareto on WebUOT.
3. It reduces the dynamic re-admission cost of rule-based retention.
4. Tank-aware v2 gives the cleanest learned-retention ORB Pareto so far.

It is not yet enough to claim robust full downstream SLAM/SfM improvement:

1. pyCOLMAP registration is now competitive with aggressive rule retention, but
   raw input still has the highest registration success.
2. WebUOT labels are incomplete tracked-target bbox masks.
3. Cross-domain synthetic performance is mixed.

The pyCOLMAP stabilization target with multi-seed or fixed initial pairs is now
done. The next target is to make the retained image/SfM representation softer
and pose-aware. The scorer should optimize track survival or downstream pose
quality rather than only keypoint pseudo-label F1.

## Follow-Up: Pose-Label v3 Retention

R096 trained a v3 retention scorer with GT-pose geometric labels on Tank stress
v2 train starts. GT pose is used for training labels only; the default scorer
input remains deployable Aqua/ORB/context features.

- Scorer:
  `tmp/aqua_retention_scorer/tank_pose_v3_train_start0000_0064_val_start0128_stress4_mlp32/retention_scorer.pt`
- Full merged result:
  `tmp/aqua_tank_pose_stress_v2_stress4_multiseed_v3_t073_merged/summary_seed_stability.csv`

Aggregated over 24 clips x 3 pyCOLMAP seeds:

| System | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 75.00% | 33.79% | 0.0469 | 0.0288 | 22.86% | 17.63% |
| v3 learned hard t=0.73 | 80.56% | 31.47% | 0.0497 | 0.0348 | 13.83% | 15.39% |
| v3 pose-aware soft t=0.73 | 77.78% | 28.39% | 0.0398 | 0.0281 | 13.83% | 15.40% |

Takeaway: v3 pose-soft is a stronger pose-trained Pareto endpoint than the
previous soft path in some stress settings, especially `mixed-fish-snow`, but
it is still not a final downstream win because overall input registration is
lower than raw. The next method step should be adaptive hard/soft selection,
not another fixed threshold.

## Follow-Up: Adaptive v3 Hard/Soft Selector

After cleanup removed old `tmp` scorer artifacts, the v3 pose-label scorer was
reproduced from the retained Tank stress v2 data:

- Train manifests:
  `data/real_underwater/tank_pose_stress_v2/splits/stress4_train_start0000_0064_manifests.txt`
- Val manifests:
  `data/real_underwater/tank_pose_stress_v2/splits/stress4_val_start0128_manifests.txt`
- Reproduced scorer:
  `tmp/aqua_retention_scorer/tank_pose_v3_train_start0000_0064_val_start0128_stress4_mlp32_repro/retention_scorer.pt`

Candidate/training summary:

| Split | Candidates | Positive Rate |
| --- | ---: | ---: |
| train start0000/0064 stress4 | 1,092,178 | 24.70% |
| val start0128 stress4 | 546,702 | 25.53% |

The reproduced MLP32 scorer reaches validation pseudo-label F1 0.9600 at
threshold 0.73, matching the intended R096 operating point.

An initial adaptive variant was implemented inside
`scripts/eval_aqua_pose_gt_validation.py`, but the firstclip sanity check showed
that copying either the hard or pose-soft rendered video into a new adaptive
variant and rerunning pyCOLMAP can introduce extra pyCOLMAP nondeterminism.
The cleaner evaluation is therefore post-hoc deployable selection over already
run hard/pose-soft variants:

- Script: `scripts/analyze_aqua_adaptive_pose_selector_v2.py`
- Firstclip result, now cleaned after full-run superseding:
  `tmp/aqua_adaptive_v3_t073_firstclip_seed42_selector_v2/summary.csv`

Firstclip seed42 summary over fish-high, fish-extreme, snow-high, and
mixed-fish-snow:

| Policy | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 hard always | 100.00% | 10.16% | 0.0617 | 0.0547 | 13.47% |
| v3 pose-soft always | 75.00% | 75.78% | 0.0318 | 0.0123 | 13.46% |
| Aqua dynamic high guard | 100.00% | 78.52% | 0.0430 | 0.0360 | 13.46% |

Interpretation: the deployable Aqua-score guard avoids the fish-extreme
pose-soft failure while preserving the high-completeness behavior of pose-soft
on the other firstclip stress cases. GT-derived `dynamic_guard` and
`conservative_soft` policies were also checked as diagnostics, but should not be
reported as deployable systems.

The full stress4 run is now complete. The shard and selector-v2 directories
below were cleaned after preserving merged summaries and after R099 superseded
selector-v2 as the main downstream result:

```text
tmp/aqua_adaptive_v3_t073_full_stress4_merged/
tmp/aqua_adaptive_v3_t073_full_stress4_selector_v3_rawfallback/
tmp/aqua_adaptive_v3_t073_full_stress4_selector_v3_sensitivity/
```

Full stress4 result over 24 clips x 3 pyCOLMAP seeds:

| Policy / System | Success | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 98.61% | 81.94% | 38.00% | 0.0435 | 0.0283 | 22.86% | 17.63% |
| v3 hard always | 98.61% | 84.72% | 32.12% | 0.0529 | 0.0362 | 13.83% | 15.39% |
| v3 pose-soft always | 95.83% | 83.33% | 23.05% | 0.0306 | 0.0253 | 13.83% | 15.40% |
| Aqua dynamic high guard | 97.22% | 83.33% | 24.11% | 0.0330 | 0.0268 | 13.83% | 15.40% |

The deployable `aqua_dynamic_high_guard` policy uses only Aqua dynamic-score
statistics for selection. It chooses v3 hard on 21 records and v3 pose-soft on
51 records. In the full run it preserves the main benefit of pose-soft
(substantially lower ATE/RPE and lower contamination than raw), but it does not
recover raw-level registration completeness.

Result-to-claim gate for R098:

- Supported: v3/adaptive selection gives a deployable low-contamination,
  low-pose-error Pareto endpoint. Against same-run raw, feature contamination
  drops 22.86% -> 13.83%, ATE drops 0.0435 -> 0.0330, and RPE drops
  0.0283 -> 0.0268.
- Not supported: the selector is not a complete downstream SfM/SLAM win,
  because input registration drops 38.00% -> 24.11% and stress-specific
  behavior remains uneven.
- Paper wording should stay in the bounded form: clean static geometry plus
  controllable downstream retention Pareto. The adaptive v3 result is useful
  evidence, but not enough to claim broad underwater SLAM superiority.

## Follow-Up: v3 Pose-Soft + Raw Fallback Selector

R099 adds a self-diagnostic raw-registration fallback on top of v3 pose-soft:
use v3 pose-soft by default, but select raw when raw input registration is at
least 0.60 and exceeds pose-soft registration by more than 0.10. This does not
use GT pose or GT masks, but it is a multi-pass selector because both raw and
pose-soft candidate reconstructions must be computed.

- Script: `scripts/analyze_aqua_adaptive_pose_selector_v3.py`
- Result:
  `tmp/aqua_adaptive_v3_t073_full_stress4_selector_v3_rawfallback/summary.csv`

Full stress4 result over 24 clips x 3 pyCOLMAP seeds:

| Policy / System | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 81.94% | 38.00% | 0.0435 | 0.0283 | 22.86% | 17.63% |
| v3 pose-soft always | 83.33% | 23.05% | 0.0306 | 0.0253 | 13.83% | 15.40% |
| v3 pose-soft + raw fallback | 90.28% | 40.13% | 0.0389 | 0.0281 | 15.17% | 15.73% |

Selection counts: v3 pose-soft on 57 records and raw on 15 records.

Takeaway: this is the strongest downstream result so far on the expanded Tank
stress4 benchmark. It beats same-run raw in aggregate pose-eval success, input
registration, ATE/RPE, and contamination. The correct caveat is that this is a
self-diagnostic multi-pass selector, not a cheap one-pass online policy, and
stress-specific wins are still uneven.

## Follow-Up: R099 Raw-Fallback Sensitivity

R100 re-analyzed the same full stress4 per-clip outputs with a small threshold
grid around the R099 selector:

- Result:
  `tmp/aqua_adaptive_v3_t073_full_stress4_selector_v3_sensitivity/summary.csv`
- Grid:
  `raw_min_registration in {0.50, 0.60, 0.70}`,
  `raw_margin in {0.05, 0.10, 0.15, 0.20}`.

Summary over all 12 fallback settings:

| System / Range | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw baseline | 81.94% | 38.00% | 0.0435 | 0.0283 | 22.86% |
| R099 selected point | 90.28% | 40.13% | 0.0389 | 0.0281 | 15.17% |
| R100 fallback grid | 90.28% | 37.76-40.97% | 0.0363-0.0437 | 0.0266-0.0282 | 14.98-15.70% |

Interpretation: the raw-fallback selector is not a brittle single-threshold
win. Lower raw-min settings recover slightly more registration; higher raw-min
settings preserve more of the pose-soft low-error/low-contamination behavior.
The paper should report one operating point, plus this sensitivity as evidence
that the adaptive-retention conclusion is stable.
