# Aqua-D4RT Head Ablation Results

Date: 2026-06-18

## Goal

This report consolidates the required no-dynamic-head and no-particle-head
ablations for the ICRA claim package. It separates three questions:

1. Do the dynamic and particle heads matter for their own labels?
2. Do they matter for static query-map cleanliness?
3. Do their effects survive downstream pyCOLMAP-style pose validation?

The answer is yes for the first two questions. For the third question, the
result is more nuanced: removing the particle loss can preserve more texture and
therefore improve some Tank pyCOLMAP registration numbers, but it fails the
particle and static-map cleanliness tests. This should be framed as a trade-off,
not as evidence that the particle head is unnecessary.

## Implemented

New configs:

- `configs/train_aqua_ablation_no_dynamic_headonly.yaml`
- `configs/train_aqua_ablation_no_particle_headonly.yaml`

New/extended evaluator features:

- `scripts/aqua_retention_utils.py`
  - Added static score modes: `full`, `no_dynamic`, `no_particle`,
    `confidence_only`.
- `scripts/eval_aqua_static_map_contamination.py`
  - Added `--static-score-modes`.
- `scripts/eval_aqua_downstream_slam_proxy.py`
  - Added `--static-score-modes`.
- `scripts/eval_aqua_pose_gt_validation.py`
  - Added `--static-score-mode`.
- `scripts/eval_aqua_pycolmap_validation.py` and
  `scripts/eval_aqua_pose_gt_validation.py`
  - Added `--pycolmap-random-seeds`, `--fixed-initial-pair`, and
    fixed-pair fallback controls.
- `src/engine/trainer.py`
  - Added `checkpoint.save_last: false` support to avoid redundant 4.4 GB
    `last.ckpt` writes during ablation training.

Checkpoints:

- no-dynamic: `output/exp_aqua_d4rt/aqua_ablation_no_dynamic_headonly/checkpoints/best.ckpt`
- no-particle: `output/exp_aqua_d4rt/aqua_ablation_no_particle_headonly/checkpoints/best.ckpt`

Both ablation configs train only the transient heads for 1000 steps from the
same base initialization and use the same synthetic split as the main head-only
training.

## Full Checkpoint: Inference Score Ablation

These ablations use the same trained full Aqua checkpoint, but disable terms in
the static score at inference time:

```text
full static_conf = sigmoid(confidence) * (1 - dynamic_prob) * (1 - particle_prob)
no_dynamic       = sigmoid(confidence) * (1 - particle_prob)
no_particle      = sigmoid(confidence) * (1 - dynamic_prob)
confidence_only  = sigmoid(confidence)
```

### Synthetic Static Query-Map

Result: `tmp/aqua_ablation_static_map_synth_test/summary_table.csv`

| Variant | Point Contam. | Static Ret. |
| --- | ---: | ---: |
| Raw D4RT query points | 10.82% | 100.00% |
| Full Aqua static_conf >= 0.55 | 0.39% | 93.00% |
| No dynamic term | 8.00% | 94.58% |
| No particle term | 4.03% | 98.72% |
| Confidence only | 10.82% | 100.00% |

Interpretation: both transient terms are needed for the synthetic clean-map
claim. The dynamic term removes most fish contamination, while the particle term
is responsible for a large additional reduction from 4.03% to 0.39%.

### WebUOT Fish30 Static Query-Map

Result: `tmp/aqua_ablation_static_map_webuot_all30/summary_table.csv`

Important caveat: WebUOT labels are tracked-target bbox masks, not complete fish
instance masks.

| Variant | Point Contam. | Static Ret. |
| --- | ---: | ---: |
| Raw D4RT query points | 10.86% | 100.00% |
| Full Aqua static_conf >= 0.55 | 4.47% | 75.53% |
| No dynamic term | 10.84% | 98.92% |
| No particle term | 5.08% | 79.43% |
| Confidence only | 10.86% | 100.00% |

Interpretation: the dynamic head is essential for real tracked-fish removal.
The particle term has a smaller measured effect here because WebUOT provides
fish bbox labels rather than particle/marine-snow labels.

## Training Ablations on Synthetic Test

### Transient Head F1

Results:

- no-dynamic: `tmp/aqua_train_ablation_no_dynamic_synth_test/summary_brief.json`
- no-particle: `tmp/aqua_train_ablation_no_particle_synth_test/summary_brief.json`
- full reference: `tmp/aqua_benchmark_eval/aqua_synth_test_real_synth_mix_headonly/summary_brief.json`

| Model | Dynamic Best F1 | Particle Best F1 | Static Best F1 |
| --- | ---: | ---: | ---: |
| Full real+synth Aqua | 0.924 | 0.698 | 0.984 |
| No dynamic loss/head target | 0.132 | 0.698 | 0.952 |
| No particle loss/head target | 0.944 | 0.078 | 0.975 |

Interpretation: the loss ablations are clean. Removing the dynamic target
collapses dynamic detection, and removing the particle target collapses particle
detection, while the non-ablated target remains largely intact.

### Static Map Contamination

Results:

- no-dynamic: `tmp/aqua_train_ablation_static_no_dynamic_synth_test/summary_table.csv`
- no-particle: `tmp/aqua_train_ablation_static_no_particle_synth_test/summary_table.csv`
- full reference: `tmp/aqua_ablation_static_map_synth_test/summary_table.csv`

| Model | Static_conf >= 0.55 Point Contam. | Static Ret. |
| --- | ---: | ---: |
| Raw D4RT query points | 10.82% | 100.00% |
| Full real+synth Aqua | 0.39% | 93.00% |
| No dynamic training | 21.04% | 10.62% |
| No particle training | 4.27% | 80.34% |

Interpretation: no-dynamic is catastrophic for the static-map path: it retains
very few static points while still having worse contamination than raw. The
no-particle model still detects fish but cannot match the full clean-map result,
especially on particle-heavy synthetic scenes.

## Tank Pose Stress Multi-Seed / Fixed-Pair Ablation

The Tank pose evaluator now supports stabilized pyCOLMAP runs with:

```text
--pycolmap-random-seeds 42,43,44 --fixed-initial-pair auto
```

Fallback from an unsuitable fixed pair is enabled by default. The summary below
aggregates 4 Tank stress variants x 3 seeds.

### No-Dynamic Checkpoint

Result: `tmp/aqua_tank_pose_ablation_no_dynamic_pose_gt/summary_seed_stability.csv`

| Variant | Success | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 91.67% | 83.33% | 50.65% | 0.0233 | 0.0143 | 11.04% | 8.94% |
| No-dynamic Aqua hard | 100.00% | 66.67% | 5.73% | 0.0233 | 0.0252 | 4.78% | 6.70% |
| No-dynamic Aqua retain | 100.00% | 83.33% | 15.10% | 0.0374 | 0.0306 | 6.17% | 7.44% |

Interpretation: without the dynamic branch, Aqua can still reject some features,
but the reconstruction becomes too sparse and RPE worsens. This supports the
dynamic-head necessity claim for fish-like transient interference.

### No-Particle Checkpoint

Result: `tmp/aqua_tank_pose_ablation_no_particle_pose_gt/summary_seed_stability.csv`

| Variant | Success | Pose-Eval Success | Input Reg. | ATE RMSE | RPE Trans. | Feature Contam. | Match Contam. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 83.33% | 75.00% | 32.55% | 0.0358 | 0.0385 | 11.04% | 8.94% |
| No-particle Aqua hard | 100.00% | 91.67% | 58.07% | 0.0211 | 0.0166 | 8.34% | 7.35% |
| No-particle Aqua retain | 100.00% | 91.67% | 62.24% | 0.0223 | 0.0157 | 9.03% | 8.32% |

Interpretation: the no-particle model retains more texture and is therefore
friendlier to pyCOLMAP on this Tank stress set. This is not a replacement for
the particle head: the same checkpoint has particle best F1 0.078 and a weaker
synthetic static-map result. The downstream lesson is that particle suppression
needs a softer pose-aware retention path, not that the particle branch should be
removed.

## Claim Boundary

Supported:

1. The dynamic and particle heads are independently necessary for their target
   transient classes.
2. The full static-confidence product is necessary for the strongest synthetic
   query-map cleanliness result.
3. On WebUOT tracked-target bbox labels, the dynamic term is the decisive real
   fish-removal component.
4. pyCOLMAP results are sensitive to texture retention and initialization, so
   multi-seed/fixed-pair reporting is required for any downstream pose claim.

Not supported:

1. A claim that removing the particle head improves the method. It improves some
   Tank registration numbers by keeping more texture, but fails particle
   detection and degrades clean-map performance.
2. A blanket ATE/RPE SOTA claim. The current evidence supports a clean-map plus
   retention Pareto claim, with encouraging pose improvements on selected
   fish-stress runs.
