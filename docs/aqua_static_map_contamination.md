# Aqua-D4RT Static Point/Map Contamination

Date: 2026-06-16

## Goal

This evaluation measures whether transient fish / particle regions are reconstructed into the static D4RT query-map. It is not a full SLAM-fused global map yet. The map here is the set of D4RT query-level `xyz_3d` predictions, optionally voxelized for map-level contamination.

## Scripts

- `scripts/eval_aqua_static_map_contamination.py`
- Checkpoint: `output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt`
- Config: `checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml`

## Metrics

- `point_contamination`: fraction of kept query points whose GT label is fish/particle transient.
- `point_static_retention`: fraction of GT-static query points retained.
- `voxel_contamination_any`: fraction of kept voxels containing at least one kept transient point.
- `voxel_static_support_retention`: fraction of GT-static-support voxels retained.

## Synthetic Test: `watermask_caves_100` 15 clips

Result: `tmp/aqua_static_map_contamination/watermask_caves_100_test_real_synth/aggregate_metrics.json`

| Variant | Point Contam. | Static Ret. | Voxel Contam. | Voxel Ret. |
| --- | ---: | ---: | ---: | ---: |
| Raw D4RT query points | 10.82% | 100.00% | 9.55% | 100.00% |
| Aqua transient filter | 1.35% | 98.08% | 1.65% | 97.93% |
| Aqua static conf >= 0.11 | 1.63% | 98.49% | 1.99% | 98.33% |
| Aqua static conf >= 0.55 | 0.39% | 93.00% | 0.48% | 92.96% |
| Temporal RGB prefilter | 8.08% | 96.90% | 7.81% | 97.09% |
| Oracle GT static | 0.00% | 100.00% | 0.00% | 100.00% |

## WebUOT Fish30: 30 real clips

Result: `tmp/aqua_static_map_contamination/webuot238_fish30_real_synth/aggregate_metrics.json`

Important caveat: WebUOT labels are tracked-target bbox masks, not full fish-instance masks. Other visible fish may be unlabeled.

| Variant | Point Contam. | Static Ret. | Voxel Contam. | Voxel Ret. |
| --- | ---: | ---: | ---: | ---: |
| Raw D4RT query points | 10.86% | 100.00% | 18.13% | 100.00% |
| Aqua transient filter | 6.47% | 89.82% | 12.25% | 87.33% |
| Aqua static conf >= 0.11 | 6.81% | 92.61% | 12.97% | 90.14% |
| Aqua static conf >= 0.55 | 4.47% | 75.53% | 8.66% | 71.92% |
| Temporal RGB prefilter | 9.49% | 95.56% | 16.69% | 94.12% |
| Oracle GT static | 0.00% | 100.00% | 0.00% | 100.00% |

## Static-Score Ablation

Result summary: `docs/aqua_head_ablation_results.md`

The full Aqua static score uses both transient heads:

```text
static_confidence = sigmoid(confidence) * (1 - dynamic_prob) * (1 - particle_prob)
```

Using the full trained checkpoint but disabling score terms at inference time:

| Dataset | Variant | Point Contam. | Static Ret. |
| --- | --- | ---: | ---: |
| Synthetic test | Full static_conf >= 0.55 | 0.39% | 93.00% |
| Synthetic test | No dynamic term | 8.00% | 94.58% |
| Synthetic test | No particle term | 4.03% | 98.72% |
| Synthetic test | Confidence only | 10.82% | 100.00% |
| WebUOT fish30 | Full static_conf >= 0.55 | 4.47% | 75.53% |
| WebUOT fish30 | No dynamic term | 10.84% | 98.92% |
| WebUOT fish30 | No particle term | 5.08% | 79.43% |
| WebUOT fish30 | Confidence only | 10.86% | 100.00% |

Training ablations on synthetic test further confirm the head roles:

| Model | Dynamic Best F1 | Particle Best F1 | Static-Map Contam. @0.55 | Static Ret. @0.55 |
| --- | ---: | ---: | ---: | ---: |
| Full real+synth Aqua | 0.924 | 0.698 | 0.39% | 93.00% |
| No dynamic training | 0.132 | 0.698 | 21.04% | 10.62% |
| No particle training | 0.944 | 0.078 | 4.27% | 80.34% |

Interpretation: the dynamic branch is the decisive component for fish-like
transients, especially on WebUOT. The particle branch is required for the
strongest synthetic fish+particle clean-map result, reducing contamination from
4.03% without the particle term to 0.39% with the full score.

## Takeaways

1. On synthetic data, Aqua-D4RT strongly supports the static-map-cleanliness claim: point contamination drops from 10.82% to 0.39% at 93.00% static retention.
2. On real WebUOT bbox labels, Aqua still reduces tracked-fish contamination, but the trade-off is harsher: 10.86% to 4.47% with 75.53% retention.
3. Temporal RGB prefilter is weaker than Aqua on both synthetic and WebUOT for map contamination.
4. The no-dynamic/no-particle ablations support the architecture claim: the
   clean-map result depends on query-level transient heads, not just OpenD4RT
   confidence.
5. For the ICRA story, this supports a static map cleanliness claim, but not
   yet a full SLAM trajectory claim.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=0 /media/data/u24conda/envs/longlive/bin/python scripts/eval_aqua_static_map_contamination.py \
  --manifest-list data/aqua_synth_benchmark/watermask_caves_100/splits/test_manifests.txt \
  --ckpt-path output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt \
  --output-dir tmp/aqua_static_map_contamination/watermask_caves_100_test_real_synth \
  --device cuda --grid-stride 8 --query-chunk-size 4096 \
  --static-thresholds 0.11,0.55 --include-rgb-prefilter

CUDA_VISIBLE_DEVICES=0 /media/data/u24conda/envs/longlive/bin/python scripts/eval_aqua_static_map_contamination.py \
  --manifest-list data/real_underwater/webuot238_fish30/manifests.txt \
  --ckpt-path output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt \
  --output-dir tmp/aqua_static_map_contamination/webuot238_fish30_real_synth \
  --device cuda --grid-stride 8 --query-chunk-size 4096 \
  --static-thresholds 0.11,0.55 --include-rgb-prefilter
```
