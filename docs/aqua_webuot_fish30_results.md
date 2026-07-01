# Aqua-D4RT WebUOT Fish30 Adaptation Results

## Date

2026-06-16

## Purpose

This experiment expands the real-fish validation from 5 WebUOT clips to 30
WebUOT-238-Test clips and tests whether lightweight query-level transient heads
can adapt Aqua-D4RT to real underwater fish videos.

Labels are WebUOT tracked-target bounding boxes rasterized into masks. They are
useful real-video supervision, but they are not full fish instance masks and do
not label every dynamic object in the frame.

## Data

Prepared dataset:

```text
data/real_underwater/webuot238_fish30/
  manifests.txt
  splits/train_24_manifests.txt
  splits/val_6_manifests.txt
  splits/train_24_x3_manifests.txt
```

Evaluation uses 32 frames per clip, 256x256 frames, stride-8 query grid.

## Systems

| System | Meaning |
| --- | --- |
| Temporal RGB prefilter | Label-free median-residual motion mask; no D4RT. |
| Phase-A synthetic only | Aqua heads trained on synthetic fish/particle corruption only. |
| Phase-C 5-clip head-only | Real adaptation on the earlier 4 train / 1 val WebUOT pilot. |
| Fish30 head-only | Heads trained on 24 WebUOT fish30 train clips, val on 6 clips. |
| Real+Synth mix head-only | Heads trained on repeated WebUOT fish30 train clips plus synthetic replay. |

Both Fish30 and Real+Synth freeze encoder, memory projection, query embedder,
and decoder. Only the transient heads are trainable.

## Commands

Fish30 head-only:

```bash
CUDA_VISIBLE_DEVICES=0 /media/data/u24conda/envs/longlive/bin/python train.py \
  --model-config checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --train-config configs/train_aqua_webuot_fish30_headonly.yaml \
  --init-model output/exp_aqua_d4rt/aqua_synth_phase_a_multiclip_100_1k/checkpoints/best.ckpt
```

Real+Synth mixed replay:

```bash
CUDA_VISIBLE_DEVICES=0 /media/data/u24conda/envs/longlive/bin/python train.py \
  --model-config checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --train-config configs/train_aqua_real_synth_mix_headonly.yaml \
  --init-model output/exp_aqua_d4rt/aqua_synth_phase_a_multiclip_100_1k/checkpoints/best.ckpt
```

## Training Summary

| Run | Steps | Best Val Step | Best Val Loss | Notes |
| --- | ---: | ---: | ---: | --- |
| Fish30 head-only | 800 | 700 | 0.659692 | Real-only dynamic BCE. |
| Real+Synth mix head-only | 1000 | 750 | 0.642727 | Real fish train list repeated 3x plus 70 synthetic train clips. |

## WebUOT Fish30 All-Clip Results

Split: all 30 prepared clips, 983,040 query-grid points, dynamic bbox coverage
10.86%.

| Method | Dynamic F1 @0.5 | Dynamic Best F1 | Best Thr | Static F1 @0.5 | Static Best F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Temporal RGB prefilter | 0.2310 | - | - | - | - |
| GrabCut-box prefilter | 0.6279 | - | WebUOT box | - | - |
| SAM-base-box prefilter | 0.6471 | - | WebUOT box | - | - |
| Phase-A synthetic only | 0.3311 | 0.3480 | 0.24 | 0.8954 | 0.9425 |
| Phase-C 5-clip head-only | 0.3131 | 0.3502 | 0.85 | 0.8574 | 0.9455 |
| Fish30 head-only | 0.4061 | 0.4163 | 0.78 | 0.8511 | 0.9447 |
| Real+Synth mix head-only | 0.3942 | 0.4349 | 0.88 | 0.8596 | 0.9440 |

Key deltas on all30:

- Fish30 head-only improves dynamic best F1 over Phase-A by +0.0684.
- Real+Synth mix improves dynamic best F1 over Phase-A by +0.0869.
- Real+Synth mix improves dynamic best F1 over Fish30 head-only by +0.0185.
- Real+Synth mix has slightly lower F1 @0.5 than Fish30 head-only, but a better
  threshold-ranked best F1 and better static F1 @0.5.
- Box-prompted SAM/GrabCut have higher 2D mask F1 because they use WebUOT GT
  boxes as prompts. They should be treated as segmentation upper/medium
  baselines, not as a fair RGB-only deployment baseline.

## Held-Out Val6 Results

| Method | Dynamic F1 @0.5 | Dynamic Best F1 | Static F1 @0.5 | Static Best F1 |
| --- | ---: | ---: | ---: | ---: |
| Fish30 head-only | 0.4516 | 0.5088 | 0.7961 | 0.8929 |
| Real+Synth mix head-only | 0.4450 | 0.4812 | 0.8130 | 0.8967 |

The mixed model is better on all30 but weaker on the small val6 best-F1 metric.
This suggests mixed replay gives a more conservative/global decision boundary
rather than over-specializing to the val split.

## Synthetic Back-Check

Split: `data/aqua_synth_benchmark/watermask_caves_100/splits/test_manifests.txt`.

| Method | Dynamic Best F1 | Particle Best F1 | Static Best F1 | Contamination @static 0.55 | Static Retention @0.55 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fish30 head-only | 0.9372 | 0.6992 | 0.9854 | 0.62% | 94.63% |
| Real+Synth mix head-only | 0.9241 | 0.6979 | 0.9844 | 0.39% | 93.00% |

Particle performance is essentially preserved despite real-fish adaptation. This
supports using synthetic replay as a forgetting-control mechanism for marine
snow / particle behavior.

## Interpretation

1. Real WebUOT supervision helps: dynamic best F1 improves from 0.3480
   (synthetic-only Phase-A) to 0.4163 (Fish30 head-only) and 0.4349
   (Real+Synth mix).
2. The current gain is meaningful but still not ICRA-final. WebUOT labels are
   coarse tracking boxes, and best F1 around 0.43 is not yet strong enough to
   claim reliable removal of all fish.
3. Mixed real+synth replay is promising: it improves all30 best F1 while
   preserving synthetic particle performance around 0.70 best F1.
4. The strongest supported claim right now is query-level real-domain adaptation
   for transient fish localization, not full underwater static reconstruction.

## Result-to-Claim Gate

Local verdict: **partial**.

Supported:

- Query-level transient heads can adapt from synthetic underwater corruption to
  real WebUOT fish clips.
- Real+synthetic replay improves all30 dynamic best F1 and avoids particle-head
  forgetting on the synthetic benchmark.
- The temporal RGB prefilter is a weak baseline on WebUOT fish30.
- Box-prompted segmentation baselines are now implemented and quantify what a
  strong prefilter can do if object boxes are available.
- A fully non-oracle GroundingDINO prefilter baseline is now implemented:
  GroundingDINO-box reaches F1 0.5337 on WebUOT fish30 all30, while
  GroundingDINO+SAM-base reaches F1 0.4155 under the tracked-target bbox-mask
  metric.

Not yet supported:

- End-to-end static 3D reconstruction quality improvement on real underwater
  scenes.
- Downstream SLAM / mapping gains.
- Robust removal of all fish and suspended particles in unconstrained videos.
- Superiority over strong box-prompted segmentation prefilters on 2D mask F1.
  Current SAM-base-box F1 is 0.6471, higher than Aqua query-head best F1 0.4349.

Next evidence needed:

1. Push the detector-generated baseline into static-map contamination and ORB
   proxy evaluation if we want a direct 3D comparison against Aqua filtering.
2. Point-cloud/static-map contamination evaluation on WebUOT and synthetic clips.
3. COLMAP or ORB-SLAM3 downstream validation on raw, RGB-prefiltered, and
   Aqua-filtered outputs.
4. More real clips or better masks if available.

## Artifacts

```text
configs/train_aqua_webuot_fish30_headonly.yaml
configs/train_aqua_real_synth_mix_headonly.yaml
data/real_underwater/webuot238_fish30/
output/exp_aqua_d4rt/aqua_webuot_fish30_headonly/
output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/
tmp/aqua_benchmark_eval/webuot238_fish30/fish30_headonly_grid8/
tmp/aqua_benchmark_eval/webuot238_fish30/fish30_headonly_val6_grid8/
tmp/aqua_benchmark_eval/webuot238_fish30/real_synth_mix_headonly_grid8/
tmp/aqua_benchmark_eval/webuot238_fish30/real_synth_mix_headonly_val6_grid8/
tmp/aqua_benchmark_eval/aqua_synth_test_fish30_headonly/
tmp/aqua_benchmark_eval/aqua_synth_test_real_synth_mix_headonly/
tmp/aqua_prefilter_masks/webuot238_fish30_temporal_rgb/
tmp/aqua_prefilter_masks/webuot238_fish30_grabcut_box/
tmp/aqua_prefilter_masks/webuot238_fish30_sam_base_box/
tmp/aqua_prefilter_masks/webuot238_fish30_groundingdino_box_all30/
tmp/aqua_prefilter_masks/webuot238_fish30_groundingdino_sam_all30/
```
