# Aqua-D4RT RGB Prefilter Baselines

## Date

2026-06-13

## Purpose

This experiment checks whether Aqua-D4RT's query-level static filtering provides
value beyond an RGB-space transient prefilter. It addresses the reviewer question:

> Is this just external fish/particle segmentation before D4RT?

## Script

```text
scripts/eval_aqua_prefilter_baselines.py
```

The script evaluates the held-out synthetic test split with the same query grid
and GT transient labels used by the Aqua-D4RT benchmark.

## Compared Systems

| System | Meaning |
| --- | --- |
| `opend4rt_raw_confidence` | Released OpenD4RT confidence score on corrupted input. |
| `opend4rt_oracle_mask_prefilter` | Oracle upper bound: GT transient query coordinates are removed before static aggregation. |
| `opend4rt_oracle_clean_confidence` | Diagnostic: transient pixels are replaced using clean frames, then OpenD4RT confidence is evaluated at original coordinates. |
| `aqua_raw_static_confidence` | Aqua-D4RT static confidence on corrupted input, no GT masks at inference. |

Important caveat:

- `opend4rt_oracle_mask_prefilter` is a cheating upper bound because it uses GT
  masks at evaluation/inference time.
- `opend4rt_oracle_clean_confidence` is diagnostic only; metrics still mark the
  original transient coordinates as transient even though the RGB pixels have
  been replaced by clean background.

## Command

```bash
/media/data/u24conda/envs/longlive/bin/python scripts/eval_aqua_prefilter_baselines.py \
  --manifest-list data/aqua_synth_benchmark/watermask_caves_100/splits/test_manifests.txt \
  --model-config checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --base-ckpt-path checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --aqua-ckpt-path output/exp_aqua_d4rt/aqua_synth_phase_a_multiclip_100_1k/checkpoints/best.ckpt \
  --output-dir tmp/aqua_prefilter_eval/watermask_caves_100_test \
  --device auto \
  --max-frames 32 \
  --grid-stride 8 \
  --query-chunk-size 4096 \
  --clean-map-max-contamination 0.005
```

Outputs:

```text
tmp/aqua_prefilter_eval/watermask_caves_100_test/
  aggregate_metrics.json
  per_clip_metrics.jsonl
  summary_table.json
```

## Held-Out Test Results

Split: 15 clips, 491,520 query-grid points, transient coverage 10.82%.

Clean-map target: choose the threshold with maximum static retention subject to
transient contamination <= 0.5%.

| System | Best Static F1 | Best Threshold | Clean-Map Found | Clean Threshold | Clean Contamination | Clean Static Retention | Kept Rate |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| OpenD4RT raw confidence | 0.9428 | 0.80 | no | - | - | - | - |
| OpenD4RT + oracle GT-mask prefilter | 1.0000 | 0.50 | yes | 0.81 | 0.00% | 100.00% | 89.18% |
| OpenD4RT + oracle clean-frame RGB | 0.9428 | 0.50 | no | - | - | - | - |
| Aqua-D4RT query-level filtering | 0.9866 | 0.09 | yes | 0.56 | 0.48% | 94.11% | 84.34% |

## Interpretation

- Raw OpenD4RT confidence is not a useful clean-map selector here: it keeps nearly
  all query points, including fish and particle regions, and cannot reach the
  <=0.5% contamination target through confidence thresholding.
- The oracle GT-mask prefilter is the expected upper bound: it removes all
  transient query coordinates by construction.
- Aqua-D4RT reaches the clean-map target without GT masks at inference:
  0.48% contamination at 94.11% static retention.
- This result supports the ICRA storyline that query-level transient prediction
  can act as a static mapping primitive, while still leaving room for a real
  non-oracle segmentation-prefilter baseline.

## Next

Remaining baseline work:

- Add a non-oracle RGB prefilter using a segmentation model or pseudo masks.
- Add clean-background geometric consistency metrics so the comparison is about
  3D/static map quality, not only query inclusion/exclusion.
- Run the same comparison on public underwater clips once real data is prepared.

## 2026-06-15 Update: Real WebUOT Non-Oracle Mask Pilot

Prepared 5 WebUOT-238-Test real fish clips:

```text
data/real_underwater/webuot238_sample/
  manifests.txt
  dataset_manifest.json
  WebUOT-1M_Test_000022/
  WebUOT-1M_Test_000025/
  WebUOT-1M_Test_000026/
  WebUOT-1M_Test_000043/
  WebUOT-1M_Test_000098/
```

Each clip has 32 resized frames, WebUOT tracking-bbox masks, cached temporal RGB
pseudo masks, preview videos, and contact sheets. WebUOT GT is only a tracked
target bounding box, not full fish instance segmentation.

Non-oracle temporal RGB mask pilot:

```bash
/media/data/u24conda/envs/longlive/bin/python scripts/eval_aqua_rgb_prefilter_masks.py \
  --manifest-list data/real_underwater/webuot238_sample/manifests.txt \
  --output-dir tmp/aqua_prefilter_masks/webuot238_sample_temporal_rgb \
  --max-frames 32 \
  --image-height 256 \
  --image-width 256 \
  --save-visuals
```

Aggregate result against WebUOT tracked-target bbox masks:

| Method | Precision | Recall | F1 | IoU | GT Coverage | Pred Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Temporal RGB pseudo-mask | 0.2187 | 0.1982 | 0.2079 | 0.1160 | 7.64% | 6.93% |

Interpretation: this is a weak but label-free non-oracle baseline. It often
captures moving regions and water-surface flicker, but misses or over-segments
fish depending on contrast and camera motion. The stronger paper baseline should
use SAM/SAM2 prompted by WebUOT boxes or a trained underwater segmentation model.

## 2026-06-16 Update: WebUOT Fish30 Box-Prompted Segmentation Baselines

After preparing the 30-clip WebUOT fish subset, we added:

```text
scripts/eval_aqua_box_prefilter_masks.py
```

This evaluates prefilter masks generated from WebUOT boxes. These are not fully
non-oracle deployment baselines because the boxes come from WebUOT GT labels.
They are useful as reviewer-facing upper/medium baselines: if a method receives
a fish box, how well can a segmentation prefilter convert it into a mask?

Commands:

```bash
/media/data/u24conda/envs/longlive/bin/python scripts/eval_aqua_box_prefilter_masks.py \
  --manifest-list data/real_underwater/webuot238_fish30/manifests.txt \
  --output-dir tmp/aqua_prefilter_masks/webuot238_fish30_grabcut_box \
  --method grabcut \
  --max-frames 32 \
  --image-height 256 \
  --image-width 256 \
  --grabcut-fallback-to-box \
  --save-visuals

CUDA_VISIBLE_DEVICES=0 /media/data/u24conda/envs/longlive/bin/python scripts/eval_aqua_box_prefilter_masks.py \
  --manifest-list data/real_underwater/webuot238_fish30/manifests.txt \
  --output-dir tmp/aqua_prefilter_masks/webuot238_fish30_sam_base_box \
  --method sam \
  --sam-model-id facebook/sam-vit-base \
  --device cuda \
  --max-frames 32 \
  --image-height 256 \
  --image-width 256 \
  --save-visuals
```

Mask-only results on WebUOT fish30 all30:

| Method | Prompt / Input | Precision | Recall | F1 | IoU | Pred Coverage | GT Coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Temporal RGB pseudo-mask | RGB only | 0.3279 | 0.1783 | 0.2310 | 0.1306 | 5.92% | 10.89% |
| GrabCut-box | WebUOT GT box | 1.0000 | 0.4577 | 0.6279 | 0.4577 | 4.98% | 10.89% |
| SAM-base-box | WebUOT GT box | 0.9778 | 0.4835 | 0.6471 | 0.4783 | 5.39% | 10.89% |

Interpretation:

- Box-prompted SAM/GrabCut are much stronger than temporal RGB masks, but they
  assume a box detector/tracker at test time.
- Aqua-D4RT query heads should not be claimed to beat SAM-box on 2D mask F1.
  The research claim must instead focus on query-level static geometry and
  downstream reconstruction/SLAM, or add a detector-generated box pipeline as a
  fair non-oracle prefilter.
- This result sharpens the next ICRA step: evaluate static point/map quality
  and COLMAP/SLAM behavior, not only mask F1.

## 2026-06-22 Update: Non-Oracle GroundingDINO + SAM Baseline

We added a fully non-oracle detector baseline:

```text
scripts/eval_aqua_detector_sam_prefilter_masks.py
```

Unlike the WebUOT-box SAM/GrabCut rows above, this script predicts boxes from
RGB frames and a text prompt using `IDEA-Research/grounding-dino-tiny`; the
optional SAM row uses those predicted boxes as prompts. No WebUOT boxes are used
at test time.

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 /media/data/u24conda/envs/longlive/bin/python scripts/eval_aqua_detector_sam_prefilter_masks.py \
  --manifest-list data/real_underwater/webuot238_fish30/splits/train_24_manifests.txt \
  --manifest-list data/real_underwater/webuot238_fish30/splits/val_6_manifests.txt \
  --output-dir tmp/aqua_prefilter_masks/webuot238_fish30_groundingdino_box_all30 \
  --method detector_box \
  --image-height 256 --image-width 256 --max-frames 32 \
  --prompt 'underwater fish.' \
  --box-threshold 0.30 --text-threshold 0.20 --max-boxes-per-frame 8 \
  --device cuda --save-visuals

CUDA_VISIBLE_DEVICES=0 /media/data/u24conda/envs/longlive/bin/python scripts/eval_aqua_detector_sam_prefilter_masks.py \
  --manifest-list data/real_underwater/webuot238_fish30/splits/train_24_manifests.txt \
  --manifest-list data/real_underwater/webuot238_fish30/splits/val_6_manifests.txt \
  --output-dir tmp/aqua_prefilter_masks/webuot238_fish30_groundingdino_sam_all30 \
  --method detector_sam \
  --image-height 256 --image-width 256 --max-frames 32 \
  --prompt 'underwater fish.' \
  --box-threshold 0.30 --text-threshold 0.20 --max-boxes-per-frame 8 \
  --sam-model-id facebook/sam-vit-base \
  --device cuda --save-visuals
```

Mask-only results on WebUOT fish30 all30:

| Method | Prompt / Input | Precision | Recall | F1 | IoU | Pred Coverage | GT Coverage | Boxes / Frame |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Temporal RGB pseudo-mask | RGB only | 0.3279 | 0.1783 | 0.2310 | 0.1306 | 5.92% | 10.89% | - |
| GroundingDINO-box | RGB + text prompt | 0.3902 | 0.8441 | 0.5337 | 0.3639 | 23.56% | 10.89% | 4.16 |
| GroundingDINO+SAM-base | predicted box prompt | 0.3530 | 0.5049 | 0.4155 | 0.2622 | 15.58% | 10.89% | 4.16 |
| GrabCut-box | WebUOT GT box | 1.0000 | 0.4577 | 0.6279 | 0.4577 | 4.98% | 10.89% | oracle |
| SAM-base-box | WebUOT GT box | 0.9778 | 0.4835 | 0.6471 | 0.4783 | 5.39% | 10.89% | oracle |

Interpretation:

- GroundingDINO-box is the strongest fully non-oracle 2D prefilter baseline so
  far: it improves over temporal RGB F1 (0.2310 -> 0.5337) and has high recall,
  but over-masks relative to WebUOT tracked-target boxes.
- GroundingDINO+SAM has lower F1 under WebUOT bbox-mask evaluation because SAM
  follows object interiors while the GT is a filled tracking box. This is an
  evaluation-label caveat, not a proof that SAM is worse at instance
  segmentation.
- The oracle-ish GT-box SAM/GrabCut rows remain stronger on 2D mask F1, so the
  paper should not claim superiority over prompted segmentation on mask F1.
  The main claim should stay on query-level static geometry cleanup and
  downstream retention/registration Pareto.
