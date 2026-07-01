# Aqua-D4RT ICRA Assessment

## Verdict

Aqua-D4RT can have ICRA-level potential, but only if the paper is framed as a
robotics mapping/localization robustness method, not as a generic underwater
image enhancement or segmentation paper.

Current prototype status is not yet ICRA-ready. It is a promising direction
with a clear path to a publishable systems/method paper if the experiments show
that query-level transient filtering improves static 3D reconstruction and
downstream SLAM under fish and marine-snow interference.

## Why It Can Be Valuable

- Underwater robotics has a real operational pain point: visual SLAM and mapping
  degrade under dynamic fish, divers, suspended particles, turbidity, and weak
  texture.
- Existing marine-snow work has shown that particles can create false features
  and hurt visual SLAM, and suppression can improve SLAM robustness.
- D4RT provides a modern feedforward 4D reconstruction/query interface. Adding
  transient-aware query filtering is a cleaner robotics mapping primitive than
  only masking RGB frames before a classical SLAM front end.
- The useful contribution is not "we segment fish"; it is "we recover a cleaner
  static map / trajectory by filtering transient query geometry before static
  aggregation."

## What Is Not Enough

- Synthetic fish overlays alone are not enough for ICRA unless they are tied to
  downstream mapping/pose improvements.
- A demo that visually hides fish is not enough.
- Training only binary heads without showing geometry/SLAM gains is not enough.
- If the method relies entirely on external segmentation masks at test time,
  novelty is weak; it becomes a prefilter baseline.

## Core Claim

Suggested main claim:

> Query-level transient prediction in a D4RT-style 4D reconstruction model
> enables robust static geometry and camera/trajectory estimation in underwater
> videos with dynamic fish and marine-snow interference.

This is stronger than:

> We remove fish and particles from underwater videos.

The latter sounds like image restoration; the former is robotics mapping.

## Minimum Experiment Package

### Exp 1: Failure Mode Baseline

Run OpenD4RT on underwater or synthetic underwater clips with fish/particles.

Show:

- fish tracks become transient geometry;
- particles create noisy points/features;
- static map quality degrades;
- confidence alone is insufficient to remove all transient interference.

### Exp 2: Synthetic Controlled Benchmark

Use clean/static background clips and inject:

- WaterMask/UIIS fish cutouts with exact dynamic masks;
- procedural or benchmark marine snow;
- underwater color/scattering degradation.

Report:

- fish/particle query F1;
- static query precision/recall;
- background point contamination rate;
- clean/static geometry consistency.

### Exp 3: D4RT vs Prefilter vs Aqua-D4RT

Compare:

- OpenD4RT;
- OpenD4RT + external segmentation prefilter;
- Aqua-D4RT query-level transient heads;
- ablations without fish head / particle head.

The paper needs to show that query-level filtering is better than only masking
input images.

### Exp 4: Downstream Robotics Validation

Run COLMAP / ORB-SLAM3 / another visual odometry baseline on:

- raw corrupted videos;
- segmentation-prefiltered videos;
- Aqua-D4RT-filtered static points or static pseudo-frames.

Report:

- success rate;
- ATE/RPE if ground truth exists;
- tracked feature survival;
- map cleanliness.

## Must-Have Qualitative Figures

1. Failure case of raw OpenD4RT: fish/particles polluting tracks/point cloud.
2. Aqua-D4RT pipeline: corrupted video -> query transient heads -> static_conf
   -> filtered static geometry.
3. Side-by-side maps: raw D4RT vs segmentation prefilter vs Aqua-D4RT.
4. Real underwater qualitative video, even if not used for quantitative claims.

## Risk Assessment

- **High risk:** no real underwater video. Mitigation: use public underwater
  datasets for qualitative validation and synthetic controlled benchmark for
  quantitative claims.
- **High risk:** synthetic-to-real gap. Mitigation: use diverse fish cutouts,
  real underwater backgrounds, marine-snow benchmark statistics, and real-video
  qualitative cases.
- **Medium risk:** contribution may look incremental. Mitigation: emphasize
  D4RT query-level geometry filtering and downstream SLAM/mapping gains.
- **Medium risk:** D4RT checkpoint has no underwater training. Mitigation:
  freeze encoder first, train transient heads/adapters, then low-LR decoder
  fine-tuning.

## Acceptance-Level Bar

For ICRA, the project becomes credible if it can show:

- clear real robotics motivation;
- measurable SLAM/reconstruction improvement under dynamic underwater
  interference;
- stronger performance than segmentation-prefilter baselines;
- at least one real underwater qualitative sequence;
- ablations proving fish and particle heads both matter.

Without those, it is more likely to read as a workshop/demo project.

## Useful References

- D4RT project: https://d4rt-paper.github.io/
- D4RT paper: https://arxiv.org/abs/2512.08924
- Marine snow and underwater visual SLAM: https://openaccess.thecvf.com/content/CVPR2022W/IMW/papers/Hodne_Detecting_and_Suppressing_Marine_Snow_for_Underwater_Visual_SLAM_CVPRW_2022_paper.pdf
- Marine Snow Removal Benchmark: https://github.com/ychtanaka/marine-snow
- WaterMask/UIIS: https://github.com/LiamLian0727/WaterMask
