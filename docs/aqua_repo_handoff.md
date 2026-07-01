# Aqua-D4RT Handoff

This repo is trimmed for source handoff, not for shipping weights or bulky run artifacts.

## What Is Kept

- Core model code: `src/model/`, `src/losses/`, `src/eval/`, `src/data/`
- Training / inference entrypoints: `train.py`, `infer_track_3d.py`
- Aqua configs: `configs/train_aqua_*.yaml`
- Reproducible analysis / visualization scripts: `scripts/eval_aqua_*.py`, `scripts/visualize_aqua_*.py`, `scripts/generate_aqua_*.py`, `scripts/plot_aqua_*.py`
- Paper-facing claim notes: `docs/aqua_icra_assessment.md`, `docs/aqua_d4rt.md`

## Core Claim

Aqua-D4RT adds query-level transient awareness to D4RT for underwater dynamic scenes.
It predicts:

- `dynamic_object_head`
- `particle_head`

and fuses them into:

`static_confidence = sigmoid(confidence) * (1 - sigmoid(dynamic)) * (1 - sigmoid(particle))`

The goal is cleaner static query-maps and better retention/control for mapping and front-end use.

## What It Is Not

- Not an underwater image restoration model
- Not a fish segmentation SOTA paper
- Not a one-pass online SLAM system
- Not a claim of broad underwater SLAM/SfM SOTA

## Best Starting Files

1. `README.md`
2. `docs/aqua_icra_assessment.md`
3. `docs/aqua_d4rt.md`
4. `configs/train_aqua_real_synth_mix_headonly.yaml`
5. `scripts/visualize_aqua_baseline_comparison_case.py`
6. `scripts/eval_aqua_static_map_contamination.py`
7. `scripts/eval_aqua_downstream_slam_proxy.py`

## Checkpoint

The working Aqua checkpoint is intentionally not committed.
Copy it locally before running:

`output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt`

## Recommended Run Pattern

1. Prepare the checkpoint locally.
2. Run the Aqua visualization / evaluation scripts on WebUOT or Tank clips.
3. Use the claim notes above to keep the story honest.

