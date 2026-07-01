# Aqua-D4RT

Aqua-D4RT is the handoff repo for our underwater transient-aware D4RT work.
It extends OpenD4RT with query-level static-reliability modeling for dynamic
underwater scenes, focusing on fish, particles, marine snow, turbidity,
non-uniform illumination, low light, blur, and flicker.

The repo is trimmed for the next person to continue experiments and paper work.
Model weights and bulky run artifacts are intentionally not included.

## What this repo is for

- Keep the Aqua-D4RT source code, configs, and paper-facing notes in one place.
- Let the next student reproduce the key claim and continue optimization.
- Provide a clean starting point for mapping / front-end robustness work.

## Current claim

Main idea:

> Aqua-D4RT adds dynamic-object and particle heads to D4RT and fuses them into
> `static_confidence` for transient-aware static query-map construction.

In plain terms, Aqua-D4RT is **not** an underwater image restoration model and
**not** a fish-segmentation paper. It is a query-level filtering / retention
method for cleaner static geometry and more controllable downstream behavior.

## What was kept

- Core model code: `src/model/`, `src/losses/`, `src/eval/`, `src/data/`
- Training / inference entrypoints: `train.py`, `infer_track_3d.py`
- Aqua configs: `configs/train_aqua_*.yaml`
- Aqua analysis / visualization scripts: `scripts/eval_aqua_*.py`,
  `scripts/visualize_aqua_*.py`, `scripts/generate_aqua_*.py`,
  `scripts/plot_aqua_*.py`
- Key paper notes:
  - `docs/aqua_repo_handoff.md`
  - `docs/aqua_icra_assessment.md`
  - `docs/aqua_d4rt.md`
  - `docs/aqua_d4rt_teacher_report_20260630.md`

## What was intentionally left out

- Checkpoints and weights
- `data/`, `output/`, `tmp/`, and other bulky experiment artifacts
- Redundant logs and scratch outputs

## Best files to read first

1. `docs/aqua_repo_handoff.md`
2. `docs/aqua_icra_assessment.md`
3. `docs/aqua_d4rt.md`
4. `configs/train_aqua_real_synth_mix_headonly.yaml`
5. `scripts/visualize_aqua_mapping_claim_case.py`
6. `scripts/eval_aqua_static_map_contamination.py`
7. `scripts/eval_aqua_downstream_slam_proxy.py`

## Model summary

The Aqua heads keep the D4RT interface and add:

- `dynamic_object_head`
- `particle_head`
- `static_confidence`

The working formula is:

```text
static_confidence = sigmoid(confidence)
                  * (1 - sigmoid(dynamic))
                  * (1 - sigmoid(particle))
```

The current main checkpoint is not committed here. Copy it locally before
running Aqua experiments:

```text
output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt
```

## Recommended workflow for the next person

1. Read the handoff docs above.
2. Copy the checkpoint locally.
3. Run the visualization / evaluation scripts on WebUOT, AQUALOC, or Tank
   cases.
4. Use the tables in `figures/aqua_paper_tables/` and the docs to keep the
   claim honest.
5. If needed, continue only with calibration / selector work before retraining
   larger parts of the model.

## Useful script entrypoints

- `scripts/visualize_aqua_baseline_comparison_case.py`
- `scripts/visualize_aqua_mapping_claim_case.py`
- `scripts/visualize_aqua_compact_mapping_claim.py`
- `scripts/visualize_aqua_degradation_hero_figure.py`
- `scripts/eval_aqua_transient_heads.py`
- `scripts/eval_aqua_prefilter_baselines.py`
- `scripts/eval_aqua_static_map_contamination.py`
- `scripts/eval_aqua_downstream_slam_proxy.py`

## Repo origin

This repo was exported from the OpenD4RT codebase and then narrowed down for
Aqua-D4RT handoff. The goal is continuity, not a full release archive.
