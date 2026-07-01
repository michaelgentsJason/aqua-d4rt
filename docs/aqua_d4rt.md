# Aqua-D4RT Prototype

This repository now includes a first-pass Aqua-D4RT prototype for dynamic
underwater scenes. The implementation is intentionally scoped to the 1-2 month
research prototype: query-level transient prediction, synthetic underwater
corruption, static-confidence filtering, and encoder-freezing support.

## Model Outputs

The D4RT heads keep the original outputs and add:

- `dynamic_object_logit`: fish, divers, or large moving foreground objects.
- `particle_logit`: marine snow or near-camera suspended particles.
- `static_confidence`: `sigmoid(confidence) * (1 - sigmoid(dynamic_object_logit)) * (1 - sigmoid(particle_logit))`.

Use `src.model.static_confidence.static_query_mask(outputs, threshold=0.5)` to
select likely static-background queries for point clouds, tracks, or pose
aggregation.

## Training Interface

The optional transient targets are:

- `target["dynamic_object"]`
- `target["particle"]`
- `mask["transient"]`

When `loss.transient.geometry_masking.enabled=true`, labeled fish/particle
queries are removed from geometry losses while still contributing to transient
binary cross-entropy losses.

## Synthetic Underwater Training

`augmentation.underwater_transient` wraps any dataset sample after the raw D4RT
query construction. It can apply:

- water color attenuation and contrast loss,
- moving fish-like elliptical foreground objects,
- marine-snow particle overlays,
- source-query labels for dynamic object and particle supervision.

The wrapper is disabled by default in `configs/train_effective.yaml`.

Run the Phase-A prototype with:

```bash
bash scripts/train_aqua_d4rt_phase_a.sh
```

Useful overrides:

```bash
TOTAL_STEPS=2000 AQUA_APPLY_PROBABILITY=0.75 bash scripts/train_aqua_d4rt_phase_a.sh
AQUA_DYNAMIC_LOSS_WEIGHT=0.2 AQUA_PARTICLE_LOSS_WEIGHT=0.2 bash scripts/train_aqua_d4rt_phase_a.sh
```

Phase A freezes the encoder and trains the decoder/query/head side from the
released OpenD4RT initialization. Phase B can reuse the same script with
`--override fine_tuning.freeze_encoder=false` for low-learning-rate decoder plus
encoder fine-tuning.

