# Aqua-D4RT Matched Baseline / Pareto Audit, 2026-06-26

本轮目标是回应一个很可能出现的 reviewer 问题：

> 如果 GroundingDINO/SAM 这类强 2D prefilter 很会抠鱼，Aqua-D4RT 在主 claim 上到底还有没有优势？

结论要诚实写：

- 在 WebUOT fish30 的 tracked-target bbox 指标上，GroundingDINO-box 是非常强的 image-level baseline，Aqua 不能声称全面优于它。
- 在更大 WebUOT dynamic100 上，GroundingDINO-box 会明显 over-mask：污染很低，但 static retention 和 ORB/SfM front-end 成功率大幅下降。
- Aqua-D4RT 的优势不应写成“比 SAM/DINO 更会分割鱼”，而应写成 D4RT query-level static reliability 和可控 contamination/retention/front-end Pareto。

## Artifacts

Implementation:

- `scripts/analyze_aqua_matched_baseline_pareto.py`

Outputs:

- `figures/aqua_matched_baselines_20260626/operating_point_table.csv`
- `figures/aqua_matched_baselines_20260626/matched_baseline_table.csv`
- `figures/aqua_matched_baselines_20260626/bootstrap_ci_table.csv`
- `figures/aqua_matched_baselines_20260626/matched_baseline_table.tex`
- `figures/aqua_matched_baselines_20260626/latex_includes.tex`
- `figures/aqua_matched_baselines_20260626/pareto_matched_baselines.pdf`
- `figures/aqua_matched_baselines_20260626/pareto_external_sanity.pdf`

Sources:

- WebUOT fish30 all30/val6 static-map and ORB external DINO/SAM results.
- WebUOT dynamic100 all100/new70 static-map and ORB Aqua-threshold/DINO-box results.
- WebUOT all238 Aqua threshold sweep.
- AQUALOC harbor07 injected-transient sanity sweep.

Bootstrap CI:

- 2,000 bootstrap resamples over clips.
- Static metrics are count-aggregated over query points/voxels.
- ORB feature contamination is feature-count weighted; E success is pair-count weighted; match contamination follows the evaluator's per-clip mean aggregation.

## Key Operating Points

| Dataset | Method | Query contam. | Static ret. | Feature contam. | Match contam. | E success |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| WebUOT fish30 all30 | Raw D4RT | 10.86% | 100.00% | 20.97% | 25.42% | 96.67% |
| WebUOT fish30 all30 | Aqua @0.55 | 4.47% | 75.53% | 3.66% | 5.57% | 64.79% |
| WebUOT fish30 all30 | GroundingDINO-box | 2.19% | 83.93% | 2.75% | 1.13% | 63.54% |
| WebUOT fish30 all30 | GroundingDINO+SAM | 6.40% | 88.76% | 6.48% | 16.50% | 75.42% |
| WebUOT dynamic100 all100 | Raw D4RT | 16.69% | 100.00% | 37.74% | 39.39% | 96.00% |
| WebUOT dynamic100 all100 | Aqua @0.15 | 10.93% | 91.55% | 14.99% | 19.80% | 83.38% |
| WebUOT dynamic100 all100 | Aqua @0.25 | 10.28% | 88.78% | 14.05% | 18.23% | 80.06% |
| WebUOT dynamic100 all100 | Aqua @0.55 | 8.80% | 74.86% | 12.25% | 11.82% | 68.44% |
| WebUOT dynamic100 all100 | GroundingDINO-box | 2.65% | 35.30% | 8.32% | 1.16% | 17.31% |
| WebUOT dynamic100 new70 | Raw D4RT | 18.60% | 100.00% | 44.85% | 44.88% | 94.29% |
| WebUOT dynamic100 new70 | Aqua @0.20 | 11.65% | 90.22% | 17.88% | 20.30% | 80.09% |
| WebUOT dynamic100 new70 | GroundingDINO-box | 2.99% | 34.79% | 9.63% | 1.43% | 17.95% |
| WebUOT all238 | Raw D4RT | 15.24% | 100.00% | 29.89% | 33.19% | 97.74% |
| WebUOT all238 | Aqua @0.45 | 8.92% | 74.67% | 11.48% | 14.81% | 81.33% |
| AQUALOC sanity | Raw D4RT | 14.43% | 100.00% | 51.46% | 37.37% | 73.61% |
| AQUALOC sanity | Aqua @0.10 | 6.00% | 91.32% | 29.21% | 23.79% | 59.03% |

## Matched Baseline Findings

### WebUOT fish30: DINO-box is stronger under bbox labels

On fish30 all30, GroundingDINO-box has lower query contamination and higher retention than Aqua clean-map:

- DINO-box: 2.19% contamination at 83.93% retention.
- Aqua @0.55: 4.47% contamination at 75.53% retention.

At matched static retention, DINO-box also remains better:

- DINO-box: 2.19% contamination at 83.93% retention.
- Aqua @0.35: 5.66% contamination at 84.20% retention.

This should be written as a caveat, not hidden. The WebUOT label is a tracked-target bounding box; detector boxes naturally align with this metric.

### WebUOT dynamic100: DINO-box over-masks, Aqua gives usable front-end trade-off

On dynamic100 all100:

- DINO-box: 2.65% query contamination, but only 35.30% static retention and 17.31% E success.
- Aqua @0.25: 10.28% query contamination, 88.78% retention, 80.06% E success.
- Raw D4RT: 16.69% query contamination, 37.74% feature contamination, 96.00% E success.

Held-out dynamic100 new70 shows the same pattern:

- DINO-box: 2.99% query contamination, 34.79% retention, 17.95% E success.
- Aqua @0.20: 11.65% query contamination, 90.22% retention, 80.09% E success.

This is currently the strongest evidence against the reviewer shortcut “just use GroundingDINO/SAM as a prefilter.”

### WebUOT all238 and AQUALOC sanity

WebUOT all238 supports the same operating-point story:

- Raw: 29.89% feature contamination and 97.74% E success.
- Aqua @0.45: 11.48% feature contamination and 81.33% E success.

AQUALOC harbor07 injected sanity supports external-background robustness but also shows domain shift:

- Raw: 51.46% feature contamination and 73.61% E success.
- Aqua @0.10: 29.21% feature contamination and 59.03% E success.

Do not write this as natural real-dynamic AQUALOC SOTA. It is external-background injected stress sanity.

## Paper Claim Implication

Safe claim:

> Aqua-D4RT estimates D4RT-native query-level static reliability. Compared with raw D4RT queries, it substantially reduces transient query/feature contamination; compared with strong detector/SAM prefilters, it exposes a more useful retention/front-end Pareto on broad dynamic WebUOT stress cases where image-level detectors over-mask static structure.

Unsafe claim:

> Aqua-D4RT beats SAM/DINO at fish segmentation.

Unsafe claim:

> Aqua-D4RT is a general underwater SLAM/SfM SOTA method.

Recommended paper use:

- Put `pareto_matched_baselines.pdf` in the main or appendix figure set.
- Use fish30 DINO-box as an honest strong baseline/caveat.
- Use dynamic100 all100/new70 to show why image-level prefiltering is not sufficient for downstream static geometry.
- Keep R099 as the stronger downstream GT-pose evidence, with the multi-pass caveat.
