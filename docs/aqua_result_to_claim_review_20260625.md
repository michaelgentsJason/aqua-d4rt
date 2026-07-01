# Aqua-D4RT Result-to-Claim Review, 2026-06-25

This note audits whether the current Aqua-D4RT evidence supports an ICRA-level
claim, with special attention to SAM/GroundingDINO baselines and the risk that
the method looks like "only two extra heads on D4RT".

## Verdict

Claim support: **partial**.

The narrow core claim is supported:

> Aqua-D4RT introduces D4RT-native query-level transient confidence for
> underwater dynamic scenes, reducing static query-map contamination and
> enabling controllable retention/downstream operating points.

The broad claims are not supported:

- not 2D fish segmentation SOTA,
- not broad underwater SLAM/SfM SOTA,
- not universally better than SAM/GroundingDINO,
- not a proven one-pass online downstream selector,
- not natural real-dynamic AQUALOC/VAROS generalization yet.

## Is "Only Two Heads" Too Thin?

If the contribution is framed as "we add two heads to D4RT", then yes, it looks
thin.

The contribution is defensible only if framed as a system/formulation:

1. Query-level transient-aware static mapping, not dense 2D segmentation.
2. Underwater transient taxonomy: dynamic object vs particle/marine snow.
3. D4RT-native static confidence for tracked video/3D queries.
4. Contamination/retention/downstream Pareto metrics and benchmark protocol.
5. Retention/calibration selectors showing task-dependent operating points.

The simplicity of the architecture can be a strength, but only if the paper
shows that query-level filtering is the right abstraction for static mapping.
The architecture alone is not the contribution.

## SAM/GroundingDINO Risk

Reviewer attack is likely. The current results show:

- WebUOT fish30 mask-level:
  - GroundingDINO-box F1 0.5337.
  - GroundingDINO+SAM F1 0.4155 under WebUOT bbox-mask labels.
  - GT-box SAM-base F1 0.6471, but oracle-ish.
  - Aqua WebUOT dynamic F1 is lower than these segmentation/prompted baselines.
- WebUOT fish30 static-map/ORB:
  - GroundingDINO-box can beat Aqua on bbox-mask contamination metrics.
- WebUOT dynamic100:
  - GroundingDINO-box becomes over-aggressive: query contamination is low
    (2.65%) but static retention is only 35.30% and E success only 17.31%.
  - Aqua low-threshold mode keeps much higher retention and E success while
    still reducing contamination strongly.

Unsafe statements:

- Aqua-D4RT outperforms SAM/GroundingDINO.
- Foundation models fail underwater.
- Aqua-D4RT is better at fish segmentation.
- Aqua-D4RT achieves dynamic segmentation SOTA.
- Aqua-D4RT improves SLAM/pose universally.

Safe framing:

> Detector/SAM prefilters are strong for image-level object removal, especially
> under WebUOT target-box labels. Aqua-D4RT instead estimates D4RT query-level
> static reliability and exposes retention-aware operating points for static
> geometry and downstream matching.

## Is Aqua-D4RT SOTA?

Not in the broad sense.

The safest narrow claim is:

> On our evaluated D4RT-native underwater transient mapping protocol, Aqua-D4RT
> gives the strongest contamination/retention trade-off among D4RT query
> filtering variants and exposes downstream retention Pareto points.

Avoid using "SOTA" in the title or abstract unless additional public real
dynamic benchmarks and stronger retention-matched baselines are added.

## What The Current Tables Actually Support

Strongly supported:

- Synthetic full-mask static query-map cleanup:
  raw 10.82% -> Aqua 0.39% at 93.00% retention.
- Dynamic and particle heads are both necessary:
  no-dynamic and no-particle ablations fail different parts of the task.
- Aqua is not just OpenD4RT confidence:
  confidence-only/raw remains 10.82% contamination on synthetic.
- DINO-box is not a free downstream replacement on larger WebUOT dynamic100:
  it over-masks static structure and collapses E success.
- R099 supports a downstream Pareto on Tank stress4:
  aggregate contamination and pose/registration metrics improve vs same-run raw,
  with a multi-pass selector caveat.

Partially supported:

- Real-world transfer:
  WebUOT all238 and AQUALOC sanity are useful, but labels/injection are imperfect.
- Adaptive threshold selection:
  calibration scorer improves a registration-friendly operating point, but it is
  not a main SOTA result.

Not supported:

- superiority over SAM/DINO for 2D masks,
- universal pose improvement,
- full real dynamic underwater SLAM,
- one-pass online deployment.

## Recommended Paper Claim

Title-style:

> Aqua-D4RT: Transient-Aware Query Confidence for Underwater Static Mapping

Abstract-style:

> Aqua-D4RT augments D4RT with underwater transient-aware query heads for dynamic
> objects and particles. The resulting static confidence reduces transient
> contamination in static query maps and provides retention-aware operating
> points for downstream matching and reconstruction.

Contribution list:

1. A query-level formulation for underwater transient-aware static mapping on
   top of D4RT.
2. Dynamic-object and particle heads with a D4RT-native static confidence score.
3. Static query-map, voxel, feature, match, and pose-front-end contamination
   metrics.
4. Strong baselines including temporal RGB, GroundingDINO, GroundingDINO+SAM,
   and oracle/prompted masks, with explicit caveats.
5. Retention/calibration selectors showing a controllable contamination vs
   registration/pose Pareto.

## Highest-Value Missing Evidence

To make the ICRA case stronger:

1. Add at least one more real underwater dataset or manually labeled subset with
   complete dynamic-object and particle labels.
2. Plot full Pareto curves instead of only fixed thresholds:
   query contamination vs retention, feature/match contamination vs E success,
   and registration/ATE/RPE when available.
3. Add retention-matched SAM/DINO comparisons:
   compare methods at matched static retention or matched E success, not only
   at their default masks.
4. Add statistical confidence:
   mean/std or bootstrap confidence intervals over clips and seeds.
5. Add explicit failure cases:
   low contrast, dense particles, static-looking fish, over-filtering, and
   camera-motion-heavy clips.
6. If time allows, implement a one-pass or lightweight online selector; keep
   R099 clearly labeled as multi-pass otherwise.

## Decision

Proceed toward paper writing, but write with a narrow and honest claim.

Do not spend the next effort trying to beat SAM/DINO at fish contours. Spend it
on:

- retention-matched baseline analysis,
- cleaner Pareto figures,
- real-data label/provenance strengthening,
- and claim-safe writing.

## Update: 2026-06-30 Degradation Evidence

The new R117/R119/R120 results make the narrow claim materially stronger.

### What changed

- Synthetic degradation robustness is now clear:
  - raw 10.82% query contamination -> Aqua @0.25 2.11% at 95.95% retention;
  - Aqua @0.55 reaches 0.97% contamination at 89.70% retention;
  - transient-query AUROC/AP is 0.9678 / 0.7959.
- External-background AQUALOC sanity is now explicit:
  - raw 14.43% -> Aqua @0.11 8.83% at 88.86% retention;
  - cleaner thresholds exist, but they become too conservative for downstream use.
- Calibration/selector evidence is now checkpoint-free:
  - the selector improves the operating-point story, but mostly confirms that
    fixed low thresholds already capture much of the useful Pareto.

### Revised best claim

> Aqua-D4RT estimates D4RT-native query-level static reliability for underwater
> dynamic scenes, reducing transient contamination and exposing a controllable
> contamination/retention/front-end Pareto under water-specific degradation.

### What remains unsafe

- do not claim broad underwater SLAM/SfM SOTA,
- do not claim better 2D fish segmentation than SAM/GroundingDINO,
- do not claim one-pass online deployment,
- do not widen AQUALOC into a general dynamic-scene win.

### Paper implication

The paper is now best written as a narrow degradation-aware reliability story:
robust query-level static filtering, controllable operating points, and honest
failure boundaries. That is a much stronger ICRA shape than the earlier
"two-heads on D4RT" framing, but still not a universal SLAM claim.
