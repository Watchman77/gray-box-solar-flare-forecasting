# Phase H1c — physical-shift decomposition protocol

Status: **PRE-EXECUTION DIAGNOSTIC FREEZE — 6 October 2026**

H1c is a diagnostic decomposition of the robust H1b physical-state shift. It does not alter the A–G routing policy and does not select a physical trust gate.

## 1. Motivation

H1b showed that physical-state distance and q99 OOD rate are elevated in 2026 even after removing a deterministic MPSV redundancy. However, the same physical distance is negatively associated with case-level fusion squared error. In addition, the largest family shifts occur in the latest magnetic-gradient and inclination summaries.

The next task is therefore not to promote physical OOD into routing. It is to determine **what the physical shift represents**.

## 2. Evidence boundary

Cycle-25 and 2026 outcomes have already been inspected. All H1c analyses are retrospective/post-hoc.

H1c may diagnose:
- temporal evolution;
- class composition;
- repeated-window dependence;
- physical-family contributions;
- source/pipeline boundaries.

It must not claim causal solar-cycle physics unless source/processing alternatives are ruled out.

## 3. Frozen H1c analyses

### H1c-A — Year-by-year applicability
For each calendar year:
- cases and active-region components;
- median and interquartile physical Mahalanobis distance;
- q99 physical-OOD rate;
- positive prevalence among known labels.

Primary question: is the 2026 increase gradual through Cycle 25 or an abrupt boundary effect?

### H1c-B — Class-conditional applicability
Within each evidence regime and year:
- median distance and OOD rate separately for y=0 and y=1;
- physical probability distribution by class;
- flare/non-flare prevalence.

Primary question: is the negative error correlation explained by flare windows occupying more magnetically extreme states?

### H1c-C — Active-region-level sensitivity
Aggregate by `region_component_id` within regime/year:
- number of forecast windows;
- median physical distance;
- maximum physical distance;
- OOD fraction;
- whether the component contains any positive 72 h window.

Report AR-level medians and distributions so repeated 96-minute windows cannot dominate the inference.

### H1c-D — Physical-family shift decomposition
For every MPSV axis:
- regime median;
- regime IQR;
- difference in median relative to training;
- standardized median shift relative to training IQR;
- year-by-year median.

Rank axes by absolute 2026 standardized median shift. This ranking is descriptive and must not be converted automatically into a policy feature selection.

### H1c-E — Source/cohort boundary audit
Use the recovered frozen ledger/tensor identity and source cohort chronology to compare:
- 2024;
- 2025;
- 2026;
- late-2025 versus early-2026 where support permits.

Flag an abrupt discontinuity at the 2026 supplementary boundary as possible source/processing shift evidence. Do not label it solar-cycle physics without additional evidence.

### H1c-F — Model-agreement versus physical-applicability contrast
Join H1b physical distance to the already-frozen statistical trust indicators when available:
- fusion entropy;
- fusion seed standard deviation;
- SHARP-AIA probability gap.

Report correlations and year/regime medians. The purpose is to test the Gray-Box proposition that physical applicability carries information not captured by model agreement.

## 4. No policy action

H1c must end with one of three outcomes:

- `PHYSICAL_SHIFT_ROBUST_REGIME_MONITOR_ONLY`
- `PHYSICAL_SHIFT_POSSIBLE_SOURCE_BOUNDARY_CONFOUND`
- `PHYSICAL_SHIFT_NOT_ROBUST`

A case-level physical gate is not permitted from H1c. Any H2 routing proposal requires a separate predeclared policy-development protocol using earlier support only.
