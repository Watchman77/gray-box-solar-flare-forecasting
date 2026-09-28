# Proposed five-layer protocol

Status: research draft. Numerical thresholds and final methods are unselected. No operational validity or formal coverage guarantee is asserted by this document.

| Layer | Input | Candidate method | Output | Evaluation | Failure mode | Proposed decision rule |
|---|---|---|---|---|---|---|
| Data-driven | Past, availability-checked SHARP histories; optional aligned AIA sequences | Reproduce SHARP baseline; evaluate AIA branch on identical cases | Per-horizon raw scores and modality-quality metadata | AP, ROC-AUC, Brier, log loss, TSS and error counts at preselected operating points | Missing history, bad channel, source drift, unavailable observation | Score only a branch whose declared input contract is satisfied; record unsupported branches. |
| Physically interpretable | Magnetic flux, shear, current/helicity or other documented physical indicators | Explicit magnetic-complexity proxy or interpretable model; ablation against its removal | Physical indicators, proxy score and documented consistency checks | Incremental paired skill, stability and interpretable failure analysis | Invalid physical proxy, invalid geometry/units, spurious association | Flag invalid physical inputs; avoid asserting deterministic flare laws or MHD compliance. |
| Fusion | Supported branch scores/features, physical indicators, quality and availability masks | Begin with late fusion; compare intermediate fusion and a quality-aware policy | Combined score or selected fallback score, branch provenance | Paired improvement on matched cases; outage robustness; full-population performance | One unreliable branch dominates; missing-modality behaviour is untrained | Use only supported fusion modes; route to a separately validated/calibrated fallback when its eligibility conditions hold. |
| Uncertainty | Scores, earlier calibration outcomes, and declared ensemble/nonconformity inputs | Platt/isotonic candidates; declared ensemble or task-appropriate conformal method | Calibrated probability and clearly typed uncertainty output | Brier/Brier skill, log loss, reliability, slope/intercept, ECE; empirical coverage and set size or interval width where applicable | Sparse positives, dependence, distribution shift, stale calibration | Report uncertainty validity flags; apply only policies selected on earlier validation data. |
| Operational safety | Quality, availability, timing, OOD, uncertainty and branch-policy outputs | Frozen state machine, with thresholds to be estimated on designated validation data | Normal/degraded/abstain state, reason codes and fallback identifier | Risk–coverage, false alarms/misses, retained-case and all-case metrics, latency and availability | Confident errors, excessive abstention, unsupported fallback | Normal when all required checks pass; degraded when a validated reduced mode is eligible; abstain if no supported forecast mode remains. |

## Uncertainty targets

Probability calibration, predictive disagreement, conformal prediction sets and confidence intervals for aggregate metrics answer different questions. Record them separately.

For binary occurrence, classification conformal sets are a candidate. They are not automatically intervals for an unknown flare probability. Conformalized quantile regression requires a defined regression target and its own label/data protocol. Temporal dependence and distribution shift must be evaluated explicitly.

## Time ordering

Each rolling iteration uses model-development/training data, a later past calibration block, a later past policy-validation block and then an unseen evaluation block. The exact dates, cadence and update schedule must be frozen in an experiment-specific protocol. Data labels are available for fitting only after the forecast window and declared outcome-reporting delay have elapsed.

The AR-disjoint and event-disjoint evaluations must be named separately from ordinary chronological deployment evaluation. Purge overlapping outcome windows at relevant boundaries and record exclusions.

## Operational states

| State | Meaning | Required record |
|---|---|---|
| `normal` | The approved forecast mode satisfies its current input and reliability checks. | Probability, model/calibrator/policy versions, quality and uncertainty fields. |
| `degraded` | A validated reduced-input or fallback mode is being used. | Probability if supported, fallback identity, unavailable inputs and reason codes. |
| `abstain` | No validated mode currently satisfies the decision policy. | No fabricated probability; reason codes and intended escalation/fallback action. |

Numerical decision thresholds remain unset. State names express policy behaviour; they do not imply that any particular uncertainty estimate guarantees correctness.
