# Final 72-hour Gray-Box scientific narrative

Status: **frozen 72-hour interpretation**. This document records the completed 72-hour scientific story before any 24-hour or 3-hour replication begins. It does not reopen model fitting, calibration, conformal thresholds, policy selection, or 2026 interpretation.

## Central question

The project began as a multimodal forecasting extension: combine temporal SHARP magnetic information with AIA image information, calibrate probabilities, quantify uncertainty, and test whether fusion improves flare prediction. The completed 72-hour work supports a more important question:

> **When should an operational user trust a solar-flare forecast, when should the system degrade to a validated fallback, and when should it withhold a forecast?**

The strongest contribution is therefore not universal fusion superiority. It is an experimentally evaluated **issue / degrade / withhold** trust-routing framework under temporal shift, uncertainty degradation, and realistic data failure.

## Frozen architecture

The current 72-hour operational states are:

- **NORMAL** — SHARP and AIA are technically available and the frozen multimodal trust gates pass. Issue the fixed 0.5/0.5 SHARP+AIA fusion.
- **DEGRADED** — NORMAL is false but SHARP remains technically available. Issue SHARP and mark the forecast DEGRADED.
- **ABSTAIN** — SHARP is technically unavailable. Withhold the forecast. Automatic AIA-only fallback is not enabled.

The UQ layer includes probability calibration, Mondrian class-conditional conformal sets, three-seed disagreement, fusion entropy, SHARP-AIA probability gap, and past-only rolling reliability monitoring.

## Data and chronology

The Gray-Box candidate inventory contains 153,366 forecast records spanning May 2010 through the 2026 extension. Of these, 113,433 have matched AIA-SHARP inputs: 55,871 Cycle-24 cases, 45,403 primary Cycle-25 cases, and 12,159 supplementary-2026 cases.

The frozen 72-hour evaluation schedule contains 96,596 cases: 25,586 training cases plus 71,010 non-training cases divided into model validation (3,905), probability calibration (2,831), conformal calibration (4,168), policy validation (13,142), retrospective Cycle 25 (35,846), and supplementary 2026 (11,118).

Event and region reconciliation were completed before final Gray-Box interpretation. Original labels and predictions were preserved, source-supported corrections were versioned separately, and unresolved event-region associations remained visible rather than being silently converted to negatives.

## A-G evidence chain

### A — Cross-cycle reliability

Predictive performance degrades materially from the earlier Cycle-24 development regime into later Cycle-25 and 2026 conditions.

| Regime | SHARP TSS | AIA TSS | Fusion TSS |
|---|---:|---:|---:|
| Earlier Cycle-24 development | 0.689 | 0.333 | 0.630 |
| Cycle 25 | 0.541 | 0.340 | 0.488 |
| 2026 | 0.323 | 0.074 | 0.292 |

Flare-class conformal coverage deteriorates in parallel. For SHARP/AIA/fusion it falls approximately 0.806/0.812/0.838 -> 0.735/0.620/0.692 -> 0.435/0.064/0.346.

The careful claim is that **models developed on earlier Cycle-24-era data degrade when transferred into later Cycle-25 conditions**, not that “Cycle 24 itself degraded.”

### B — Calibration diagnostics

Calibration also worsens across regimes. SHARP Brier Skill Score falls approximately 0.336 -> 0.215 -> 0.160. Fusion Brier Skill Score falls approximately 0.247 -> 0.176 -> 0.097. Expected calibration error increases for all branches. The 2026 calibration slopes/intercepts support a genuine temporal calibration problem rather than a discrimination-only effect.

### C — Applicability and shift

Seed spread, fusion entropy, and SHARP-AIA gap remain useful **within a regime** for ranking case-level risk, but they are poor standalone **global shift detectors**. Into 2026 the signals become smaller even while flare reliability deteriorates.

The exact historical SHARP feature-space OOD rerun remains provenance-limited because the original fitted feature-distance artifacts are no longer available. No substitute OOD result is fabricated.

### D — AR and event disjointness

Mapped region-component overlap with the training set is zero for all evaluated non-training chronological roles. The 2026 degradation persists when results are aggregated at AR level, so it is not merely repeated-window inflation.

Exact event-disjoint detection and lead-time recomputation remain provenance-limited because the original event-link artifacts are unavailable in the current execution environment.

### E — Real AIA acquisition failures

Actual upstream AIA acquisition/recovery inventories contain 6,461 incident samples across 2025-2026. A total of 2,290 map exactly to Gray-Box cases.

Replaying all mapped AIA acquisition incidents reduces fusion availability, but the frozen SHARP fallback preserves 100% issuance when SHARP is available.

The incidents are strongly clustered. In 2025, 332/359 empirical episodes are multi-sample clusters and 6,172/6,199 failed samples occur inside clustered episodes. Median episode span is 19.2 h and the maximum is 124.8 h. Of the mapped Gray-Box exposure, 2,085 cases occur in long episodes of more than ten failed samples.

This means independent random masking is an incomplete operational missingness model.

### F — AIA-only fallback feasibility

Automatic AIA-only degraded issuance is **not supported**.

| Regime | AIA TSS | Recall | AP | Flare conformal coverage | Singleton rate |
|---|---:|---:|---:|---:|---:|
| Earlier Cycle-24 | 0.343 | 0.911 | 0.122 | 0.837 | 0.635 |
| Cycle 25 | 0.340 | 0.743 | 0.184 | 0.620 | 0.727 |
| 2026 locked | 0.074 | 0.117 | 0.159 | 0.064 | 0.973 |

The 2026 branch becomes highly decisive while detecting only about 11.7% of positive windows. That is the strongest evidence for retaining ABSTAIN when SHARP is unavailable.

### G — Prospective-style rolling replay

The frozen v3 policy is replayed chronologically. Outcomes become visible only after the 72-hour horizon plus a 24-hour reporting delay. Monitoring never feeds back into the policy.

Nominal Cycle-25 replay:
- availability 1.000
- NORMAL 0.777
- DEGRADED 0.223
- TSS 0.507

Real-AIA-incident Cycle-25 replay:
- availability 1.000
- NORMAL 0.721
- DEGRADED 0.279
- TSS 0.512

Nominal 2026 replay:
- availability 1.000
- NORMAL 0.925
- DEGRADED 0.075
- TSS 0.296

The rolling UQ endpoint in August 2026 shows fusion flare conformal coverage about 0.621 while singleton rate is about 0.934 and median entropy, seed spread, and modality gap are low. Internal agreement therefore does not guarantee cross-regime reliability.

## Final interpretation

The 72-hour results support five central conclusions.

1. **Operational reliability is not equivalent to raw discrimination.**
2. **Uncertainty estimates can become overconfident under temporal/domain shift.**
3. **Model agreement is useful within-regime but is not a reliable standalone detector of global shift.**
4. **Real observational failure is strongly correlated in time and region, making fallback-aware evaluation necessary.**
5. **SHARP is currently the defensible degraded branch; AIA-only automatic fallback is not supported.**

## Claim boundaries

Supported:
- outage-resilient SHARP fallback when SHARP remains technically available;
- strongly clustered AIA acquisition/recovery failures;
- failure of AIA-only automatic fallback under the frozen 72-hour evidence;
- within-regime selective-risk value but global-shift weakness of disagreement/entropy;
- degradation of class-conditional conformal reliability under later temporal shift;
- mapped AR-component disjointness from training and AR-level persistence of 2026 degradation.

Supported with caution:
- the rolling evaluation is **prospective-style**, not a true prospective deployment trial.

Not supported:
- universal predictive superiority over the strongest constituent model;
- first-principles MHD or PINN modelling;
- conformal coverage guarantees under non-exchangeable future regimes.

Provenance-limited:
- exact historical feature-space OOD rerun;
- exact event-disjoint and lead-time recomputation.

## Transition to 24 hours

The architecture is reusable; the numerical parameters are not. The 24-hour study must re-estimate its own calibration maps, conformal quantiles, alarm threshold, trust cut-offs, and fallback decision from earlier 24-hour data. The 72-hour evidence is now frozen and must not be retroactively altered by later-horizon results.
