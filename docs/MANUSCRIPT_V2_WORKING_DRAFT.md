# Manuscript V2 — working draft

## Working title

**A Gray-Box Trust Framework for 72-Hour Solar Flare Forecasting Under Temporal Shift, Uncertainty Degradation, and Data Failure**

## Central contribution

This paper is not positioned as a fusion-superiority paper. Its central contribution is an experimentally validated operational trust framework for deciding when a solar-flare forecast should be issued normally, degraded to a robust fallback, or withheld under temporal shift, uncertainty degradation, and realistic input failures.

The framework combines learned SHARP and AIA predictors with calibration, conformal uncertainty, statistical agreement signals, an explicit interpretable magnetic-state applicability monitor, data-quality/provenance checks, and a frozen NORMAL / DEGRADED / ABSTAIN routing policy.

---

## Abstract — draft v1

Reliable solar-flare forecasting requires more than high discrimination under a fixed retrospective test set. Operational systems must remain useful when probability calibration drifts, uncertainty estimates degrade, input modalities fail, and the physical feature distribution moves away from the development regime. We present a Gray-Box trust framework for 72-hour M/X-class solar-flare forecasting that combines frozen SHARP and AIA predictors with calibration, conformal uncertainty, statistical agreement measures, an interpretable magnetic-state applicability layer, and an explicit NORMAL / DEGRADED / ABSTAIN routing policy. The framework was developed using time-ordered earlier data and evaluated across later Solar Cycle 25 support and a supplementary 2026 extension, with no policy reselection on later outcomes.

Across regimes, predictive reliability degraded despite increasing model agreement. For SHARP, TSS declined from 0.689 on earlier Cycle-24 development support to 0.541 on Cycle-25 evaluation and 0.323 in 2026. Equal-weight SHARP–AIA fusion did not provide universal predictive gains and was significantly worse than SHARP in Cycle-25 TSS and average precision under paired active-region block bootstrap. A simple 16-feature latest-state SHARP logistic baseline was also competitive, confirming that the contribution is not raw predictive superiority. Conformal flare coverage degraded substantially in later regimes, while entropy, ensemble spread and SHARP–AIA disagreement became smaller, showing that internal statistical agreement can increase even as rare-event reliability worsens.

The frozen routing policy preserved forecast availability during realistic AIA acquisition failures by degrading to SHARP rather than relying automatically on an unstable AIA-only fallback. AIA outage analysis showed clustered empirical failure episodes, while AIA-only fallback performance deteriorated sharply in 2026. The explicit physical layer revealed a gradient-family-dominated shift in the locally stored SHARP feature distribution that was not detected by statistical confidence measures; matched same-record JSOC queries showed exact parity between the 16 SHARP summary keywords in the CCD and CEA series, although the local solar-versus-pipeline origin of the April 2026 shift remains unresolved. Physical applicability was therefore retained as an independent monitoring dimension rather than a direct abstention rule.

These results support a trust-oriented view of operational flare prediction: calibration, uncertainty, physical applicability, provenance, and graceful degradation should be treated as complementary system properties rather than inferred from predictive confidence alone.

---

## Core claim set for manuscript

1. **Operational trust contribution.** The main contribution is a reproducible issue/degrade/withhold framework under temporal shift and realistic AIA failures.
2. **Reliability paradox.** Statistical agreement tightened while rare-flare reliability and conformal coverage deteriorated.
3. **Fallback finding.** SHARP preserved availability under AIA failures; frozen AIA-only fallback was not stable enough for automatic use.
4. **Physical applicability finding.** An interpretable magnetic-state monitor identified a gradient-dominated distribution shift that was not reflected by statistical agreement, but physical distance was not a valid case-level abstention score.
5. **Baseline finding.** A simple latest-state SHARP logistic was competitive; temporal SHARP showed only a modest Cycle-25 TSS advantage, and equal-weight fusion was not universally superior.
6. **Claim boundary.** The work does not claim MHD/PINN physics, universal predictive superiority, true live prospective validation, or a causal solar-cycle explanation of the 2026 gradient shift.

---

## Recommended manuscript structure

### 1. Introduction
Motivate the gap between predictive accuracy and operational trustworthiness. Introduce calibration drift, uncertainty degradation, temporal shift, modality failure, and applicability monitoring as separate operational concerns. End with a concise contribution list centered on trust routing rather than fusion.

### 2. Related work
Cover solar-flare prediction from SHARP and imaging, multimodal fusion, calibration and uncertainty, conformal prediction, temporal robustness / cross-cycle evaluation, and operational forecasting. Position the paper against prior work without claiming broad first-of-kind status unless the literature audit supports it.

### 3. Data and frozen evaluation design
Describe SHARP, AIA, GOES-derived labels, 72-hour target, 96-minute issue cadence, time-ordered role structure, AR leakage controls, frozen hashes, and the distinction between earlier development, Cycle-25 retrospective evaluation, and supplementary 2026 post-hoc evidence.

### 4. Gray-Box trust framework
Describe the frozen predictors, probability calibration, conformal label sets, agreement signals, explicit magnetic-state applicability layer, data-quality/provenance checks, and NORMAL / DEGRADED / ABSTAIN state machine.

### 5. Experimental protocol
Describe threshold freezing, policy freezing, outage replay, AIA-only fallback audit, rolling prospective-style replay, simple baseline suite, active-region block bootstrap, and claim boundaries.

### 6. Results
Organize around:
- cross-cycle performance and calibration drift;
- conformal reliability degradation;
- agreement-versus-reliability paradox;
- outage and fallback experiments;
- operational routing replay;
- physical applicability diagnostics;
- baseline comparison and uncertainty intervals.

### 7. Discussion
Explain why confidence is not sufficient for trust, why physical applicability is monitoring rather than a gate, why fusion was not universally superior, and why graceful degradation is valuable even without a state-of-the-art predictor claim.

### 8. Limitations
State post-hoc elements, lack of true live prospective deployment, unresolved local-versus-JSOC origin of the April-2026 gradient shift, dependence among repeated forecast windows, and the absence of a first-principles physical model.

### 9. Conclusion
Reiterate the operational trust contribution and the evidence for separating predictive confidence, calibration, physical applicability, provenance, and availability.
