# 72-hour manuscript rewrite map

## Why a rewrite is necessary

The earliest Gray-Box draft was primarily a multimodal prediction paper. The frozen 72-hour evidence now supports a broader and more coherent trustworthiness paper. The manuscript should therefore be rewritten around operational reliability rather than patched sentence-by-sentence.

## Old framing -> final framing

| Earlier emphasis | Final emphasis |
|---|---|
| Can SHARP+AIA fusion improve prediction? | When should the system issue, degrade, or withhold? |
| Predictive performance as the main endpoint | Joint discrimination, calibration, UQ validity, availability, and routing |
| Missingness as a synthetic robustness test | Real clustered AIA acquisition failures plus synthetic sensitivity |
| Conformal prediction as a static UQ layer | Empirical conformal reliability that itself degrades under temporal shift |
| Agreement/spread as confidence signals | Within-regime risk indicators that can fail to detect global shift |
| Fusion as the intended operational model | Fusion only in NORMAL; SHARP fallback in DEGRADED; ABSTAIN if no validated branch remains |
| Multi-horizon claim in principle | 72 h frozen now; 24 h and 3 h must be replicated independently |

## Section-by-section rewrite

### Abstract
Rewrite completely. Lead with operational trust, temporal shift, UQ degradation, realistic outages, and the three-state routing framework. Do not lead with fusion superiority.

### Introduction
Add the distinction between prediction and operational trust. Motivate non-stationarity, calibration drift, UQ failure, data outages, and selective issuance.

### Related work
Organise around:
1. solar-flare ML forecasting;
2. cross-cycle / temporal generalisation;
3. calibration and uncertainty;
4. conformal prediction under changing regimes;
5. selective prediction / abstention / fallback;
6. multimodal missingness and observational outages.

A targeted novelty review is still required before any “first” claim.

### Data and experimental design
Retain the full provenance story: 153,366 candidates; 113,433 matched cases; chronological roles; label/event reconciliation; unresolved outcomes; AR component checks; no random-split claims.

### Methods
Present the five-layer architecture and clearly distinguish:
- SHARP branch;
- AIA branch;
- fixed 0.5/0.5 fusion;
- probability calibration;
- Mondrian conformal;
- seed spread / entropy / modality gap;
- NORMAL / DEGRADED / ABSTAIN routing.

Preserve v1 -> v2 -> v3 development history.

### Results
Rebuild around A-G:
- A: cross-cycle degradation;
- B: calibration drift;
- C: applicability/UQ shift;
- D: AR-disjoint;
- E: real outages and clustering;
- F: AIA-only fallback rejection;
- G: rolling prospective-style replay.

### Discussion
The central discussion should be that confidence-like signals can become more reassuring while rare-event reliability worsens. Emphasise that trust needs regime-level monitoring, not only case-level UQ.

### Limitations
State explicitly:
- not a true prospective deployment;
- exact feature-space OOD provenance missing;
- exact event-disjoint/lead-time provenance missing;
- no universal superiority claim;
- no MHD/PINN claim;
- no conformal guarantee under temporal shift.

### Conclusion
Conclude with issue/degrade/withhold routing and graceful degradation, not “fusion performed best.”

## Figures to prioritise

1. Five-layer architecture and state machine.
2. Cross-regime SHARP/AIA/fusion skill.
3. Flare conformal coverage vs singleton rate.
4. Rolling TSS and UQ monitor.
5. Real outage episode structure and NORMAL -> DEGRADED routing.

## Tables to prioritise

1. Chronological roles and support.
2. Cross-regime performance/calibration.
3. UQ/conformal transfer.
4. Outage replay and fallback feasibility.
5. Claim ledger and limitations.
