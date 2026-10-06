# Manuscript-ready tables — 72 h Gray-Box paper

Status: **working journal tables — 6 October 2026**

## Table 1. Chronological roles and matched 72 h support

| Role | Date interval | Matched cases | Positives | AR components | Evidence function |
|---|---|---:|---:|---:|---|
| train | 2010–2013 | 25,586 | 1,476 | 629 | model fitting and training-only transforms |
| model_validation | Jan–Jun 2014 | 3,905 | 416 | 99 | seed/model and alarm-threshold selection |
| probability_calibration | Jul–Dec 2014 | 2,831 | 184 | 69 | probability-calibration development |
| conformal_calibration | Jan–Jun 2015 | 4,168 | 384 | 95 | Mondrian conformal and trust-cutoff reference |
| policy_validation | Jul 2015–2019 | 13,142 | 422 | 313 | earlier policy evaluation |
| retrospective_cycle25 | 2021–2025 | 35,846 | 3,860 | 943 | diagnostic/post-hoc evidence for final v3 semantics |
| supplementary_2026 | 2026 | 11,118 | 992 | 128 | clean future evaluation for frozen v3; later Phase-H analyses post-hoc |

Total frozen matched 72 h schedule: 96,596 cases. The 2020 gap is retained because no accepted matched input package was available.

## Table 2. Gray-Box trust components and operational use

| Component | Frozen/developed on | Interpretation | NORMAL | DEGRADED | Can block issuance? |
|---|---|---|---|---|---|
| SHARP GRU probability | train + model_validation | temporal magnetic forecast | fusion member | issued fallback | only if technically unavailable |
| AIA temporal probability | train + model_validation | image forecast | fusion member | not used as automatic fallback | no, if SHARP remains available |
| fixed 0.5/0.5 fusion | prespecified | multimodal probability | forecast source | replaced by SHARP | yes for NORMAL only |
| probability calibration | probability_calibration | probabilistic reliability | monitored | monitored | no direct v3 block |
| fusion Mondrian set | conformal_calibration | class-conditional ambiguity/coverage | must be singleton | advisory only | blocks NORMAL, not SHARP fallback |
| fusion seed spread | earlier calibration support | ensemble disagreement | q90 gate | advisory | blocks NORMAL only |
| SHARP-AIA gap | earlier calibration support | cross-modal disagreement | q90 gate | advisory | blocks NORMAL only |
| fusion entropy | earlier calibration support | probability decisiveness | q90 gate | advisory | blocks NORMAL only |
| physical MPSV / MD² | train reference; Phase H post-freeze | physical applicability/provenance monitor | advisory | advisory | no |
| AIA availability | runtime/engineering status | image-branch availability | required | not required | blocks NORMAL only |
| SHARP availability | runtime status | fallback availability | required | required | yes: unavailable ⇒ ABSTAIN |

## Table 3. Cross-regime branch reliability

| Regime | Branch | TSS | AP | Brier | ECE | Flare conformal coverage |
|---|---|---:|---:|---:|---:|---:|
| Earlier Cycle-24 development | SHARP | 0.689 | 0.538 | 0.0365 | 0.0056 | 0.806 |
| Earlier Cycle-24 development | AIA | 0.333 | 0.122* | 0.0551 | 0.0217 | 0.812 |
| Earlier Cycle-24 development | Fusion | 0.630 | 0.524 | 0.0414 | 0.0303 | 0.838 |
| Cycle 25 | SHARP | 0.541 | 0.440 | 0.0774 | 0.0307 | 0.735 |
| Cycle 25 | AIA | 0.340 | 0.184 | 0.0973 | 0.0674 | 0.620 |
| Cycle 25 | Fusion | 0.488 | 0.391 | 0.0812 | 0.0373 | 0.692 |
| Supplementary 2026 | SHARP | 0.323 | 0.396 | 0.0691 | 0.0474 | 0.435 |
| Supplementary 2026 | AIA | 0.074 | 0.159 | 0.0869 | 0.0793 | 0.064 |
| Supplementary 2026 | Fusion | 0.292 | 0.392 | 0.0743 | 0.0597 | 0.346 |

*Earlier AIA AP shown from the fallback-feasibility same-support table because the cross-cycle branch-metrics artifact does not include AP.

## Table 4. Operational replay and AIA fallback evidence

| Scenario / branch | Regime | Availability | NORMAL | DEGRADED | Recall | TSS | AP |
|---|---|---:|---:|---:|---:|---:|---:|
| v3 nominal | Cycle 25 | 1.000 | 0.777 | 0.223 | 0.655 | 0.507 | 0.437 |
| v3 mapped AIA incidents | Cycle 25 | 1.000 | 0.721 | 0.279 | 0.665 | 0.512 | 0.438 |
| v3 nominal | 2026 | 1.000 | 0.925 | 0.075 | 0.346 | 0.296 | 0.393 |
| AIA-only branch | Cycle 25 | 1.000 | — | — | 0.743 | 0.340 | 0.184 |
| AIA-only branch | 2026 | 1.000 | — | — | 0.117 | 0.074 | 0.159 |

Interpretation: mapped AIA failures mainly change routing state while preserving issuance through SHARP. AIA-only 2026 performance does not support automatic fallback.

## Table 5. Key paired active-region bootstrap comparisons

| Role | Comparison (A − B) | Metric | Median delta | 95% CI | Interpretation |
|---|---|---|---:|---:|---|
| policy_validation | SHARP − logistic | TSS | +0.027 | [-0.054, 0.161] | inconclusive |
| Cycle 25 | SHARP − logistic | TSS | +0.039 | [0.0003, 0.0819] | modest SHARP advantage |
| 2026 | SHARP − logistic | TSS | +0.052 | [-0.004, 0.123] | inconclusive |
| policy_validation | Fusion − SHARP | AP | -0.083 | [-0.178, -0.003] | SHARP higher |
| policy_validation | Fusion − SHARP | Brier | +0.00445 | [0.00152, 0.00804] | fusion worse |
| Cycle 25 | Fusion − SHARP | TSS | -0.053 | [-0.091, -0.013] | SHARP higher |
| Cycle 25 | Fusion − SHARP | AP | -0.047 | [-0.084, -0.013] | SHARP higher |
| 2026 | Fusion − SHARP | TSS | -0.030 | [-0.074, 0.008] | inconclusive |

All intervals are percentile 95% intervals from paired region_component_id block-bootstrap replicates.