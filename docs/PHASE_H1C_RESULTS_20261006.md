# Phase H1c results — physical-shift decomposition

Status: **EXECUTED — 6 October 2026**

H1c decomposed the nonredundant H1b physical-state shift by year, class, active region, physical axis, recent month and statistical-agreement signals. It made no policy change.

## 1. Key result: physical applicability and statistical agreement diverge

On the 71,010-case frozen multimodal support:

| Evidence regime | Median physical MD² | Median fusion seed std | Median fusion entropy | Median SHARP–AIA gap |
|---|---:|---:|---:|---:|
| Earlier Cycle-24 development | 11.5955 | 0.012219 | 0.138411 | 0.027830 |
| Cycle 25, 2021–2025 post-hoc | 12.5417 | 0.006433 | 0.118344 | 0.019232 |
| Supplementary 2026 post-hoc | 16.7415 | 0.001626 | 0.048715 | 0.004042 |

Relative to the earlier Cycle-24 multimodal support, the 2026 median physical-state distance is about 44% higher, while fusion seed spread is about 87% lower, fusion entropy about 65% lower, and the SHARP–AIA probability gap about 85% lower.

Within each regime, physical distance is also negatively associated with these statistical agreement measures. In 2026 the Spearman correlations are approximately -0.403 with fusion seed spread, -0.541 with entropy and -0.544 with the SHARP–AIA probability gap.

This is strong descriptive evidence that **physical-domain applicability and model agreement are distinct axes of trust**. A model ensemble can become more internally consistent and more decisive while the measured magnetic state moves farther from its development reference.

## 2. Active-region aggregation preserves the distance trend, but the any-OOD statistic is exposure-sensitive

Active-region aggregation preserves the direction of the median physical-distance result:

| Regime | Active regions | Positive ARs | Median AR median MD² | Median AR OOD fraction | Median windows per AR |
|---|---:|---:|---:|---:|---:|
| Earlier Cycle-24 development | 1,223 | 106 | 11.331 | 0.0000 | 45.0 |
| Cycle 25 post-hoc | 948 | 138 | 13.092 | 0.0000 | 39.0 |
| Supplementary 2026 post-hoc | 128 | 22 | 16.553 | 0.01227 | 72.5 |

The previously reported fraction of active regions with *any* OOD window is deliberately demoted because 2026 regions contain more forecast windows on average, increasing the opportunity to observe at least one tail event. The more defensible AR-level evidence is the rise in median AR-level physical distance and the non-zero median OOD fraction in 2026.

Therefore the 2026 physical-distance shift is not solely a repeated-window artefact, but the strength of the AR-level claim should rest on distance and OOD fraction rather than an any-window indicator.

## 3. The shift survives class conditioning

Physical distance is systematically lower for positive flare windows than for non-flare windows. This explains why physical distance was negatively associated with case-level prediction error and why it should not be used as a naive rejection score. The careful wording is that **non-flare windows occupy more distant physical-state distributions on average than flare windows**; the analysis does not equate every label-0 window with a physically quiet active region.

Regime medians:

- Earlier Cycle 24: non-flare 11.187, flare 7.075.
- Cycle 25: non-flare 13.072, flare 9.293.
- 2026: non-flare 17.235, flare 11.057.

Nevertheless, both classes move farther from the earlier reference in 2026. The 2026 q99 OOD rate is 2.85% for non-flare windows and 2.52% for flare windows, compared with 1.37% and 0.14% respectively in the earlier Cycle-24 descriptive support.

This supports a regime/applicability interpretation rather than a prevalence-only explanation.

## 4. The year-by-year pattern is not monotonic

The 2026 year has median MD² 16.741 and q99 OOD rate 2.82%, but earlier late-Cycle-24 years also show elevated distance:

- 2018: median 19.450, OOD 3.79%, with zero positive windows in the known-label support.
- 2019: median 15.803, OOD 2.75%.
- 2024: median 12.572, OOD 1.89%.
- 2025: median 13.614, OOD 1.43%.

Therefore the evidence does **not** support a simple monotonic statement that physical distance increases continuously from Cycle 24 through Cycle 25. Physical-state distance appears sensitive to solar-regime composition and/or source/processing conditions.

The defensible statement is that the 2026 population is materially shifted from the training-reference distribution, not that solar-cycle number alone causes the shift.

## 5. The 2026 increase is not a clean January boundary jump

Monthly 2024–2026 diagnostics show:

- January 2026 median MD² 14.810, OOD 0.66%.
- February 13.371, 0.22%.
- March 12.448, 1.87%.
- April 17.586, 3.50%.
- May 16.583, 2.96%.
- June 17.209, 3.16%.
- July 19.145, 1.89%.
- August 17.727, 3.84%.

Thus the elevated 2026 aggregate is driven mainly by April–August rather than an immediate January transition. This weakens a simple “new calendar year/source cohort caused the entire shift” explanation, but it does not rule out an internal source or processing boundary.

## 6. The 2026 shift is gradient-dominated and should not yet be called broad magnetic-state shift

The largest standardized median shift is overwhelmingly:

- `gradients__latest`: -0.809 training-IQR units.

The next largest are:

- `flux_pil__latest`: +0.233;
- `inclination__latest`: -0.195;
- `current_helicity__latest`: +0.074.

All temporal net-change and curvature terms are much smaller in median standardized shift.

This indicates that the 2026 applicability change is driven primarily by the **latest-state magnetic-gradient family**, not by a broad large shift across every physical axis.

Accordingly, the phrase **broad magnetic-state shift** is not supported at H1c. The defensible description is a **physical-applicability shift dominated by the latest magnetic-gradient family** until H1d establishes whether the separation survives feature-family removal and source-lineage checks.

That finding requires a dedicated source/physics sensitivity audit before attributing the effect to solar-cycle physics.

## 7. Scientific conclusion after H1c

H1c supports the Gray-Box proposition that physical applicability supplies information that is not captured by statistical confidence/agreement. The physical-distance contrast survives nonredundant representation, class conditioning and active-region aggregation.

However, the year-by-year non-monotonicity, the exposure sensitivity of any-OOD-per-AR statistics, the late-2026 rise, and the dominance of the latest magnetic-gradient axis mean the mechanism is not yet identified. The next step is **not H2 routing**. A source/feature sensitivity audit is required first.

The frozen v3 NORMAL / DEGRADED / ABSTAIN policy remains unchanged.

Allowed current claim:

> In the frozen 72 h system, later 2026 cases occupy a more distant magnetic-state distribution relative to the early training reference even as ensemble spread, predictive entropy and cross-modal disagreement decrease. This contrast persists after class conditioning and active-region aggregation, indicating that physical applicability and statistical agreement measure different aspects of forecast trust.

Not allowed:
- “Solar Cycle 25 caused the physical shift.”
- “Physical OOD should trigger abstention.”
- “The gradient-family shift is necessarily solar-physical rather than data-product related.”
