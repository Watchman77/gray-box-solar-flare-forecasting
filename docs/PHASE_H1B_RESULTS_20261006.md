# Phase H1b results — nonredundant physical-state audit

Status: **EXECUTED — 6 October 2026**

H1b removed the deterministic redundancy between endpoint net change and three-point linear slope in H1. The corrected Magnetic Physical State Vector (MPSV) uses latest state, endpoint net change, and three-point curvature for each physical family.

## 1. Frozen provenance

H1b used the same exact hash-pinned inputs as H1:

- SHARP tensor: `c9921be7b7fe381b43436a0e927d15d84de83e15376025ae0d8a79a6ef917639`
- frozen 72 h SHARP ledger: `7368c002a978321cdc67221564f3b7901687badc7aba28ab9df615179d1e017b`
- frozen multimodal master: `17300093b88b7c61ecadf5eb28acba0d6d023b340d7740807852dd4352b22b46`

Training reference: 25,586 frozen training cases, 1,476 positives.

## 2. Physical applicability shift survives the structural correction

| Evidence regime | Cases | Positives | Physical OOD rate (training q99) | Median physical distance |
|---|---:|---:|---:|---:|
| Earlier Cycle-24 development | 50,007 | 2,929 | 0.01302 | 10.859 |
| Cycle 25, 2021–2025 post-hoc | 35,846 | 3,860 | 0.01437 | 12.542 |
| Supplementary 2026 post-hoc | 11,118 | 992 | 0.02824 | 16.741 |

The 2026 OOD rate is approximately 2.17 times the earlier Cycle-24 descriptive rate and 1.97 times the Cycle-25 rate. Median physical distance is approximately 54.2% higher than the earlier Cycle-24 median and 33.5% higher than the Cycle-25 median.

The magnitude is smaller than the first H1 representation, but the direction remains. Therefore the evidence for a later physical-state distribution shift is **robust to removal of the deterministic temporal-feature redundancy**.

This remains retrospective/post-hoc for Cycle 25 and 2026.

## 3. Interpretable physical reference also degrades

| Evidence regime | AP | ROC-AUC | Brier |
|---|---:|---:|---:|
| Earlier Cycle-24 development | 0.5259 | 0.9061 | 0.03845 |
| Cycle 25 post-hoc | 0.4371 | 0.8497 | 0.07596 |
| Supplementary 2026 post-hoc | 0.3720 | 0.7497 | 0.07066 |

The broad degradation pattern is essentially unchanged from H1. This supports the interpretation that the later regime differs materially from the development regime even for a low-capacity interpretable magnetic-state model.

Cross-regime Brier values should not be interpreted without prevalence/reference context.

## 4. Physical-family shifts

The largest median state changes in 2026 relative to the earlier regime occur in:

- `gradients__latest`: 0.0360 → -0.7638
- `inclination__latest`: -0.0203 → -0.1999
- `flux_pil__latest`: -0.0675 → 0.1571

Other families shift more modestly.

These values are robust-z family summaries relative to the training transform. They establish a distribution change in the measured/derived magnetic-state representation. They do **not** by themselves establish a solar-physics cause. Source-product, processing, instrument, selection and solar-cycle effects remain possible contributors.

## 5. Physical distance is not a case-level error score

Spearman association with fusion squared error remains negative and becomes slightly stronger after the nonredundant correction:

| Evidence regime | Physical distance | Physical/fusion logit gap |
|---|---:|---:|
| Earlier Cycle-24 development | -0.5117 | -0.3347 |
| Cycle 25 post-hoc | -0.4499 | -0.3578 |
| Supplementary 2026 post-hoc | -0.4765 | -0.5479 |

Therefore H1b reinforces the H1 negative result: larger physical distance should **not** be promoted directly into an abstention or degradation gate.

A likely explanation is class/decision-boundary structure: magnetically extreme regions may be easier to classify, while difficult errors occur in less extreme or physically ambiguous states. This must be tested explicitly rather than assumed.

## 6. Scientific interpretation after H1b

H1b supports a distinction between:

- **physical-domain applicability / regime shift**, and
- **case-level predictive uncertainty / error risk**.

The physical Mahalanobis distance appears promising as a regime-monitoring signal because it moves in the expected direction into 2026 while earlier statistical agreement signals became smaller. However, it is not a monotonic case-level error indicator.

This makes the Gray-Box framework more precise: physical applicability should not automatically be conflated with confidence or used as a simple rejection score.

## 7. Required next diagnostic before H2

Before any physical signal is added to routing, Phase H1c must:

1. decompose physical distance/OOD by year, class and active region;
2. separate flare and non-flare distributions;
3. determine whether the 2026 jump is gradual, abrupt, or concentrated near the source/pipeline boundary;
4. quantify which physical families drive the shift;
5. compare repeated-window results with active-region-level summaries;
6. test whether physical distance identifies regime shift after controlling for class prevalence;
7. retain the explicit possibility that the observed physical-state shift reflects data-product or processing differences rather than only solar-cycle physics.

No H2 gate should be frozen until H1c is reviewed.
