# Phase H1 results — explicit physical-state Gray-Box audit

Status: **EXECUTED — 6 October 2026**

This note records the first executed Phase-H physical-state audit. It does not modify the frozen A–G 72 h policy.

## 1. Frozen provenance

The H1 run verified all three principal inputs before construction:

- canonical SHARP tensor SHA-256: `c9921be7b7fe381b43436a0e927d15d84de83e15376025ae0d8a79a6ef917639`
- frozen 72 h SHARP ledger SHA-256: `7368c002a978321cdc67221564f3b7901687badc7aba28ab9df615179d1e017b`
- frozen multimodal master SHA-256: `17300093b88b7c61ecadf5eb28acba0d6d023b340d7740807852dd4352b22b46`

The physical reference was fit on the frozen `train` role only: 25,586 cases, including 1,476 positives.

## 2. First H1 result: physical-state distribution shift is visible

Using the predeclared Magnetic Physical State Vector (MPSV) and training-only Ledoit-Wolf squared Mahalanobis distance:

| Evidence regime | Cases | Positives | Physical OOD rate (training q99) | Median physical distance |
|---|---:|---:|---:|---:|
| Earlier Cycle-24 development | 50,007 | 2,929 | 0.01382 | 7.379 |
| Cycle 25, 2021–2025 post-hoc | 35,846 | 3,860 | 0.01515 | 8.359 |
| Supplementary 2026 post-hoc | 11,118 | 992 | 0.07627 | 11.733 |

The 2026 physical-OOD rate is about 5.5 times the earlier Cycle-24 descriptive rate and about 5.0 times the Cycle-25 rate. Median physical-state distance is about 59% higher than the earlier Cycle-24 descriptive median.

This is directionally important because earlier A–G statistical trust indicators such as fusion seed spread, entropy and SHARP–AIA probability gap became smaller into 2026 even while rare-flare reliability deteriorated. H1 therefore provides preliminary evidence that **physical-state applicability contains regime-shift information not captured by model agreement alone**.

This finding remains post-hoc for Cycle 25 and 2026 because those outcomes were already inspected in A–G.

## 3. Interpretable physical reference degrades across regimes

The low-capacity logistic model on the MPSV shows:

| Evidence regime | AP | ROC-AUC | Brier |
|---|---:|---:|---:|
| Earlier Cycle-24 development | 0.528 | 0.906 | 0.0384 |
| Cycle 25 post-hoc | 0.438 | 0.849 | 0.0759 |
| Supplementary 2026 post-hoc | 0.373 | 0.748 | 0.0706 |

The earlier Cycle-24 row includes development support and is descriptive rather than an independent test result. The decline in AP/AUC is nevertheless consistent with the broader A–G finding that the later regime differs materially from the development regime.

The 2026 Brier value should not be interpreted as a monotonic improvement over Cycle 25 because prevalence differs; Brier skill against a fixed reference is more suitable for cross-regime interpretation.

## 4. Coefficient interpretation

The largest positive latest-state coefficient is the free-energy/shear family, followed by gradients, current/helicity and flux/PIL complexity. The latest inclination coefficient is negative while its temporal-change coefficient is positive.

These coefficients are **associational, not causal**. They are estimated from correlated magnetic-state summaries and must not be read as physical laws.

## 5. Important negative result: physical distance is not a naive case-level error gate

Spearman association with fusion squared error is negative in every regime:

| Regime | Physical distance | Physical/fusion logit gap |
|---|---:|---:|
| Earlier Cycle-24 development | -0.412 | -0.333 |
| Cycle 25 post-hoc | -0.368 | -0.356 |
| 2026 post-hoc | -0.443 | -0.544 |

Therefore H1 **does not support** the rule “larger physical distance means larger forecast error” at the individual-case level. A plausible interpretation is that highly magnetically active/extreme regions may be easier for the existing predictors to classify, whereas errors concentrate in physically ambiguous regions nearer the decision boundary. That interpretation is a hypothesis and requires class-conditional diagnostics.

The physical OOD score may still be useful as a **regime-level applicability monitor** even if it is not a monotonic case-level abstention score.

## 6. Structural issue discovered after H1 execution

The original MPSV included, for each physical family:

- latest state;
- net change from issue-288 to issue-96 minutes;
- linear slope across the same three equally spaced samples.

For equally spaced three-point histories, the fitted linear slope is exactly one-half of the endpoint net change. Therefore `net_change` and `slope_per96m` are perfectly collinear.

This is visible directly in the fitted coefficients: every slope coefficient is exactly one-half of its corresponding net-change coefficient.

The Ledoit-Wolf covariance prevents numerical singularity, but the redundant representation weakens interpretability and can distort distance geometry. H1 is preserved as executed evidence, but **its physical OOD values should be treated as provisional until a nonredundant sensitivity rerun is completed**.

This structural correction is mathematical and outcome-independent; it is not motivated by later performance.

## 7. Predeclared H1b correction

H1b will retain:
- latest state;
- endpoint net change;
- three-point curvature `x0 - 2*x1 + x2`.

Curvature is independent of endpoint change and represents acceleration/nonlinearity in the short magnetic evolution.

H1b will use the same exact hash-pinned inputs, same training-only transform, same physical families, same low-capacity logistic model, and same training-only Ledoit-Wolf applicability procedure. No policy threshold will be selected in H1b.

## 8. Claim boundary after H1

Supported now:
- later physical-state shift is visible in the first explicit Gray-Box audit;
- 2026 has substantially more physical-state OOD relative to the training reference;
- physical applicability and model agreement are not interchangeable concepts;
- larger physical distance is not a valid naive per-case error gate.

Not yet supported:
- physical OOD improves NORMAL/DEGRADED/ABSTAIN routing;
- physical/ML disagreement should trigger abstention;
- the H1 OOD magnitude is final before removal of the deterministic feature redundancy;
- a causal solar-cycle mechanism for the shift.
