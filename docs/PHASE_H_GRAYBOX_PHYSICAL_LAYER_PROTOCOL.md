# Phase H — Explicit physical-state Gray-Box layer protocol

Status: **PRE-EXECUTION PROTOCOL FREEZE — 6 October 2026**

This document defines the post-A–G extension required to make the term **Gray-Box** scientifically explicit rather than relying only on physically meaningful SHARP inputs. It is written and committed **before Phase H results are inspected**.

## 1. Why Phase H exists

The completed 72 h programme established strong operational findings about temporal degradation, calibration, conformal reliability, multimodal disagreement, acquisition failure, fallback and abstention. However, the explicit physically interpretable layer in the original five-layer architecture remains less developed than the data-driven, uncertainty and operational-safety layers.

SHARP inputs are physical measurements and derived magnetic-field quantities, but using those variables as inputs to a GRU does not by itself create a constrained physical model. Phase H therefore adds a **transparent magnetic-state layer** that is computed directly from the frozen SHARP histories and is kept distinguishable from the black-box SHARP and AIA predictors.

The intended Gray-Box meaning is:

> learned forecast + interpretable magnetic state + uncertainty/applicability evidence + operational routing.

This project does **not** claim to implement a first-principles MHD model, a PINN, or a deterministic flare law.

## 2. Evidence boundary

Phase H is a **hypothesis-driven post-freeze extension** motivated by the completed A–G findings. The protocol is frozen before executing Phase H, but Cycle-25 and 2026 outcomes have already been examined in prior phases. Therefore:

- fitting and threshold estimation are restricted to designated earlier Cycle-24 development support;
- Cycle-25 and 2026 Phase-H analyses are retrospective/post-hoc, not untouched confirmation;
- no Phase-H result may rewrite the already-frozen A–G 72 h results;
- 2026 cannot be called a new clean test for Phase H;
- a later 24 h / 3 h replication and future prospective logging are needed for stronger confirmation.

## 3. Frozen source feature contract

The aligned dataset manifest defines the 16 SHARP features, in this exact order:

1. MEANGBZ
2. MEANGAM
3. MEANGBT
4. MEANGBH
5. MEANJZD
6. TOTUSJZ
7. MEANALP
8. MEANJZH
9. ABSNJZH
10. SAVNCPP
11. MEANSHR
12. SHRGT45
13. R_VALUE
14. USFLUX
15. TOTPOT
16. TOTUSJH

The history slots are issue minus 288, 192 and 96 native minutes. The Phase-H code must verify the manifest feature order and the raw SHARP tensor hash before any analysis.

## 4. Pre-declared physical families

The physical layer will not select features by later test performance. The following groups are declared in advance from the physical meaning of the SHARP quantities.

| Physical family | SHARP quantities | Interpretation |
|---|---|---|
| Magnetic flux / PIL complexity | USFLUX, R_VALUE | Amount of unsigned magnetic flux and strong-gradient polarity-inversion-line flux context |
| Free energy / shear | TOTPOT, MEANSHR, SHRGT45 | Non-potentiality, integrated free-energy proxy, and magnetic shear |
| Current / helicity / twist | TOTUSJZ, TOTUSJH, ABSNJZH, SAVNCPP, MEANJZD, MEANJZH, MEANALP | Electric-current, current-helicity and twist-related state |
| Field gradients | MEANGBZ, MEANGBH, MEANGBT | Spatial magnetic-field gradients |
| Field inclination / geometry | MEANGAM | Mean field inclination; retained separately because a monotonic flare-risk direction is not assumed |

For signed mean-current/helicity/twist quantities, the complexity magnitude is based on the absolute value; the original sign is retained separately for audit. No claim is made that a large value of every quantity is individually necessary or sufficient for flaring.

## 5. Training-only physical-state transform

For each SHARP feature and history slot:

1. Preserve the raw value.
2. Use an absolute-value transform only for the signed quantities declared above.
3. Apply `log1p` to magnitude-like quantities with strongly positive scale.
4. Estimate median and IQR using the frozen **training reference only**.
5. Convert each transformed value to a robust standardized value.
6. Clip only for numerical stability at a predeclared broad bound; clipping must not be chosen from later outcomes.

For each physical family, the family state is the median of its robust standardized members. This gives equal-family interpretability rather than allowing one high-dimensional family to dominate.

For each family, Phase H records:

- latest state at issue-96 min;
- net change from issue-288 to issue-96 min;
- three-point linear slope across the 192-minute history.

The resulting vector is the **Magnetic Physical State Vector (MPSV)**.

## 6. Interpretable physical probability

A simple logistic model is fitted only on the MPSV using the designated early training block. It is deliberately low-capacity and interpretable. Its purpose is **not** to replace the frozen SHARP GRU. It produces an independent physical-state probability, `p_phys`, against which the learned branches can be checked.

Coefficient tables must be exported so the contribution of each physical axis and its temporal evolution is visible.

No neural network, tree ensemble or hidden nonlinear feature search is permitted in the primary Phase-H physical model.

## 7. Physical applicability / OOD

A new physical-state applicability measure is defined in the low-dimensional MPSV space.

- Fit a robust covariance estimator (Ledoit-Wolf) on the training-reference MPSV only.
- Compute squared Mahalanobis distance for every case.
- Freeze a primary applicability threshold at the training-reference 99th percentile.
- Report the continuous distance and the binary out-of-reference flag.

This is a **new Phase-H physical OOD definition**. It must never be described as a reconstruction of the provenance-limited historical feature-space OOD artifact from Phase C2.

## 8. Physical consistency with learned forecasts

For cases with a fusion probability, define physical consistency using the absolute log-odds difference between `p_phys` and the frozen fusion probability. The probability inputs are clipped only to a fixed numerical epsilon before taking logits.

A large gap means the data-driven fusion and the interpretable magnetic-state model disagree. The raw gap remains a diagnostic. If a threshold is later considered for routing, that threshold must be frozen on earlier policy-development support only.

## 9. Primary Phase-H questions

H1. Does the MPSV provide a physically interpretable description of the magnetic regimes seen across Cycle 24, Cycle 25 and 2026?

H2. Does physical-state distance increase under later temporal shift, unlike entropy/seed-disagreement signals that became more reassuring?

H3. Does physical-model/fusion inconsistency identify elevated forecast-error or conformal-failure risk within regimes?

H4. Does the interpretable physical probability provide a meaningful same-support baseline without claiming superiority over the GRU?

H5. If physical applicability or consistency is added as an operational trust signal, can it improve error concentration without catastrophically deferring flare windows?

## 10. Success and failure criteria

The physical layer is **not guaranteed** to improve routing.

A signal is considered potentially useful only if:
- it has a stable, interpretable relationship with error or flare reliability on earlier policy-development support;
- any proposed threshold is frozen before later-regime evaluation;
- the retained/issued subset does not achieve lower error merely by discarding a disproportionate fraction of flare windows;
- the result survives AR-level sensitivity analysis where possible.

A negative result is valid. If physical-state distance does not track later shift, or a physical-consistency gate removes too many positive windows, the signal remains descriptive and is **not** promoted into the operational state machine.

## 11. Phase-H execution sequence

### H1 — Physical-state construction and audit
Notebook 25:
- verify source hashes and feature order;
- construct MPSV from frozen SHARP histories;
- fit training-only robust transforms and covariance;
- fit the simple interpretable physical logistic model;
- export coefficients, physical state, distance and consistency diagnostics;
- do not alter the frozen v3 policy.

### H2 — Physical trust-gate development
Only after H1:
- use earlier development/policy support only;
- compare candidate physical applicability and consistency gates;
- quantify risk-coverage and positive retention;
- either freeze one transparent physical gate or formally reject promotion.

### H3 — Retrospective later-regime evaluation
After H2 freeze:
- apply the frozen Phase-H rule to Cycle-25 and 2026;
- label the evidence post-hoc because those outcomes were already seen in A–G;
- compare v3 versus physical-aware routing without changing the original A–G claim ledger.

### H4 — Horizon replication
The same *architecture* may then be tested at 24 h and 3 h, but every transform, calibration, conformal threshold and routing cut-off must be horizon-specific.

## 12. What makes the final system Gray-Box

If Phase H is successfully completed, the framework will contain four distinct kinds of information:

1. **Black-box learned forecasts:** temporal SHARP GRU and AIA image model.
2. **Explicit physical state:** transparent magnetic flux/PIL, energy/shear, current/helicity, gradient and geometry axes plus their temporal evolution.
3. **Statistical reliability:** calibration, class-conditional conformal uncertainty, seed spread, entropy and cross-modal disagreement.
4. **Operational safety:** explicit NORMAL / DEGRADED / ABSTAIN routing with data-availability and applicability reason codes.

The physical layer does not dictate a flare through first-principles equations. Instead, it provides an interpretable magnetic-state reference against which learned forecasts and their domain of applicability are assessed. That is the precise Gray-Box claim to defend.

## 13. Literature anchor for the physical layer

The physical feature families are grounded in the SHARP data product and established flare-prediction literature. Bobra et al. (2014) define the SHARP summary parameters as active-region magnetic quantities including flux, field gradients, vertical current density, current helicity and an integrated free-energy proxy. Bobra & Couvidat (2015) and related vector-magnetogram forecasting work establish that a subset of these physically meaningful quantities is useful for M/X flare discrimination. Earlier work by Leka & Barnes shows both the forecasting value and the important limitation that no single small set of magnetic properties is necessary and sufficient for flaring.

Primary anchors:
- Bobra, M. G. et al. (2014), Solar Physics 289, 3549–3578. DOI: 10.1007/s11207-014-0529-3.
- Bobra, M. G. & Couvidat, S. (2015), ApJ 798, 135. DOI: 10.1088/0004-637X/798/2/135.
- Leka, K. D. & Barnes, G. (2007), ApJ 656, 1173–1186. DOI: 10.1086/510282.

## 14. Claim discipline

Before H2/H3 execution, the paper may say only that the framework **defines** an explicit physical-state layer.

After execution, stronger language is allowed only if the results support it. In particular:
- do not call the framework MHD-constrained;
- do not call it a PINN;
- do not call the physical score a causal flare mechanism;
- do not claim untouched confirmation from 2026;
- do not state that Gray-Box routing improves skill unless the frozen comparison demonstrates it.
