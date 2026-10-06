# 72 h baseline suite results — exact frozen support

Status: **EXECUTED — 6 October 2026**

This suite compares training climatology, one latest-slot 16-feature SHARP logistic baseline, and the already-frozen SHARP/AIA/equal-weight-fusion branches. No neural model was retrained and no operational policy was changed.

## 1. Provenance and protocol

- frozen training rows: 25,586
- training positives: 1,476
- training prevalence: 0.0576878
- logistic fit: one L2 logistic on the 16 SHARP features at the latest available history slot (t-96 min), training role only
- logistic alarm threshold selected on model_validation only
- frozen SHARP/AIA/fusion thresholds reused unchanged
- primary evaluation roles: policy_validation, retrospective_cycle25, supplementary_2026
- active-region cluster bootstrap: 2,000 paired draws over region_component_id

Frozen thresholds:
- logistic latest SHARP: 0.1026533
- frozen SHARP: 0.0752000
- frozen AIA: 0.0251018
- frozen fusion: 0.0905993

## 2. Point metrics

### policy_validation

| model | TSS | AP | Brier | BSS vs training climatology |
|---|---:|---:|---:|---:|
| climatology | 0.000 | 0.032 | 0.03173 | 0.000 |
| latest-slot logistic | 0.597 | 0.497 | 0.02184 | 0.312 |
| frozen SHARP | 0.628 | 0.528 | 0.02072 | 0.347 |
| frozen AIA | 0.319 | 0.056 | 0.03629 | -0.143 |
| frozen fusion | 0.626 | 0.432 | 0.02526 | 0.204 |

### retrospective Cycle 25

| model | TSS | AP | Brier | BSS |
|---|---:|---:|---:|---:|
| climatology | 0.000 | 0.108 | 0.09859 | 0.000 |
| latest-slot logistic | 0.501 | 0.445 | 0.07827 | 0.206 |
| frozen SHARP | 0.541 | 0.440 | 0.07736 | 0.215 |
| frozen AIA | 0.340 | 0.184 | 0.09732 | 0.013 |
| frozen fusion | 0.488 | 0.391 | 0.08119 | 0.176 |

### supplementary 2026

| model | TSS | AP | Brier | BSS |
|---|---:|---:|---:|---:|
| climatology | 0.000 | 0.089 | 0.08226 | 0.000 |
| latest-slot logistic | 0.269 | 0.348 | 0.07095 | 0.137 |
| frozen SHARP | 0.323 | 0.396 | 0.06910 | 0.160 |
| frozen AIA | 0.074 | 0.159 | 0.08687 | -0.056 |
| frozen fusion | 0.292 | 0.392 | 0.07430 | 0.097 |

Absolute AP/Brier values should not be compared casually across roles because prevalence changes materially. The main comparisons are within-role and paired-bootstrap deltas.

## 3. Main paired-bootstrap findings

### Frozen SHARP versus latest-slot logistic

- policy_validation TSS difference: +0.0275, 95% AR-block CI [-0.0536, 0.1615] — inconclusive.
- Cycle 25 TSS difference: +0.0391, CI [0.0003, 0.0819] — small advantage for frozen SHARP.
- 2026 TSS difference: +0.0519, CI [-0.0040, 0.1229] — inconclusive.

AP and Brier differences between frozen SHARP and the simple logistic do not exclude zero in the later independent roles.

Interpretation: the temporal SHARP GRU is stronger in Cycle-25 TSS, but the simple latest-slot logistic is a strong comparator and the magnitude of the advantage is modest.

### Frozen fusion versus frozen SHARP

- policy_validation TSS difference: -0.0013, CI [-0.1083, 0.1480] — no TSS difference resolved.
- policy_validation AP difference: -0.0829, CI [-0.1784, -0.0035] — SHARP higher.
- policy_validation Brier difference: +0.00445, CI [0.00152, 0.00804] — fusion worse.

- Cycle 25 TSS difference: -0.0530, CI [-0.0907, -0.0133] — SHARP higher.
- Cycle 25 AP difference: -0.0467, CI [-0.0835, -0.0126] — SHARP higher.
- Cycle 25 Brier difference: +0.00373, CI crosses zero — inconclusive.

- 2026 TSS difference: -0.0304, CI [-0.0744, 0.0076] — inconclusive.
- 2026 AP difference: -0.00357, CI [-0.0136, 0.00727] — inconclusive.
- 2026 Brier difference: +0.00498, CI crosses zero — inconclusive.

Interpretation: equal-weight fusion is not supported as universally superior. On Cycle 25, frozen SHARP is clearly better in TSS and AP. In 2026 the point estimates still favor SHARP, but AR-block uncertainty includes zero.

### AIA branch

AIA has useful discrimination above climatology in AP/TSS, but its probabilistic calibration/skill is weak:
- policy_validation BSS -0.143 with AR-block CI below zero;
- Cycle 25 BSS near zero with CI crossing zero;
- 2026 BSS -0.056 with CI below zero.

This is consistent with the earlier frozen conclusion that AIA should not be treated as an automatic standalone fallback.

## 4. Threshold discipline matters

At the auxiliary 0.5 threshold, AIA and fusion TSS collapse to zero or near zero in several roles because their probability scales are far below 0.5. Therefore manuscript TSS must use the pre-frozen alarm thresholds; TSS@0.5 is diagnostic only.

## 5. Manuscript consequence

The baseline suite does **not** support a predictive-superiority claim for fusion or for the Gray-Box framework as a whole.

It does support:
- substantial skill above climatology for SHARP, the latest-slot logistic and fusion;
- a modest temporal-SHARP advantage over the simple logistic in Cycle-25 TSS;
- no universal fusion advantage over SHARP;
- the interpretation that this paper's novelty should be the operational trust framework rather than a state-of-the-art predictor claim.

## 6. Claim boundary

No policy threshold, conformal rule, physical monitor or NORMAL / DEGRADED / ABSTAIN routing state was changed by this suite.
