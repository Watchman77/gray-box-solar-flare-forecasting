# 72 h baseline suite protocol

Status: **PRE-EXECUTION FREEZE — 6 October 2026**

Purpose: strengthen the manuscript comparison using simple, leakage-controlled baselines on the exact frozen 72 h support. No neural model is retrained and no operational policy is changed.

## Frozen inputs

- canonical non-training AIA master (71,010 rows), SHA256 `17300093b88b7c61ecadf5eb28acba0d6d023b340d7740807852dd4352b22b46`;
- frozen SHARP 72 h ledger (113,433 rows), SHA256 `7368c002a978321cdc67221564f3b7901687badc7aba28ab9df615179d1e017b`;
- exact recovered SHARP tensor `(113433,3,16)`, SHA256 `c9921be7b7fe381b43436a0e927d15d84de83e15376025ae0d8a79a6ef917639`;
- already-frozen alarm thresholds from model validation.

## Baselines / frozen branches

1. **Training climatology**: constant probability equal to the frozen training prevalence.
2. **Latest-slot SHARP logistic**: one ordinary L2 logistic regression fit only on the 25,586 frozen training rows using all 16 SHARP values at the latest available history slot (t-96 min). Median imputation and standard scaling are fit on training only.
3. **Frozen SHARP**: existing GRU probability, unchanged.
4. **Frozen AIA**: existing AIA probability, unchanged.
5. **Frozen fusion**: prespecified 0.5 SHARP + 0.5 AIA probability, unchanged.

No tree model, neural retraining, hyperparameter search or policy reselection is permitted.

## Threshold discipline

The manuscript's primary TSS must not be compared at arbitrary 0.5 thresholds when the frozen Gray-Box branches already use alarm thresholds selected on `model_validation`.

- frozen SHARP/AIA/fusion: use the already-frozen alarm thresholds;
- latest-slot logistic: select one threshold on `model_validation` only, using the same deterministic max-TSS / max-HSS / max-recall / lower-threshold tie rule used by the frozen branches;
- climatology: threshold 0.5 (its TSS is uninformative and expected to be 0 for the rare-event prevalence).

TSS@0.5 is retained only as an auxiliary diagnostic.

## Evaluation support

Primary evaluation roles:

- `policy_validation`
- `retrospective_cycle25`
- `supplementary_2026`

Development/calibration roles are not presented as independent evaluation.

## Metrics

For every model and evaluation role:

- TSS at the prespecified/frozen threshold;
- TSS at 0.5 (auxiliary);
- average precision;
- Brier score;
- Brier skill score relative to the frozen training-climatology probability.

## Confidence intervals

Use an active-region cluster bootstrap over `region_component_id`.

- 2,000 deterministic bootstrap draws;
- resample active-region components with replacement and retain all windows from each sampled component;
- use the **same bootstrap draw for every model within a role**;
- report percentile 95% CIs for TSS, AP, Brier and BSS;
- report paired bootstrap deltas for each non-climatology model versus climatology and for the latest-slot logistic versus frozen SHARP/fusion.

The bootstrap is inferential support for repeated windows; it does not create independence that is absent from the data.

## Claim boundary

This suite supports manuscript comparison only. It does not alter the Phase-H physical-monitor conclusion and does not change NORMAL / DEGRADED / ABSTAIN routing.
