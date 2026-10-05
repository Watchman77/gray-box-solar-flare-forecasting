# Gray-Box 72 h checkpoint — 5 October 2026

## Purpose

This checkpoint preserves the 72-hour Gray-Box methodology and evidence before the next operational-policy revision. The central research question is **when should a flare forecast be trusted?**

## Frozen prediction corpus

The canonical non-training master prediction table contains **71,010 unique cases**:
- model validation: 3,905
- probability calibration: 2,831
- conformal calibration: 4,168
- policy validation: 13,142
- retrospective Cycle 25: 35,846
- supplementary 2026: 11,118

Training rows (25,586) are excluded from the master prediction corpus. The master-table SHA256 is recorded in `results/72h_graybox/20261005/provenance_sha256.csv`.

## Probability calibration

Raw, Platt and isotonic maps were compared on the earlier probability-calibration block. **Raw probability was selected for SHARP, AIA and fixed 0.5/0.5 fusion**. No neural-network refitting or later-role scoring was used for this choice.

December inner-selection Brier scores:
- SHARP: raw 0.062285; Platt 0.087041; isotonic 0.080173
- AIA: raw 0.101253; Platt 0.115175; isotonic 0.116370
- fusion: raw 0.075362; Platt 0.086189; isotonic 0.121418

## Marginal conformal v1

Primary alpha was frozen at 0.10, with 0.05 and 0.20 sensitivities. The first marginal split-conformal implementation used nonconformity `1 - p_true`.

The subsequent policy-validation diagnostic exposed a serious rare-event failure: despite high marginal coverage, positive-class coverage collapsed:
- SHARP positive coverage: 0.151659
- AIA positive coverage: 0.000000
- fusion positive coverage: 0.000000

This result is intentionally retained. It demonstrates that apparently good marginal coverage can hide unacceptable minority-class behaviour.

## Operational policy v1

Before opening policy-validation outcomes, q80/q90/q95 policy candidates were frozen and q90 was predeclared as primary. Alarm thresholds were selected on model-validation only.

On policy validation (13,142 cases; 422 positives), the predeclared q90 policy:
- availability: 0.969639
- positive retention: 0.504739
- abstained flares: 209
- false alarms: 927
- issued-case TSS: 0.437755

The always-issue fusion baseline:
- availability: 1.000000
- positive retention: 1.000000
- abstained flares: 0
- false alarms: 1,951
- TSS: 0.626240

Therefore the q90 v1 policy is **not accepted as an operational improvement**. It reduced false alarms but disproportionately withheld flare-producing cases and reduced issued-case TSS.

## Mondrian/class-conditional conformal v2

Because the v1 marginal conformal diagnostic showed minority-class collapse, a class-conditional (Mondrian) conformal repair was frozen using **only the original conformal-calibration block**. Policy-validation was not used to fit the v2 thresholds.

At primary alpha=0.10, calibration-block empirical coverage is approximately balanced:
- class 0: 0.900370
- class 1: 0.903646
- overall: 0.900672

The v2 thresholds and diagnostics are preserved in the accompanying CSV.

## Scientific status

`scientific_acceptance` remains false. The next step is to freeze the v2 operational policy without reading future outcomes, then evaluate it once on still-unopened future-period roles (retrospective Cycle 25 and supplementary 2026).

Large prediction tables and archives are intentionally not committed to GitHub; their cryptographic hashes are recorded instead.
