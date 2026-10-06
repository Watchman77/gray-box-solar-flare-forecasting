# Phase H input recovery record — 6 October 2026

Status: **EXACT SHARP TENSOR RECOVERY VERIFIED**

This record documents the recovery of the exact SHARP physical-input tensor required for the predeclared Phase-H explicit physical-state Gray-Box extension.

## Why this record exists

Phase C2 did not rerun the historical feature-space OOD analysis because the exact assembled SHARP tensor and fitted historical distance artifact were not available in that execution context. That provenance-limited decision remains unchanged.

After the final A–G 72 h freeze and before executing Phase H, the original cohort-specific SHARP source artifacts were located on the VM. Their identities were verified cryptographically against the dataset-package manifest. The canonical assembled tensor was then reconstructed using the original row mapping and verified against the manifest's canonical output hash.

This is a **newly recovered exact input for Phase H**. It does not retroactively convert Phase C2 into an executed analysis.

## Located source root

VM source root:

`/home/abmoses2000/aia_sharp16_refit9_evaluation_preparation_20261002/inputs/sharp16_matched_v1_20261001`

The following source artifacts were verified.

### Cohort indices

| Cohort | SHA-256 |
|---|---|
| Cycle 24 `index.csv.gz` | `3037e96158bae46528f2ebd508b4e49a15a814d31b33c13cef726dbb84eb055e` |
| Primary Cycle 25 `index.csv.gz` | `69aa3296579152cf6abefe5ed8109030665333f205364b5ce370a63c31c93e43` |
| Supplementary 2026 `index.csv.gz` | `1b6fe42109d8c0a472d3e77d4de84137eb83afec44facd1964b8c338febe145a` |

All three match `results/dataset_package_20261002/manifest.json`.

### Cohort SHARP tensors

| Cohort | SHA-256 |
|---|---|
| Cycle 24 `sharp16_raw_three_slot.npy` | `569daffd16b2e033dd91cfd4ae2876ddf0951c57f135164ef063c589abf40d01` |
| Primary Cycle 25 `sharp16_raw_three_slot.npy` | `21746e63ced9ae290d3332c6219502f42a4ff5f39a38fb5c485d682c075b54e2` |
| Supplementary 2026 `sharp16_raw_three_slot.npy` | `d4c930a7019514dd0dad6636feb4886bdd3882411fef93f5ee44f1d23b6bc564` |

All three match the frozen dataset-package manifest.

## Canonical reconstruction

The three tensors were reordered by each cohort's original `sharp16_row` mapping and concatenated in the original package order:

1. Cycle 24
2. Primary Cycle 25
3. Supplementary 2026

The reconstructed tensor has:

- shape: `(113433, 3, 16)`
- dtype: `float64`
- canonical SHA-256:
  `c9921be7b7fe381b43436a0e927d15d84de83e15376025ae0d8a79a6ef917639`

This hash exactly matches the canonical `sharp.npy` hash in the frozen package manifest.

Recovered working path:

`/home/abmoses2000/graybox_phase_h_recovered_input_20261006/sharp.npy`

## Scientific consequence

The exact physical-input tensor required for Phase H is no longer missing.

What has **not yet** been declared recovered by this record is the full canonical aligned package required by Notebook 25, specifically the exact case/label sidecars such as `input_index.csv.gz`, `y_primary_72h.npy`, and `known_primary_72h.npy`. Those must be recovered or rebuilt from the original versioned inventory/outcome sources rather than inferred from later predictions.

Until those sidecars are verified, Phase-H model fitting must not start.

## Provenance rule

The paper should describe the chronology accurately:

> During the original Phase-C applicability audit, the exact historical SHARP feature-space OOD artifact was unavailable and no substitute was reconstructed. After the 72 h A–G scientific freeze, the original hash-pinned cohort tensors were recovered and the canonical SHARP tensor was reproduced exactly. The recovered tensor was used only in the separately predeclared Phase-H physical-state extension.

This protects the original evidence boundary and prevents hindsight from being presented as prior confirmation.
