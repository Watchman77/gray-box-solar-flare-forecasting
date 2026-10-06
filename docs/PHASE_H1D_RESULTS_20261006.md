# Phase H1d results — source and feature sensitivity audit

Status: **EXECUTED — 6 October 2026**

H1d tested whether the H1b/H1c 2026 physical-applicability shift was broad across magnetic-state dimensions or concentrated in a specific feature family / source lineage. It made no policy change.

## 1. Main result: the 2026 physical-distance increase is gradient-dominated

The all-axis H1b representation reproduced the 2026 elevation:

| Representation | Earlier Cycle 24 median MD² / OOD | Cycle 25 median MD² / OOD | 2026 median MD² / OOD |
|---|---:|---:|---:|
| all H1b axes | 10.716 / 1.38% | 12.673 / 1.39% | 16.731 / 2.84% |
| no gradients family | 8.299 / 1.34% | 9.608 / 1.34% | 9.385 / 1.08% |
| no gradients latest only | 9.839 / 1.37% | 11.600 / 1.41% | 11.410 / 1.40% |
| no gradients or inclination | 6.225 / 1.28% | 6.835 / 1.36% | 6.818 / 1.09% |
| no inclination family | 8.569 / 1.33% | 10.090 / 1.36% | 13.520 / 5.46% |

Removing the entire gradients family eliminates the 2026 elevation relative to Cycle 25. Removing only `gradients__latest` is already sufficient to eliminate it. Therefore the H1b/H1c aggregate physical-applicability shift should be classified as **gradient-dominated**, not broad magnetic-state OOD.

The inclination family partially counteracts the gradient-driven distance: removing inclination alone increases the 2026 OOD rate rather than explaining it.

## 2. The 2026 change is concentrated after March

Post-hoc 2026 internal comparison:

| Period | Cases | ARs | Known / positive | Median MD² | q99 OOD |
|---|---:|---:|---:|---:|---:|
| Jan–Mar | 1,958 | 48 | 1,597 / 152 | 13.416 | 1.17% |
| Apr–Aug | 10,201 | 83 | 9,521 / 840 | 17.483 | 3.16% |

The dominant family moves sharply over the same partition:

- `gradients__latest`: -0.020 in Jan–Mar versus -0.888 in Apr–Aug.
- `flux_pil__latest`: -0.050 versus +0.196.
- `inclination__latest`: -0.141 versus -0.207.

The boundary was chosen after H1c and is diagnostic/post-hoc. It cannot establish causality.

## 3. High physical distance is not unique to 2026

The 2018–2019 comparison has median MD² 18.174 and OOD rate 3.39%, versus 16.731 and 2.84% for the full 2026 input cohort. However, the underlying family direction differs:

- 2018–2019 `gradients__latest`: +0.291
- 2026 `gradients__latest`: -0.764
- 2018–2019 `flux_pil__latest`: -0.587
- 2026 `flux_pil__latest`: +0.157

Therefore the scalar physical-distance score can be high for different magnetic-state compositions. It is a domain/applicability statistic, not a unique solar-cycle marker.

## 4. Source lineage changes in the supplementary 2026 cohort

The native-slot lineage reveals a material source-series change:

- Cycle 24 native slots: `source_origin=locked_sharp96`, `series=hmi.sharp_720s`.
- Primary Cycle 25 native slots: `source_origin=locked_sharp96`, `series=hmi.sharp_720s`.
- Supplementary 2026 native slots:
  - 30,240 slots from `source_origin=definitive_extension`, `series=hmi.sharp_cea_720s`;
  - 6,237 slots from `source_origin=locked_sharp96`, `series=hmi.sharp_720s`.

All 2026 index history URIs use the `jsoc_2025_2026_production_v1` namespace, while earlier cohorts also contain the older `samples_npz` namespace.

This establishes a **source/processing lineage difference** in the same cohort in which the gradient-dominated physical-distance change appears. It does not by itself prove that the series change causes the gradient shift, but it prevents an astrophysical interpretation until same-series sensitivity is performed.

## 5. Interpretation status

H1d outcome:

**PHYSICAL_SHIFT_GRADIENT_DOMINATED + SOURCE_BOUNDARY_SUSPECTED**

Supported:
- the statistical-agreement / physical-applicability contrast in H1c was real for the chosen representation;
- the aggregate 2026 physical-distance increase is almost entirely driven by `gradients__latest`;
- source-series provenance differs materially in the supplementary 2026 cohort;
- physical applicability is valuable as a diagnostic/monitor because it exposed a hidden input-distribution/source change that ensemble agreement did not reveal.

Not supported:
- broad 2026 magnetic-state OOD after removing the gradient signal;
- attribution of the gradient shift to Solar Cycle 25;
- promotion of physical MD² into a NORMAL/DEGRADED/ABSTAIN gate;
- causal attribution of the gradient shift to the CEA/non-CEA series difference without within-cohort same-series analysis.

## 6. Consequence for the Gray-Box framework

This result strengthens a different Gray-Box interpretation than initially expected.

The physical layer is not currently justified as a direct case-level rejection gate. Instead, it acts as an **independent input/applicability monitor** capable of identifying a physically meaningful feature-distribution change that is hidden by decreasing entropy, seed spread and SHARP–AIA disagreement.

That is operationally useful even when the monitor does not alter a forecast automatically.

## 7. Next step

Before H2 routing, perform H1e:

- map each 2026 forecast case and each of its three SHARP slots to `hmi.sharp_720s` versus `hmi.sharp_cea_720s`;
- quantify the exact date and case-level transition/mixing pattern;
- compare raw MEANGBZ/MEANGBH/MEANGBT and gradient-family state within the same calendar period across source-series strata where overlap exists;
- recompute physical distance on harmonized/same-series subsets;
- determine whether the April–August shift persists after source-series control.

Until H1e, the physical layer remains monitoring-only.
