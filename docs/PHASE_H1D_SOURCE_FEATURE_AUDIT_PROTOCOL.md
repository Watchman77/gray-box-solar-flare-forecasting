# Phase H1d — source and feature sensitivity audit

Status: **PRE-EXECUTION DIAGNOSTIC FREEZE — 6 October 2026**

H1d investigates whether the H1c physical-state shift is broad solar-regime structure or disproportionately driven by a feature family / source-processing change. It does not alter the A–G operational policy and cannot create a case-level physical gate.

## 1. Why H1d is required

H1c established a robust contrast between rising 2026 physical distance and falling statistical uncertainty/agreement signals. However:

- the year-by-year pattern is non-monotonic;
- 2018–2019 also show elevated physical distance;
- the 2026 increase becomes prominent mainly from April onward;
- `gradients__latest` dominates the standardized 2026 median shift.

These facts require a source/feature audit before interpreting the shift as solar-cycle physics.

## 2. Frozen H1d analyses

### A. Raw-feature monthly audit
For all 16 raw SHARP features:
- monthly median and IQR from 2024 through 2026;
- robust standardized median relative to the original training transform;
- explicit focus on MEANGBZ, MEANGBH and MEANGBT.

### B. 2026 internal boundary audit
Report Jan–Mar versus Apr–Aug 2026 for:
- raw feature medians/IQRs;
- MPSV latest-state families;
- physical MD² and OOD rate;
- active-region count and class prevalence.

This boundary is diagnostic and selected after H1c; it must be labelled post-hoc.

### C. Quiet-regime comparison
Compare 2018–2019 with 2026:
- class prevalence;
- active-region count;
- raw gradient distributions;
- inclination and flux/PIL summaries;
- physical MD².

Purpose: determine whether high physical distance is a generic low/transition-activity effect rather than a unique 2026 phenomenon.

### D. Leave-family-out physical applicability
Recompute the training-only Ledoit-Wolf MD² with:
1. all H1b axes;
2. gradients family removed;
3. inclination family removed;
4. gradients + inclination removed.

Threshold each representation at its own training q99.

If the 2026 shift disappears when gradients are removed, describe the applicability signal as gradient-dominated rather than broad-state OOD.

### E. Leave-axis-out sensitivity
Repeat after removing only `gradients__latest`.

This tests whether one dominant axis alone produces the result.

### F. Source-lineage inventory
Inspect the recovered 2026 cohort index, build receipt, independent review and native-slot ledger (when present) for:
- path/source identifiers;
- row-generation or storage changes;
- processing-version fields;
- changes around March/April 2026.

This is provenance inspection, not proof of instrument calibration equivalence.

## 3. Decision outcomes

H1d must end with one of:

- `PHYSICAL_SHIFT_BROAD_AND_SOURCE_STABLE`
- `PHYSICAL_SHIFT_GRADIENT_DOMINATED`
- `PHYSICAL_SHIFT_SOURCE_BOUNDARY_SUSPECTED`
- `PHYSICAL_SHIFT_MIXED_OR_UNRESOLVED`

No H2 policy gate is permitted unless the source/feature interpretation is sufficiently stable.
