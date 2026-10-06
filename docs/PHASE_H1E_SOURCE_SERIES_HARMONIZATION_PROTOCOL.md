# Phase H1e — source-series harmonization audit

Status: **PRE-EXECUTION DIAGNOSTIC FREEZE — 6 October 2026**

H1e is required because H1d showed that the aggregate 2026 physical-applicability shift disappears when `gradients__latest` is removed and that the supplementary-2026 native-slot ledger mixes `hmi.sharp_720s` with `hmi.sharp_cea_720s`.

No routing rule may be selected in H1e.

## Primary questions

1. Exactly when does the source-series composition change within the 2026 cohort?
2. Are 2026 forecast histories pure `hmi.sharp_720s`, pure `hmi.sharp_cea_720s`, or mixed across the three slots?
3. Within overlapping dates/active regions where both source series exist, do raw gradient quantities differ systematically by series?
4. Does the `gradients__latest` shift persist when analysis is restricted to a single source series?
5. Does the physical MD² / OOD contrast persist after excluding mixed-series histories?
6. Is the April–August H1d boundary aligned with source-series composition?

## Required outputs

- native-slot ledger schema and identity audit;
- case-level source-series composition table;
- monthly source-series fractions;
- raw MEANGBZ/MEANGBH/MEANGBT summaries by series and month;
- same-period/same-AR paired or matched comparisons where possible;
- H1b physical distance by source-series composition;
- a final interpretation category:
  - `SOURCE_SERIES_EXPLAINS_GRADIENT_SHIFT`
  - `SOURCE_SERIES_PARTIAL_CONFOUND`
  - `SHIFT_PERSISTS_WITHIN_SERIES`
  - `INSUFFICIENT_OVERLAP_TO_RESOLVE`.

## Claim discipline

Even if a strong association is observed, H1e establishes a source-series confound or sensitivity, not a causal instrument-calibration mechanism unless the upstream product definitions and processing lineage support that stronger statement.

The frozen v3 policy remains unchanged.
