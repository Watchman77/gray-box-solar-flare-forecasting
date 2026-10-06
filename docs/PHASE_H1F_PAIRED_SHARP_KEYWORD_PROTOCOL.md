# Phase H1f — paired same-record SHARP keyword parity audit

Status: **PRE-EXECUTION PROVENANCE TEST — 6 October 2026**

H1f directly tests whether the 16 SHARP summary keywords differ between `hmi.sharp_720s` and `hmi.sharp_cea_720s` when queried for the **same HARP and same T_REC**.

No forecasting model, threshold or routing policy is changed.

## Rationale

H1e showed an almost exact temporal transition from `hmi.sharp_720s` to `hmi.sharp_cea_720s` on 4 April 2026, but existing cohort records do not contain simultaneous same-time values from both series.

Authoritative HMI documentation distinguishes the two series by their map segments / coordinate representation while describing SHARP active-region summary parameters as keywords generally computed from high-confidence values in the remapped CEA representation. Direct same-record parity is therefore the decisive provenance test.

## Deterministic sample

Use the supplementary-2026 native-slot ledger.

Primary sample:
- 8 evenly spaced latest-slot records per calendar month from January through August 2026;
- all available latest-slot records nearest the 4 April transition within a fixed ±12-hour window;
- de-duplicate exact HARP/T_REC pairs.

Expected total is roughly 64–80 records.

This sample is chosen without using flare labels or forecast errors.

## Query

For each selected `HARPNUM, T_REC`, query both:

- `hmi.sharp_720s[HARPNUM][T_REC]`
- `hmi.sharp_cea_720s[HARPNUM][T_REC]`

Retrieve:
- the 16 frozen SHARP feature keywords;
- `HARPNUM`, `T_REC`, `DATE`, `CODEVER7`, `CALVER64`, `CMASK`, `QUALITY`.

No image segments are downloaded.

## Outputs

- paired query receipt;
- per-record/per-feature numeric differences;
- exact-equality and numerical-equality rates;
- maximum absolute and relative difference by feature;
- metadata comparison;
- missing-record audit.

## Interpretation

Possible conclusions:

- `SUMMARY_KEYWORDS_PARITY_CONFIRMED`
- `SUMMARY_KEYWORDS_DIFFER_BY_SERIES`
- `PARTIAL_PARITY_OR_RECORD_GAPS`
- `JSOC_QUERY_UNAVAILABLE`

If parity is confirmed, the April 2026 shift cannot be attributed simply to choosing the CEA series name for the summary keywords; temporal population / selection / other lineage effects must be investigated instead.

If systematic same-record differences exist, quantify them before any physical-domain interpretation.

The frozen v3 policy remains unchanged.
