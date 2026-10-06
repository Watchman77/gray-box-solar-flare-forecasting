# Phase H1f results — paired same-record SHARP keyword parity

Status: **EXECUTED — 6 October 2026**

H1f directly queried `hmi.sharp_720s` and `hmi.sharp_cea_720s` for the same HARP and exact same T_REC and compared the 16 frozen SHARP summary keywords.

## 1. Query coverage

- deterministic paired records: 120
- total JSOC series queries: 240
- successful queries: 240
- failed queries: 0
- paired same-record comparisons: 120

Both requested series were available for every sampled record.

## 2. Exact numerical parity

All 16 frozen SHARP features were numerically identical across the two series for all 120 paired records.

For every feature:

- exact numeric equality at 1e-12 tolerance: 100%
- equality at 1e-9 tolerance: 100%
- maximum absolute difference: 0
- median absolute difference: 0
- maximum relative difference: 0

Features:
`ABSNJZH, MEANALP, MEANGAM, MEANGBH, MEANGBT, MEANGBZ, MEANJZD, MEANJZH, MEANSHR, R_VALUE, SAVNCPP, SHRGT45, TOTPOT, TOTUSJH, TOTUSJZ, USFLUX`.

## 3. Metadata parity

The paired records also had exact string equality for:

- HARPNUM
- T_REC
- CODEVER7
- CALVER64
- CMASK
- QUALITY

All metadata equality rates were 100%.

## 4. Interpretation

H1f outcome:

**SUMMARY_KEYWORDS_PARITY_CONFIRMED ON THE QUERIED JSOC PAIRS**

For the 120 tested same-HARP / same-T_REC records, the 16 SHARP summary keywords are exactly identical between `hmi.sharp_720s` and `hmi.sharp_cea_720s`.

This rules out a numerical keyword-definition difference **within those matched JSOC pairs**.

It does **not**, by itself, establish the cause of the local April 2026 drop in the stored cohort. Two evidentiary links must remain explicit:

1. the queried sample must demonstrably include post-4-April 2026 CEA-era times (the H1f notebook was designed to sample January–August plus a transition window, but the compact parity table alone does not document the realized time coverage);
2. the local stored tensor values for the same HARP/T_REC records must be shown to match the corresponding JSOC keyword values.

Until local-versus-JSOC equality is demonstrated on overlapping CEA-era records, the April 2026 gradient shift remains a gradient-dominated change at the product/provenance boundary whose solar-versus-pipeline origin is unresolved.

Remaining explanations include genuine temporal population change, active-region selection/composition, viewing geometry, upstream extraction/join effects, or combinations of these factors.

## 5. Consequence for the physical layer

The physical-applicability monitor remains scientifically meaningful:

- it detected a large gradient-family distribution shift in the stored cohort;
- matched JSOC records do not reproduce a CEA-versus-CCD keyword difference;
- statistical confidence/agreement signals were simultaneously becoming smaller;
- physical distance remains unsuitable as a naive case-level rejection score.

Do not state that the stored 2026 population shift is definitively genuine or astrophysical until local-versus-JSOC parity is documented for overlapping CEA-era records.

Accordingly, Phase H supports an **independent physical/applicability monitoring layer**, not a direct physical abstention gate.

## 6. Routing consequence

The frozen v3 NORMAL / DEGRADED / ABSTAIN routing remains unchanged.

No evidence from H1–H1f justifies changing routing based directly on physical Mahalanobis distance.

The next scientific step should characterize whether the gradient-family change persists after controlling for active-region composition and viewing geometry, rather than forcing an H2 gate.
