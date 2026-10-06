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

**SUMMARY_KEYWORDS_PARITY_CONFIRMED**

The April 2026 switch from cohort records sourced as `hmi.sharp_720s` to records sourced as `hmi.sharp_cea_720s` does **not** explain the observed 2026 gradient-family shift through a change in the 16 SHARP keyword values themselves.

For the tested same HARP / same T_REC records, the summary keywords are exactly identical between series.

Therefore the H1d/H1e gradient shift should no longer be described as a series-definition artefact. The remaining plausible explanations include:

- genuine temporal change in the sampled active-region population;
- active-region selection/composition effects;
- viewing-geometry / disk-position effects;
- other upstream sampling or lineage differences not represented by the series name;
- combinations of these factors.

The result still does not establish a unique solar-cycle causal mechanism.

## 5. Consequence for the physical layer

The physical-applicability monitor remains scientifically meaningful:

- it detected a large gradient-family distribution shift;
- that shift is not caused by numerical disagreement between the two SHARP series' summary keywords;
- statistical confidence/agreement signals were simultaneously becoming smaller;
- physical distance remains unsuitable as a naive case-level rejection score.

Accordingly, Phase H supports an **independent physical/applicability monitoring layer**, not a direct physical abstention gate.

## 6. Routing consequence

The frozen v3 NORMAL / DEGRADED / ABSTAIN routing remains unchanged.

No evidence from H1–H1f justifies changing routing based directly on physical Mahalanobis distance.

The next scientific step should characterize whether the gradient-family change persists after controlling for active-region composition and viewing geometry, rather than forcing an H2 gate.
