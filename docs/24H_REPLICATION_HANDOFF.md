# 24-hour replication handoff

Status: prepared after the final 72-hour scientific freeze.

## Scientific architecture to reuse

The 24-hour study should reuse:
- five-layer Gray-Box design;
- chronological development/evaluation philosophy;
- A-G research sequence;
- explicit NORMAL / DEGRADED / ABSTAIN semantics;
- calibration and class-conditional conformal evaluation;
- within-regime and cross-regime applicability analysis;
- real correlated acquisition-failure replay;
- past-only rolling monitoring with explicit outcome-maturity lag;
- claim-ledger discipline.

## Numerical quantities that must NOT be reused automatically

Do not reuse:
- 72-hour calibration parameters;
- 72-hour Mondrian quantiles;
- 72-hour alarm thresholds;
- 72-hour q90 entropy/spread/gap cut-offs;
- 72-hour fallback conclusions if the 24-hour evidence differs;
- 72-hour expected performance levels.

## 24-hour A-G sequence

A. Cross-cycle 24-hour reliability audit.
B. 24-hour probability calibration and diagnostics.
C. 24-hour UQ/applicability/shift analysis.
D. AR-disjoint and event-disjoint analysis where provenance permits.
E. Real AIA acquisition-failure replay and episode structure.
F. AIA-only fallback feasibility at 24 h.
G. Past-only rolling operational replay.

## Freeze rule

The 24-hour policy must be frozen using earlier 24-hour development data before any clean later evaluation is opened. The completed 72-hour interpretation is historical evidence and must not be altered to match later-horizon findings.
