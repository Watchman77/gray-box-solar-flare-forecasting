# Manuscript figure and table architecture — 72 h Gray-Box paper

Status: **working journal layout — 6 October 2026**

The manuscript should tell one coherent systems story rather than reproduce every audit. Main-text figures are therefore limited to the evidence required to support the operational-trust claim. Detailed calibration plots, source-provenance checks and Phase-H sensitivities move to Supplementary Material.

## Main-text figures

### Figure 1 — Gray-Box architecture and evidence chronology

**Purpose:** orient the reader before any metrics.

**Panel A — trust architecture**
- SHARP temporal branch
- AIA temporal image branch
- fixed 0.5/0.5 fusion
- calibration / Mondrian conformal layer
- seed spread / entropy / SHARP-AIA gap
- explicit magnetic-state applicability monitor
- input quality / provenance
- NORMAL / DEGRADED / ABSTAIN routing

**Panel B — chronological role timeline**
- train: 2010–2013
- model validation: Jan–Jun 2014
- probability calibration: Jul–Dec 2014
- conformal calibration: Jan–Jun 2015
- policy validation: Jul 2015–2019
- Cycle-25 diagnostic: 2021–2025
- supplementary-2026 clean v3 evaluation
- mark Phase H as post-freeze/post-hoc

**Key visual warning:** show that Cycle 25 informed the semantic v3 redesign but not new numerical trust thresholds.

**Source:** manuscript Sections 3–5 and frozen v3 policy protocol.

### Figure 2 — Cross-regime reliability degradation

**Purpose:** establish the main transfer problem.

Recommended three aligned panels:
A. TSS by branch and regime
B. Brier Skill Score by branch and regime
C. flare-class conformal coverage by branch and regime

Branches: SHARP, AIA, equal-weight fusion.

**Existing source artifacts**
- cross_cycle_branch_metrics.csv
- fig3_tss_by_regime.png
- fig1_flare_coverage_by_regime.png
- calibration_diagnostics.csv
- bss_by_regime.png

### Figure 3 — Reliability paradox: agreement tightens while trust degrades

**Purpose:** show the paper's most distinctive diagnostic result.

Plot:
- median fusion seed standard deviation
- median fusion entropy
- median SHARP-AIA probability gap
- fusion flare conformal coverage or another rare-event reliability measure

Core agreement values:
- earlier: 0.01222 / 0.13841 / 0.02783
- Cycle 25: 0.00643 / 0.11834 / 0.01923
- 2026: 0.00163 / 0.04872 / 0.00404

**Caption message:** smaller ensemble/cross-modal disagreement does not imply preserved rare-flare reliability under distribution shift.

### Figure 4 — Real AIA failure structure and graceful degradation

**Purpose:** demonstrate why the state machine is operationally useful.

Recommended panels:
A. empirical outage episode lengths/severity
B. Cycle-25 state mix under nominal inputs versus mapped AIA incidents
C. availability and TSS under nominal versus incident replay

Key result:
- nominal Cycle 25: NORMAL 0.777, DEGRADED 0.223, availability 1.0
- incident replay: NORMAL 0.721, DEGRADED 0.279, availability 1.0

### Figure 5 — Same-support baseline comparison with AR-block uncertainty

**Purpose:** prevent the paper from looking like a complexity or fusion-superiority claim.

Forest/point-range plot by evaluation role:
- latest-slot logistic
- frozen SHARP
- frozen AIA
- frozen fusion

Primary metric: TSS with active-region block-bootstrap 95% CI.

Annotate paired Cycle-25 deltas:
- SHARP − logistic TSS: +0.039 [0.0003, 0.0819]
- fusion − SHARP TSS: -0.053 [-0.091, -0.013]
- fusion − SHARP AP: -0.047 [-0.084, -0.013]

### Figure 6 — Physical applicability as a complementary monitor

**Purpose:** justify the Gray-Box physical layer without overselling it.

Recommended panels:
A. median physical MD² by regime alongside statistical agreement trend
B. leave-family-out sensitivity: all axes, no gradients, no gradients-latest
C. class-conditioned physical distance for flare versus non-flare windows

Key messages:
- H1b physical distance: 10.86 → 12.54 → 16.74
- removing gradients removes the 2026 elevation
- flare windows are closer to the training cloud than non-flare windows
- therefore physical distance is monitoring evidence, not an abstention score

## Main-text tables

### Table 1 — Dataset chronology and support
Columns: role, date interval, matched cases, positives, active-region components, scientific role/evidence status.

### Table 2 — Gray-Box components and routing semantics
Rows: SHARP probability, AIA probability, fixed fusion, raw probability calibration, Mondrian conformal state, seed spread, fusion entropy, modality gap, physical applicability, AIA availability, SHARP availability.

### Table 3 — Cross-regime branch reliability
Columns: regime, branch, TSS, AP, Brier/BSS, ECE, flare conformal coverage, singleton rate.

### Table 4 — Operational routing and fallback results
Rows: nominal Cycle 25, mapped AIA incident Cycle 25, nominal 2026, AIA-only feasibility.

### Table 5 — Baseline paired comparisons
Primary paired deltas: SHARP vs latest-state logistic, fusion vs SHARP, fusion vs latest-state logistic.

## Supplementary figures

S1. Reliability diagrams for earlier / Cycle 25 / 2026.
S2. Calibration slope, intercept and ECE by regime.
S3. Full conformal set composition and class-conditional coverage.
S4. Monthly NORMAL / DEGRADED / ABSTAIN state mix.
S5. Largest AIA outage episodes and severity distribution.
S6. AIA-only fallback recall, AP, singleton rate and flare coverage.
S7. Full physical-family shift heatmap.
S8. Physical distance by year and class.
S9. Source-series transition and monthly raw gradient values.
S10. Same-record JSOC CCD/CEA keyword parity.
S11. Full baseline point metrics and Brier/BSS intervals.
S12. Full paired baseline bootstrap delta matrix.

## Figure-order logic

The preferred narrative order is:

**system → transfer failure → confidence paradox → outage routing → baseline reality check → physical monitor**

This order keeps the paper centered on operational trust and prevents the physical-layer extension from overwhelming the primary contribution.