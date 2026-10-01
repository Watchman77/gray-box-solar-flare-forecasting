# Research roadmap

The detailed near-term sequence is in [EXECUTION_PLAN.md](EXECUTION_PLAN.md). The historical 72-hour cohort/prediction audit has been executed; the new operational experiments remain planned. Completed upstream experiments remain attributed to their original repositories and frozen protocols.

| Milestone | Work | Completion evidence |
|---|---|---|
| 1. Protocol and scope | Agree targets, horizons, issue cadence, layers, operational states and primary comparisons. | Versioned protocol, definitions and decision register. |
| 2. Data readiness | Consume the completed AIA extension inventory; establish new 72-hour labels and verify SHARP/AIA timing, region mapping, quality and missingness. | Versioned inventory and master-case manifest with audit counts. |
| 3. Production-shift investigation | Reuse completed upstream frozen 48-hour extension evaluation; investigate processing effects only with justified matched evidence. | Preserved baseline, monthly metrics and matched processing analysis separating evidence from causal claims. |
| 4. Matched baselines | Reproduce SHARP; compare AIA and fusion on identical cases, initially at 72 hours. | Row-level predictions, complete provenance, paired uncertainty and availability accounting. |
| 5. Calibration and UQ | Select methods using designated earlier data; evaluate probability reliability and explicitly defined uncertainty. | Calibration fits, reliability plots, coverage/set-size or interval-width results as appropriate. |
| 6. Safety and outages | Implement normal/degraded/abstain decisions and validated fallback; replay realistic block/channel outages. | Frozen policy, risk–coverage, missed events, false alarms, latency and full-population metrics. |
| 7. Multi-horizon forward evaluation | Extend to 24/3 hours, run ordered rolling replay, then prospective logging where feasible. | Dated forecast ledger, mature outcomes and reproducible evaluation reports. |
| 8. Manuscript and release | Compare closest work, report limitations/ablations, document data/code access and contributions. | Traceable tables/figures, reproducibility package and reviewed manuscript. |

## Evaluation boundaries

The upstream 2021–2025 and supplementary 2026 temporal AIA tests have already been evaluated and diagnosed. Preserve their original results. If they inform a new policy, treat that policy as a new experiment with independent evaluation; do not describe the same cases as untouched confirmation.

Retrospective replay and prospective forecasting must be described separately. A newly downloaded historical observation is still historical.

## Immediate dependencies

1. AIA acquisition and pipeline provenance from the AIA project.
2. Event catalogue coverage and reliable region associations.
3. An agreed common forecast-case table.
4. Operational requirements from the supervisor and any confirmed external collaboration.

Met Office participation, data access and endorsement are not assumed. Public data can support the initial prototype while any collaboration is clarified.
