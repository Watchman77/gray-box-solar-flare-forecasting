# Gray-Box Solar Flare Forecasting

Research repository for **A Gray-Box Operational Framework for Multi-Horizon Solar Flare Forecasting: Calibration, Uncertainty, and Prospective Evaluation**.

The central question is: **when should an operational user trust a solar flare forecast?**

## Status

This repository contains a working research plan, a proposed operational protocol, executable historical-baseline audits, and a versioned candidate dataset. On 1 October 2026, the local 72-hour cohort and recovered original prediction arrays were reconciled; see the [audit findings](docs/BASELINE_72H_AUDIT.md). The [new dataset build](docs/DATASET_BUILD_20261001.md) retains 153,366 candidate cases, links 113,433 verified upstream AIA–SHARP inputs, and audits provisional 48/72-hour outcomes. Its labels and operational availability are not finalized. No new Gray-Box model has been trained or operational policy validated. Protocol choices and thresholds must be finalized before confirmatory evaluation.

The initial focus is the 72-hour horizon, followed by 24-hour and 3-hour evaluation. Existing 48-hour AIA experiments remain separate baselines with their original labels and scope.

**Label-source review:** the [event reconciliation](docs/EVENT_RECONCILIATION_20261001.md) identified 1,033 source-supported positive-label correction proposals for existing 48-hour windows, including 798 matched Cycle-25 cases. The original calculation reproduces, but event-source completeness needs repair. Corrections and source evidence are versioned separately; existing labels and model results are preserved. Complete the label review before treating inherited outcomes as final Gray-Box truth.

The subsequent [daily-source region review](docs/REGION_ADJUDICATION_20261001.md) checks the 651-event association queue against 696 NOAA reports. Candidate v2 fills 55 missing regions, retains 193 disputed originals with daily support and leaves 403 associations unresolved. Its 306,732 case/horizon rows pass a separate full computational check; all 51 unit tests pass. Continuous coverage, class convention and operational availability remain dataset gates.

## Research scope

- Reuse the SHARP temporal forecasting backbone and evaluate an additional AIA image branch on matched forecast cases.
- Fit probability calibration on earlier data and assess reliability at each horizon.
- Define predictive uncertainty separately from confidence intervals for evaluation metrics.
- Detect unsupported input conditions and select **normal forecast**, **degraded forecast**, or **abstention/fallback**, with recorded reasons.
- Evaluate realistic missing observations, correlated channel outages, source changes, and solar-cycle changes.
- Combine chronological evaluation with AR-disjoint and event-disjoint sensitivity studies.
- Develop rolling historical replay with explicit availability assumptions, followed by genuinely prospective evaluation when forecasts can be logged before outcomes occur.

## Five-layer architecture

1. **Data-driven component:** temporal SHARP and optional AIA predictors.
2. **Physically interpretable component:** magnetic-complexity features or an explicitly specified physical proxy/model.
3. **Fusion layer:** combine supported branch outputs, accounting for input quality and availability.
4. **Uncertainty layer:** probability calibration and task-appropriate predictive uncertainty.
5. **Operational safety layer:** determine forecast state, reason and fallback behaviour.

SHARP and AIA are physically meaningful observations. Their inclusion alone does not establish a first-principles simulator or a validated MHD/PINN component.

## Start here

| File | Purpose |
|---|---|
| [Execution plan](docs/EXECUTION_PLAN.md) | Immediate work sequence, uncertainty candidates, paper boundaries and completion evidence. |
| [72-hour baseline audit](docs/BASELINE_72H_AUDIT.md) | Executed cohort/prediction reconciliation, source defects and rerun commands. |
| [Dataset build and quality findings](docs/DATASET_BUILD_20261001.md) | Executed inventory, provisional event labels, source hashes, checks and remaining scientific decisions. |
| [Event and label reconciliation](docs/EVENT_RECONCILIATION_20261001.md) | Source-supported correction proposals, direct NOAA report checks and provisional region recovery. |
| [Daily-source region adjudication](docs/REGION_ADJUDICATION_20261001.md) | Executed region decisions, separately versioned 48/72-hour outcomes and full computational verification. |
| [Layer protocol](docs/LAYER_PROTOCOL.md) | Inputs, methods, outputs, metrics, failure modes and decision rules for each layer. |
| [Data contract](docs/DATA_CONTRACT.md) | Forecast unit, timestamps, labels, provenance and master prediction table. |
| [Roadmap](docs/ROADMAP.md) | Ordered research milestones and completion evidence. |
| [Initial evidence](docs/INITIAL_EVIDENCE.md) | Existing work and unresolved issues carried into this project. |
| [Example protocol configuration](configs/protocol.example.json) | Machine-readable draft; no operational thresholds have been selected. |
| [Contributor guidance](AGENTS.md) | Authorship, reproducibility and research conventions. |

## Related repositories

- [solar-flare-multihorizon](https://github.com/Watchman77/solar-flare-multihorizon): original multi-horizon forecasting work.
- [solar-flare-aia-training](https://github.com/Watchman77/solar-flare-aia-training): AIA acquisition, image and magnetic baselines, fusion experiments, and cross-cycle analysis.

The AIA repository remains the home of its acquisition and training work. This repository develops the operational framework and records the exact upstream versions used by each experiment.

## Data and results

Keep large images, tensors, model checkpoints, credentials and private correspondence out of Git. Commit small reviewed manifests, schemas, configurations and results with provenance. Read [the data policy](data/README.md) before adding an artefact.

Repository owner and maintainer: **Bamidele Akinwumi — [Watchman77](https://github.com/Watchman77)**.

Licensing and manuscript authorship will be documented separately before a research release. A repository scaffold does not establish publication results or an institutional partnership.
