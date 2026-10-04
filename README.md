# Gray-Box Solar Flare Forecasting

Research repository for **A Gray-Box Operational Framework for Multi-Horizon Solar Flare Forecasting: Calibration, Uncertainty, and Prospective Evaluation**.

The central question is: **when should an operational user trust a solar flare forecast?**

## Status

**Dataset available:** [open and use the aligned package](docs/DATASET_QUICKSTART.md). It includes a full readable CSV (153,366 rows × 104 columns), SHARP tensors for 113,433 matched cases, existing AIA object references, 48/72-hour candidate labels and missingness masks. A small preview and a NumPy-only exploratory baseline runner are included. No repeated AIA download is needed.

**Pipeline exercised:** both exploratory SHARP horizons have completed fitting and saved-model replay. The AIA reader now supports older channel-name and newer wavelength metadata. Dataset preflight, failed-run recovery and completed-run reuse are covered by 67 passing tests. See [implemented fixes and remaining scientific limits](docs/PIPELINE_STATUS.md).

**Start training in the notebook:** [72-hour SHARP training](notebooks/01_SHARP_72h_Training.ipynb) contains the data checks, frozen chronological blocks, three-seed GRU training, matched logistic reference, checkpoint replay, learning curves and raw evaluation results. It is self-contained apart from the pinned data archive and listed libraries. These are exploratory candidate-label results.

**Calibration executed:** [notebook 02](notebooks/02_SHARP_72h_Calibration.ipynb) fits Platt-style and isotonic maps on earlier data, compares them with raw probabilities, and saves reliability diagrams, proper scores, calibration diagnostics and paired group-resampling intervals. December 2014 selection retained raw probabilities for both backbones. The fitted maps did not improve Brier scores on the recorded Cycle-25/2026 evaluations. [Verification and outputs](results/sharp72_calibration_20261002/verification.json).

**Conformal uncertainty evaluated:** [notebook 03](notebooks/03_SHARP_72h_Conformal_Uncertainty.ipynb) compares pooled and class-conditional sets at 90% and 95% targets, fitting only the reserved January–June 2015 block. At the 90% target, GRU class-conditional flare coverage is 73.5% in Cycle 25 and 43.5% in partial 2026. These fixed thresholds fail to retain nominal class coverage; aggregate coverage alone conceals much poorer flare coverage for pooled sets. The notebook preserves all comparisons, annual plots, group-resampling intervals and missing-input cases. All 14 code cells executed and all 84 tests passed. [Verification and outputs](results/sharp72_conformal_20261002/verification.json). The operational policy remains unvalidated.

**Rolling uncertainty evaluated:** [notebook 04](notebooks/04_SHARP_72h_Rolling_Conformal.ipynb) compares fixed class-conditional thresholds with daily updates using 90, 180 and 365 days of matured earlier outcomes. July 2015–2019 method selection retained 180 days for both frozen backbones. At the primary 90% target, GRU flare coverage rises from 73.5% to 87.9% in Cycle 25 and from 43.5% to 78.3% in partial 2026, while sets containing both outcomes rise to about 36–37%. Coverage remains below target. All 14 code cells executed, all 92 tests passed, and a separate calculation verified the results. [Verification and outputs](results/sharp72_rolling_20261002/verification.json). This retrospective replay uses later labels only after their assumed availability; it is not a label-free cross-cycle or prospective validation.

**Decision-state experiment completed:** [notebook 05](notebooks/05_SHARP_72h_Decision_Policy.ipynb) tests candidate normal, degraded and abstention states using input checks, recent calibration support, feature distance and three-seed disagreement. At the parent 90% conformal target, GRU-only guards reduce observed error among issued decisions from 16.9% to 12.8% in 2021–2025 and from 17.0% to 15.7% in partial 2026, while deferring about 53% of flare windows. The logistic fallback makes errors on 49.2% and 46.4% of its routed cases; this does not support adopting it as an approved degraded mode. All 12 code cells executed and all 101 tests passed. [Independent verification and outputs](results/sharp72_policy_20261002/verification.json).

**Guard tradeoffs diagnosed:** [notebook 06](notebooks/06_SHARP_72h_Gate_Diagnostics.ipynb) reconciles all 16 guard subsets, overlapping triggers and fallback error types. Removing only seed disagreement would restore 687 true alerts and 836 false alarms in 2021–2025, and 124 true alerts and 165 false alarms in partial 2026, at the parent 90% target. Much of the fallback reinstates exactly the GRU decision rejected by that check. No new threshold or policy is selected. All 11 cells executed and all 109 tests passed. [Independent verification](results/sharp72_gate_diagnostic_20261002/verification.json).

**Multimodal integration prepared:** [notebook 07](notebooks/07_SHARP_AIA_GOES_72h_Integration.ipynb) exports 96,596 candidate comparison cases, reconciles all 113,433 parent SHARP rows, and lists 113,037 unique AIA objects plus past-only GOES case requests. It defines SHARP, AIA, GOES, SHARP+GOES, SHARP+AIA and three-source comparisons, with checks against wrong horizons, changed splits, future observations and silent case loss. All eight cells executed and all 118 tests passed. This is preparation, not AIA/GOES training or a fusion result; archive-wide image integrity and an accepted continuous-XRS predictor matrix remain pending. [Independent verification](results/multimodal72_preparation_20261002/verification.json).

**AIA technical check completed:** [notebook 08](notebooks/08_AIA_72h_Loader_and_Model_Check.ipynb) checks 42 existing images from 14 boundary cases across the seven frozen roles. It uses the manifest's 72-hour targets, training-only preprocessing, fresh temporal CNN–GRU weights, two technical optimization steps, and exact saved-model replay. This checks software compatibility across both image metadata formats; it is not full AIA training or representative forecast evaluation. The temporary checkpoint/statistics must not initialize the scientific experiment. [Recorded results and limitations](results/aia72_canary_20261002/summary.json).

**AIA GPU training prepared:** [notebook 09](notebooks/09_AIA_72h_Shared_GPU_Training.ipynb) fits fresh 72-hour models using the existing shared VM cache, fixed earlier training/selection roles, three seeds and resumable checkpoints. Execution requires the agreed AIA GPU handoff and common exclusive lock. The source notebook's presence is not evidence that full training has completed; use the executed notebook and run receipts. [Protocol and resource limits](docs/AIA72_SHARED_GPU_PROTOCOL_20261002.md).

**AIA completion run active — 4 October:** One continuous invocation started at 08:59 UTC, following seven passing checks both locally and on the VM. The 09:15 UTC coordination probe verified one L4 GPU model process, seed 29 completing epoch 7 with best epoch 3, and seed 43 training at epoch 1, cursor 1,680/25,586 and 105 steps. Seed 17 is fitted and replayed; seed 29 still requires final checkpoint/replay acceptance. [Verified transition](results/aia72_completion_training_preparation_20261004/seed29_to_seed43_transition.json). [Notebook 15](notebooks/15_AIA_72h_Completion_Training.ipynb) preserves the scientific stopping rules. The ceiling is 35 hours fitting within a 36-hour reservation, with immediate stop on completion; this is a ceiling, not expected consumption. The seed-17 timing gives a roughly five-hour remaining scenario if seeds 29/43 finish at epochs 7/6. All six CPU preview cells passed; full GPU notebook completion remains pending. [Live start evidence](results/aia72_completion_training_preparation_20261004/live_start_verification.json) and [full phase plan](results/aia72_completion_training_preparation_20261004/compute_proposal.json). All prior timing remains preserved, including the expired unused reservation. AIA yields scheduling priority through Gray-Box's subsequent saved-model replay, inference and matched SHARP/AIA analysis; the technical return at a phase boundary does not release that priority. Full training, later AIA evaluation and fusion are not yet complete.

**Next-phase preparation — 4 October:** The inference-only saved-model replay core passed eight CPU tests; no scientific replay of seeds 29/43 has run yet. All 113,037 required image metadata records are pinned for the 96,596-case comparison, preserving 35,050 original training-image pins. The remaining 77,987 pins reconcile to raw cloud metadata responses. There are 192 blocks of at most 512 cases, preserving 16-case batches and within-role order; the largest block needs at most 3,958,743,130 bytes of image files. Four block/source-boundary tests passed. This used metadata only: no image downloads or VM operations. [Preparation verification](results/aia72_inference_metadata_preparation_20261004/verification.json). Actual cache reconciliation, reviewed streaming, saved-model replay and later inference remain pending.

This repository contains a working research plan, a proposed operational protocol, executable historical-baseline audits, a versioned candidate dataset, and executed SHARP training, uncertainty and decision-policy notebooks. On 1 October 2026, the local 72-hour cohort and recovered original prediction arrays were reconciled; see the [audit findings](docs/BASELINE_72H_AUDIT.md). The [new dataset build](docs/DATASET_BUILD_20261001.md) retains 153,366 candidate cases, links 113,433 verified upstream AIA–SHARP inputs, and audits provisional 48/72-hour outcomes. Its labels and operational availability are not finalized. AIA fusion and operational validity remain unestablished. Protocol choices and thresholds must be finalized before confirmatory evaluation.

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
