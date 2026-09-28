# Initial evidence and boundaries

Recorded 28 September 2026. These observations describe reviewed upstream snapshots, not newly executed Gray-Box experiments.

| Upstream repository | Reviewed commit |
|---|---|
| [solar-flare-multihorizon](https://github.com/Watchman77/solar-flare-multihorizon) | `f05ff192356a79ec7fbbc0ba32dbdc289085a23a` |
| [solar-flare-aia-training](https://github.com/Watchman77/solar-flare-aia-training) | `9dd35d301e153030261ea78d5419ad5e7bf5eda4` |

## Reusable work

- SHARP temporal predictors and a physically motivated proxy comparator.
- Six-channel regional AIA acquisition and loaders.
- Executed AIA–SHARP intermediate fusion at a 48-hour target horizon.
- Temporal AIA CNN–GRU evaluation with a frozen Platt calibrator and threshold.
- Cross-cycle results and post-test input/score-shift diagnostics.

## Issues to carry forward

- The saved SHARP and fusion summary scores use different eligible test populations. Incremental fusion benefit requires a matched comparison.
- The saved temporal AIA recall at the frozen threshold changes from approximately 70.14% in 2024 to 0.78% in 2025, while ROC-AUC is approximately 0.737 and 0.728. These are results on forecast cases, not independent flare counts.
- A production/source change coincides with the year boundary. The association does not prove that processing caused the score shift.
- The inspected May 2026 event-catalogue snapshot does not support complete outcomes through August. The active AIA extension needs updated, verified labels and complete horizon follow-up.
- Source timing, historical availability and label completeness remain unresolved in the upstream scientific-clearance record.
- The original notebook split path requires stronger time-boundary/purge and dependence-aware evaluation for operational claims.
- The full uncertainty and three-state safety architecture is not established by the upstream configuration checklist.

## Sources

- [Fusion branch comparison](https://github.com/Watchman77/solar-flare-aia-training/blob/9dd35d301e153030261ea78d5419ad5e7bf5eda4/results/metrics/aia_resnet18_sharp24h_intermediate_fusion_cycle24_fullnatural_allfold_branch_comparison_with_fusion.csv)
- [Yearwise temporal AIA metrics](https://github.com/Watchman77/solar-flare-aia-training/blob/9dd35d301e153030261ea78d5419ad5e7bf5eda4/results/metrics/19A4_cycle25_yearwise_metrics.csv)
- [Production/score-shift diagnostic](https://github.com/Watchman77/solar-flare-aia-training/blob/9dd35d301e153030261ea78d5419ad5e7bf5eda4/docs/19A4E_E3_CYCLE25_AIA_SHIFT_DIAGNOSTIC_CHECKPOINT_2026-09-19.md)
- [Upstream decision register](https://github.com/Watchman77/solar-flare-aia-training/blob/9dd35d301e153030261ea78d5419ad5e7bf5eda4/docs/AIA_DECISION_REGISTER_2026-09-16.md)

This project should establish novelty through controlled operational evaluation and demonstrated value. Multimodal fusion, calibration and conformal prediction are not new solely because they are combined under a Gray-Box title.
