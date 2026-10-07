# Multi Horizon Gray Box Protocol

This protocol extends the frozen 72-hour Gray-Box study to 24-hour and 3-hour forecasts. It preserves the 72-hour artifacts and requires horizon-specific labels, splits, calibration maps, conformal quantiles, policy cutoffs, model checkpoints, and result receipts.

## Current gate

The committed aligned dataset currently contains explicit 48-hour and 72-hour targets only. No 24-hour or 3-hour target arrays are treated as available. The first task is therefore label construction and verification from the versioned event sources before any model fitting.

## Forecast definition

For each horizon `h` in `{24, 3}`, one row represents a `forecast_case_id`, issue time, active-region identity, target definition, and horizon. The primary target is an M/X-class event associated with the same region in the half-open interval `(issue_time, issue_time + h hours]`, with elapsed time computed in TAI before UTC presentation. Unknown or immature outcomes remain unknown and are never converted to negatives.

## Planned evidence roles

The 24-hour and 3-hour studies will use separate chronological role assignments. Training, model validation, probability calibration, conformal calibration, policy validation, Cycle-25 evaluation, and supplementary-2026 evaluation must be recorded per horizon. Existing 72-hour roles and thresholds must not be copied into either shorter-horizon study.

## Evaluation sequence

1. Build and independently verify 24-hour and 3-hour labels.
2. Audit prevalence, maturity, missingness, region overlap, and event-disjoint support.
3. Train SHARP models with horizon-specific targets and frozen chronological splits.
4. Add probability calibration and conformal uncertainty using earlier data only.
5. Evaluate SHARP before allocating GPU resources to AIA.
6. Train/replay AIA branches only when the required image support and GPU reservation are available.
7. Compare SHARP-only, AIA-only, and fusion branches on identical support.
8. Evaluate the Gray-Box trust policy, structured outages, AR/event-disjoint sensitivity, and prospective-style replay.

## Claim boundary

Until the label builder and its verification receipt exist, 24-hour and 3-hour results are a planned extension, not completed experiments. A successful SHARP run does not establish multimodal or operational validity for either horizon.
