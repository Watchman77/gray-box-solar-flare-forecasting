# Pipeline fixes and executed checks — 2 October 2026

The exploratory SHARP path has now completed real model fitting, saved-model inference and full-population output at both 48 and 72 hours. The AIA input reader accepts both source metadata schemas and has decoded three pinned cloud objects, including the last August 2026 case. This closes the software issues found in this pass; it is not validation of a trained AIA fusion or operational safety model.

| Issue | Implemented handling | Executed check |
|---|---|---|
| Older AIA objects use `channels`; newer objects use `wavelengths` | `aia_io.load_aia_frame` accepts either, requires the exact six-channel order, rejects disagreement, preserves pixels and ignores embedded old labels. | Actual 2019, 2025 and 17 August 2026 files loaded; all have finite `(512,512,6)` arrays. |
| Dataset files or row order can change | Streamed checksums enabled by default; case/index/tensor identities and all four target/mask combinations checked before fitting. | Entire current package passed. Altered-file and mismatched-label tests fail clearly. |
| Missing values could be confused with no flare | Unknown labels excluded from fitting/scoring metrics; unsupported feature rows excluded from fitting and receive no forecast. | Boundary tests and both actual horizon runs passed. Unknown-label cases can still receive predictions when inputs are valid. |
| Timestamp format variation or future inputs | Accept ISO time formatting variants; require consistent identities, past-only ordered histories and valid outcome endpoints. | Mixed-format failure test and current histories passed. Training labels must mature by the declared cutoff. |
| Interrupted output could block retries or appear complete | Write to an incomplete attempt directory; publish the final directory only after completion. Keep failed attempts for diagnosis. | Injected fit failure left the final path free; retry succeeded. |
| Accidental reuse after code, data or target changes | `--resume` requires matching data/code/target/runtime contract and unchanged saved artifact hashes. | Actual 72-hour run reused without retraining; corrupted-result test rejected reuse. |
| Saved weights alone might not reproduce preprocessing | Save the training transform with the model; load it and replay all supported predictions before completion. | Predictions were identical after reload at both horizons. |
| Dropped missing-input cases could disappear from reporting | Save `population_predictions.csv.gz` with every candidate and an explicit forecast status. | Each horizon retains all 153,366 cases; 39,933 have no assembled SHARP input and no score. |

**67 tests pass.** The [receipts](../results/pipeline_validation_20261002/) contain both completed baseline summaries, the bounded AIA check and code hashes. Model weights and per-case predictions are local under `outputs/sharp48_pipeline_v2/` and `outputs/sharp72_pipeline_v2/`. The simple logistic model converged at both horizons. Its numerical scores are exploratory diagnostics, not paper conclusions.

## Use the corrected pipeline

Use the current repository scripts, or extract `data/processed/gray_box_pipeline_tools_v1.zip` separately from the unchanged dataset archive. The tool bundle includes the corrected reader, baseline runner and AIA schema adapter. Keeping code separate preserves the original dataset checksum and avoids another download of the scientific data.

```bash
python scripts/run_exploratory_sharp.py \
  --dataset data/processed/gray_box_aligned_v1 \
  --output-dir outputs/sharp72_pipeline_v2 \
  --horizon 72 --scope primary --fit --resume
```

For a new experiment, use a new output path. The completed-run contract prevents accidentally mixing horizons, target scopes, dataset versions or revised code. An interrupted fit can be rerun using its original final output path; the incomplete attempt remains separate. The small CPU baseline restarts fitting after failure rather than claiming optimizer-checkpoint resumption.

For AIA, call `load_aia_frame` from `scripts/aia_io.py`; do not assume every raw cloud object has a `channels` key. Only three actual objects were checked here. Full-image integrity and a trained fusion model remain subsequent work. Historical AIA/SHARP delivery and continuous GOES/XRS predictor availability have not been established by these checks.

## Scientific limits remain explicit

The original source-association uncertainties have not been made to disappear. The dataset retains unresolved outcomes and provisional no-event labels. This pipeline can run reproducible **exploratory** comparisons while those flags remain visible. `preflight_dataset(..., purpose="confirmatory")` rejects this candidate package. Final publication claims still require an agreed target definition, treatment of coverage and missing labels, and the planned independent/grouped evaluation. No code check can substitute for those scientific decisions.

## Temporal SHARP training

The first 72-hour GRU training experiment has completed with seeds 17, 29 and 43, alongside a logistic reference using the identical training cases. The primary interface is now [the training notebook](../notebooks/01_SHARP_72h_Training.ipynb). Its model and training functions are included in notebook cells, so it does not depend on repository Python imports. The matching script remains as tested support code. All 71 tests pass in the training environment.

The frozen blocks use 2010–2013 for fitting (25,586 eligible cases), January–June 2014 for early stopping (3,905), July–December 2014 for later probability calibration (2,831), January–June 2015 for later conformal calibration (4,168), and July 2015–2019 for policy development (13,142). A 24-hour reporting delay is assumed and outcome windows crossing each boundary are purged. Calibration/policy blocks have not been used for model selection or fitted by this stage.

Retrospective Cycle-25 evaluation has 35,846 eligible cases: raw GRU-ensemble Brier 0.077365 and AP 0.439603, versus logistic Brier 0.077341 and AP 0.429880. The small descriptive differences do not establish a significant advantage. Supplementary 2026 has 11,118 eligible cases. All three saved checkpoints reproduce their predictions exactly. No mapped region-component overlaps were found between the chosen eligible blocks, but the experiment is not claimed as independently validated AR/event-disjoint or prospective evaluation.

The [training receipts](../results/sharp72_training_20261002/) record configuration, histories, output hashes, verification and the original script run. Checkpoints and per-case predictions remain local. The working readable CSV had changed; integrity checks stopped the first launch. Training then used a separate snapshot extracted from the original verified archive. Working files were preserved.

## Probability calibration

[Notebook 02](../notebooks/02_SHARP_72h_Calibration.ipynb) has executed Platt-style logistic recalibration and increasing isotonic regression for both frozen backbones. The internal fit uses 1,900 July–November 2014 windows (76 positive); the December selection block has 764 windows (98 positive). Another 167 windows are purged at the internal boundary. Each prespecified calibrator is subsequently fitted on the full 2,831-window calibration block, while the chosen method is frozen before evaluation. **Raw probabilities won the earlier method-selection comparison for both models.**

For the GRU on retrospective Cycle 25, Brier scores are 0.077365 raw, 0.078457 Platt and 0.084477 isotonic. Corresponding log losses are 0.267044, 0.273704 and 0.444994. The fitted maps also fail to improve Brier on supplementary 2026. This result is specific to these earlier calibration data and candidate labels; it does not show that the raw probabilities are well calibrated or that calibration never helps. No method was selected from these evaluation scores.

The notebook records Brier skill against training climatology, log loss, ten-bin ECE, reliability with bin support, and jointly fitted calibration intercept/slope as descriptive diagnostics. Paired Brier differences have 1,000-replicate region-component bootstrap intervals and a seven-day UTC block sensitivity. These intervals are conditional, per-comparison summaries, not simultaneous inference or independent validation. At the end of Notebook 02, conformal/policy blocks remained reserved; Notebook 03 below uses only the conformal block for fitting.

All 76 tests pass. The 12-cell notebook executes successfully, all source/backbone/output hashes are checked, and scores/bin counts were independently cross-checked with scikit-learn and NumPy for all 12 model/method/evaluation combinations. [Receipts and numerical results](../results/sharp72_calibration_20261002/) preserve the negative findings as well as the successful execution. The full 153,366-case population remains represented, including 39,933 cases without supported inputs and without fabricated forecasts.

## Conformal uncertainty

[Notebook 03](../notebooks/03_SHARP_72h_Conformal_Uncertainty.ipynb) has executed pooled and class-conditional split conformal prediction using the frozen earlier-selected probabilities. Calibration uses 4,168 January–June 2015 windows (3,784 negative, 384 positive) from 95 mapped region components. Thresholds are exact finite-sample order statistics, with inclusive ties and an infinite-threshold fallback for insufficient class support. They are saved before later evaluation; 90% coverage is primary and 95% is a declared sensitivity. Neither level nor method is selected from evaluation results.

| GRU evaluation | Pooled overall coverage, 90% target | Pooled flare coverage, 90% target | Class-conditional flare coverage, 90% target | Class-conditional flare coverage, 95% target |
|---|---:|---:|---:|---:|
| Cycle 25, 2021–2025 | 86.4% | 12.6% | 73.5% | 91.8% |
| Partial 2026 | 90.9% | 12.1% | 43.5% | 71.9% |

The logistic reference shows similar failures. The fixed earlier thresholds do not retain requested class coverage on later data. Pooled coverage can conceal very poor flare coverage; class-conditional fitting improves flare inclusion in these comparisons but does not achieve the requested levels. The result does not identify the cause of the transfer failure or invalidate conformal theory under its assumptions. At the 90% target, the GRU class-conditional flare-coverage region-bootstrap intervals are 67.4–79.5% for Cycle 25 and 23.7–60.9% for 2026; these are conditional sensitivity intervals, not simultaneous guarantees.

Outputs preserve coverage numerators and denominators, annual results, empty/singleton/both-label fractions, conditional singleton errors, and 1,000-replicate region-component intervals with a seven-day UTC block sensitivity. Annual coverage plots use region-component intervals. All 153,366 candidate cases remain present; 39,933 missing-input cases have no sets. Cases before the conformal fitting cutoff have no backdated sets. Unknown-outcome cases can receive sets after the cutoff but never enter outcome metrics. A blank set code is distinct from an empty set.

All 14 notebook code cells executed successfully and all 84 repository tests pass. A separate calculation using sorted scores and rational-number ranks checked all 16 aggregate comparisons, annual coverage numerators, and saved set membership. Calibration-sample coverage was checked as an implementation sanity check, not held-out validation. The rendered tables and all three figures were inspected. [Compact results and verification](../results/sharp72_conformal_20261002/) are versioned; full population sets remain local in the timestamped output directory recorded by the verification receipt.

At the end of Notebook 03, policy-development outcomes remained unused for fitting and evaluation. Notebook 04 below uses that block to select a rolling update method. Previously inspected Cycle-25/2026 data remain retrospective. These candidate-label results do not establish operational readiness, independent event validation, a continuous-target quantile regression model, or an AIA fusion result.

## Rolling conformal replay

[Notebook 04](../notebooks/04_SHARP_72h_Rolling_Conformal.ipynb) has executed a daily replay with frozen backbones and probability maps. It compares the exact Notebook 03 class-conditional thresholds with 90-, 180- and 365-day class-conditional updates. Each 00:00 UTC update uses only earlier known-outcome cases whose full 72-hour follow-up and assumed 24-hour reporting delay have elapsed. Each class requires 50 windows from five mapped region components; otherwise that class is always included. These support guards are operational heuristics, not coverage guarantees. There are no scored 2020 cases, so the 2021 restart explicitly begins with insufficient recent support.

Method selection uses only the 13,142 eligible July 2015–2019 development cases (422 positive), minimizing the largest class coverage deficit at the primary 90% target, then mean set size and a fixed tie order. Both backbones select 180 days; this choice is retained at the 95% sensitivity target. The development block is now used and cannot be presented as untouched policy validation.

| Primary 90% target | Fixed flare coverage | Selected rolling flare coverage | Selected rolling both-label fraction |
|---|---:|---:|---:|
| GRU, 2021–2025 | 73.5% | 87.9% | 36.0% |
| GRU, partial 2026 | 43.5% | 78.3% | 37.4% |
| Logistic, 2021–2025 | 73.9% | 87.5% | 35.9% |
| Logistic, partial 2026 | 44.9% | 80.0% | 37.9% |

Recent calibration improves observed flare inclusion in these comparisons but does not achieve the requested 90% coverage. More cases retain both possible outcomes. GRU nonflare coverage changes from 84.0% to 89.2% in 2021–2025 and from 91.7% to 90.4% in 2026; the tradeoff is not a uniform improvement. At the 95% sensitivity target, selected GRU flare coverage is 92.0% and 88.8%, respectively, also below target.

All 153,366 candidate cases remain represented. The 39,933 missing-input cases receive no set; unknown outcomes never calibrate or enter metrics. Daily journals record support, guards, availability cutoffs, thresholds and support-case hashes. Region-component and seven-day UTC block bootstraps use 1,000 replicates for fixed/selected coverage and paired differences. They are conditional on the realized sequence of sets: they do not rerun adaptation or include model, calibration, label or selection uncertainty.

All 14 code cells executed without errors and all 92 repository tests passed. Independent calculations reproduced earlier selection, all 32 aggregate comparisons, annual coverage numerators and all 1,137,200 issued set memberships. A stratified sample of 71 update dates independently checked 1,704 threshold records and exact history membership; every journal record passed the latest-availability cutoff check. Rendered result tables and all three figures were inspected. [Results and verification](../results/sharp72_rolling_20261002/) record the exact run and hashes.

This is exploratory historical replay designed after earlier retrospective findings. Later labels enter calibration only after assumed maturity, so it is not label-free transfer from Cycle 24. The method updates rolling quantiles at fixed alpha; it does not implement an adaptive conformal inference error-feedback controller. Temporal dependence, shift, provisional labels and unverified historical delivery prevent an operational coverage guarantee. Notebook 05 below evaluates candidate forecast-state rules; future forecasts logged before outcomes remain subsequent work.

## Forecast-state decision experiment

[Notebook 05](../notebooks/05_SHARP_72h_Decision_Policy.ipynb) has executed eight declared policy variants at both parent conformal levels. It preserves the original predictors and the 180-day rolling sets. A Ledoit–Wolf covariance uses 25,586 training histories from 2010–2013 after the original signed-log transform. Limits use 3,905 January–June 2014 reference cases: the primary squared-distance cutoff is 379.555863 (99th percentile), and probability seed-SD cutoff is 0.03214736 (95th percentile). These are heuristics, not validated OOD or error guarantees; the 2014 block already informed early stopping.

The primary prototype assigns a candidate normal state to supported GRU singletons that pass feature-distance, disagreement and conflict checks. A candidate degraded state uses a supported logistic singleton when the GRU has both labels or excessive seed disagreement, without bypassing shared feature failures or opposite singleton decisions. Other cases abstain and have no issued probability or binary decision. Neither state certifies operational safety. Both modes require the same assembled SHARP histories; this is not a missing-modality fallback. Source-level instrument quality and actual delivery times remain unverified.

| Parent 90% target, matched known outcomes | 2021–2025 | Partial 2026 |
|---|---:|---:|
| Ungated GRU singleton error among issued decisions | 16.9% | 17.0% |
| Guarded GRU error among issued decisions | 12.8% | 15.7% |
| Guarded GRU share of all flare windows deferred | 53.3% | 53.5% |
| Guarded GRU share of all flare windows receiving a flare alert | 34.8% | 28.9% |
| Fallback-only errors / routed windows | 1,098 / 2,233 (49.2%) | 217 / 468 (46.4%) |

The fallback recovers some flare alerts but adds many false alerts and some false clears. Its observed error rate does not support promotion to an approved degraded mode. The guarded GRU's lower selective error also does not establish a satisfactory policy: it defers more than half of flare windows. No operational error tolerance or cost ratio has been agreed. All variants and both levels remain reported; the primary candidate has not been replaced after seeing the later results.

Performance tables keep inherited known-outcome roles. Full-population availability separately includes missing inputs, unknown outcomes and year-boundary cases. Every state and reason is preserved for all 153,366 cases; abstentions are never counted as correct predictions or quiet outcomes. Region-component and seven-day UTC block intervals use 1,000 resamples conditional on the realized decisions. They exclude repeated gate fitting, sequential adaptation, model, label and selection uncertainty. The development block and later years are reused evidence, not independent confirmation.

All 12 notebook code cells executed without errors and all 101 tests passed. Independent calculations reproduced all 113,433 feature distances and seed spreads, all gate order statistics, 2,453,856 state assignments across 16 policy/level combinations, 48 aggregate metric rows, 144 state-specific rows and 44 annual rows. Full-population state totals and null issued probabilities on abstentions were checked. The summary, rendered tables and all three figures were inspected. [Verification and compact outputs](../results/sharp72_policy_20261002/) record the frozen run; large per-case files and the covariance estimator remain local.
