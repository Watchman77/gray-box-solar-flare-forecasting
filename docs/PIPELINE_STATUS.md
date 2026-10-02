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
