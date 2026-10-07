# Multi-horizon manuscript evidence

This directory contains the descriptive evidence build for the 3-hour, 24-hour
and 72-hour comparison.

- `calibration_reliability.csv` and `figures/calibration_reliability.png` show
  raw and selected-calibrator reliability displays for the SHARP-only 3-hour
  and 24-hour runs.
- `multihorizon_comparison.csv` combines the short-horizon SHARP rows with the
  frozen 72-hour fallback-v3 summaries. The `scope` and `note` columns are
  required when interpreting the table: the 72-hour rows are multimodal and
  are not like-for-like replacements for the short-horizon SHARP rows.
- `manuscript_evidence_receipt.json` records the claim boundary.

The short-horizon labels remain provisional candidate labels. The evidence
build does not refit the operational policy or add AIA/GOES predictors.
