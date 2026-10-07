# 72 h baseline suite v2 — archived executed record

This folder archives the already-executed Notebook 31 manuscript comparator. **No experiment was rerun to create this repository record.**

The suite was added after the later Cycle-25 and supplementary-2026 outcomes had been seen. Its inferential status is therefore post-hoc manuscript comparison. The simple logistic model itself was fit only on the frozen training role, and its decision threshold was selected only on model_validation. Frozen SHARP/AIA/fusion thresholds were reused unchanged. TSS values in `baseline_metrics.csv` use those model-specific validation-chosen/frozen thresholds; `tss_0.5` is retained separately and must not be substituted for the primary TSS.

The suite did not refit any neural model and did not change the v3 policy or the Phase-H physical monitor.
