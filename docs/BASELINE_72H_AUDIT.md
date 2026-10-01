# Historical 72-hour baseline reconciliation

Executed 1 October 2026. **The recovered headline probability arrays reproduce TSS 0.593565 and PR-AUC (average precision/AP) 0.430048.** This verifies historical predictions and cohort support, not model retraining or operational readiness.

## Sources and reproducibility

- Original repository snapshot: `Watchman77/solar-flare-multihorizon`, commit `f05ff192356a79ec7fbbc0ba32dbdc289085a23a`.
- Notebook: `72H_CLEAN_Journal_Revision_Main_Transformer_DLSTM.ipynb`, SHA-256 `41138dda5d403b66e60c4afc1fad6999e655568362d1434e15a5355fc0a8f392`.
- Source CSV: `HMI_AR_2010_2025_ML_READY_16_LABELED_MX3D.csv`, SHA-256 `c3cd3d14588410acec2357b2e9e44ae5045a01863e8d261e202078cf203352d3`. The two local copies in Downloads/CSVfiles and Downloads/csv are byte-identical.
- Drive inventory located 45 files in `Journal_Revision_72h_PI_DLSTM_CLEAN` and 29 files plus one subfolder in `Journal_Revision_72h_CLEAN_MAIN`. Eight selected arrays/tables were retrieved read-only, with original names, IDs, timestamps and content hashes retained locally. File discovery alone is not content verification; only those eight were downloaded.
- Committable summaries: [cohort audit](../results/baseline_72h_20261001/cohort_audit.json), [prediction reconciliation](../results/baseline_72h_20261001/prediction_reconciliation.json), [download hashes](../results/baseline_72h_20261001/download_manifest.json).
- Local detailed case manifests and original recovered files remain under Git-ignored `outputs/legacy_72h_audit_20261001/`.

## Reproduced support and metrics

The source has 22,355 rows and 21 columns. The original complete-case rule removes 225 rows, leaving 22,130. Seven-row within-region histories produce **9,438 sequences, 652 positives**. There are no duplicate region/time keys or nonfinite retained features. Independent vectorized counts agree with the window-by-window audit.

| Variant | Train rows / positives | Validation rows / positives | Test rows / positives |
|---|---:|---:|---:|
| Headline: integer truncation, original default time sort | 6,606 / 339 | 1,415 / 163 | 1,417 / 150 |
| Supplement: rounded counts, second default time sort | 6,607 / 339 | 1,416 / 164 | 1,415 / 149 |
| Diagnostic: integer truncation, explicit deterministic ties | 6,606 / 339 | 1,415 / 164 | 1,417 / 149 |

Both historical count variants reproduce the recorded notebook outputs. The third is a sensitivity diagnostic, not a replacement baseline. Equal-time ordering changes membership even when role sizes remain constant. The published Table 5.4 support corresponds to the rounded variant, while the headline Table 6.3 confusion counts and recovered probabilities correspond to the 1,417-case variant. Preserve this distinction when using the paper as a comparator.

Recovered simple-average ensemble: threshold **0.707**, 1,417 test cases, 150 positives, prevalence **10.5857%**; TN **997**, FP **270**, FN **29**, TP **121**. Independently recomputed TSS **0.5935648514**, AP **0.4300479299**, ROC-AUC **0.8610155222** match the saved result table within `1e-8`. The threshold reproduces the saved binary predictions. The reconstructed validation and test label order matches the saved arrays, and the recovered May and August headline test-label/probability arrays are identical. Labels alone do not uniquely prove historical case identity; original row IDs/times are still preferable.

## Findings that govern the new baseline

| Finding | Evidence and interpretation | Action |
|---|---|---|
| High: histories can cross unrelated epochs | 708 histories exceed the nominal 144-hour span; 49 have an individual gap over 48 hours. Region ID 11114 links 2010 and 2025, producing two histories spanning 128,568 hours. These are directly observed source/timing anomalies; their upstream cause and effect on skill have not been established. | Recover HARP/NOAA associations and segment valid region episodes; set a justified continuity rule in a new version. Do not automatically delete every longer history or change old results. |
| High: row-fraction splits are not operational information cutoffs | Headline train/validation and validation/test each share four region IDs and contain equal issue times on both sides. Eleven training and ten validation windows end after the next block's first issue time under a 72-hour, zero-reporting-delay assumption. | Use timestamp blocks, label-maturity cutoffs and explicit purging; evaluate region/event-disjoint variants separately. |
| High: historical availability and labels remain unverified | CSV timestamps are naive; original event IDs, catalogue coverage, reporting delay and source delivery metadata are absent from these fields. | Reconstruct the source/label contract before operational claims. Reproducing an embedded label does not validate its physical meaning. |
| Medium: tie order affects reproducibility | Deterministic tie sorting changes validation positives 163→164 and test positives 150→149 without changing split sizes. | Persist case IDs and exact role membership. Do not rely on implicit array sorting across environments. |

The headline training interval reaches March 2023; it is not a pure Cycle-24 training experiment. Keep it separate from the AIA project's Cycle-24-to-Cycle-25 design.

## Rerun

Use Python 3.12 with [requirements-audit.txt](../requirements-audit.txt). From the repository root, supply the original source and notebook paths:

```sh
python scripts/audit_legacy_72h.py \
  --source /path/to/HMI_AR_2010_2025_ML_READY_16_LABELED_MX3D.csv \
  --compare-copy /path/to/second/HMI_AR_2010_2025_ML_READY_16_LABELED_MX3D.csv \
  --notebook /path/to/72H_CLEAN_Journal_Revision_Main_Transformer_DLSTM.ipynb \
  --output-dir outputs/legacy_72h_audit_20261001

python scripts/reconcile_legacy_predictions.py \
  --artifact-dir outputs/legacy_72h_audit_20261001/recovered_predictions \
  --case-manifest outputs/legacy_72h_audit_20261001/candidate_case_roles.csv \
  --output outputs/legacy_72h_audit_20261001/prediction_reconciliation.json

python -m unittest discover -s tests -v
```

No source files, original predictions or upstream repositories were modified. No model was trained. The next step is region/time/label reconstruction and an immutable operational case manifest; fitting calibration to this historical test would not resolve those issues.
