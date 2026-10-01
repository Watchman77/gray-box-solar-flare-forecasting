# Region adjudication evidence — 1 October 2026

Read the [research report](../../docs/REGION_ADJUDICATION_20261001.md) for definitions, limitations and reproduction commands. These are executed candidate-dataset outputs, not final training labels or new model results.

- `source_receipt.json`: hashes, public URLs and acquisition status for 696 daily reports; two unavailable dates retained explicitly.
- `adjudication_summary.json` and `event_decisions.csv`: all 651 decisions and original/candidate region fields.
- `accepted_raw_region_evidence.csv`: raw source rows supporting the accepted region decisions; exact-XRA and group identifiers remain in the decision table.
- `source_time_review.csv`: 16 retained report rows with invalid/inconsistent timing.
- `outcome_build_summary.json`: versioned large-output hashes and readiness limits.
- `candidate_support.csv`: counts by horizon, scope and input cohort.
- `label_transitions.csv`: every transition from the original science-source candidate build, including unchanged outcomes. `<NA>` means unresolved.
- `decisions_by_year.csv`: decision counts by science peak year.
- `verification.json`: separate computational cross-check of all labels and event links.
- `artifact_manifest.json`: code and small-artifact hashes, local staging paths, and execution notes.

Original AIA experiments and the earlier source-reconciliation artifacts remain unchanged. Shared source lineage means the daily reports are not an independent set of physical observations.
