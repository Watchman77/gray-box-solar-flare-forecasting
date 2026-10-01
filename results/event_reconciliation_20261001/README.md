# Event reconciliation evidence

These results concern event-source lineage, provisional region recovery and a bounded queue of source-supported **48-hour** positive-label corrections. They are not model metrics or a final 72-hour dataset.

| Evidence | Meaning |
|---|---|
| `artifact_manifest.json` | Source generations, code hashes, local output locations and validation scope. |
| `reconciliation_summary.json` | Reference sources, matching-rule outcomes, region-fill sensitivity and exact legacy replay. |
| `label_lineage_by_year_cohort.csv` | Original support and disagreement counts by issue year and accepted input cohort. |
| `reference_source_profile.csv` | Reference-row counts and timing limitations; duplicates are not independent observations. |
| `event_reconciliation_by_year.csv` | Matching status for science M/X entries by peak year. |
| `event_review_queue.csv` | 651 events with missing/conflicting regions or proposed fills. |
| `case_label_review_queue.csv` | 1,033 original negative windows selected for direct source review. |
| `computational_cross_check.json` | All-row and direct-reference verification of reconciliation outputs. |
| `swpc_spot_checks.json`, `swpc_spot_source_receipt.json` | Initial three-report checks; superseded in scope by the complete bounded audit below. |
| `swpc_full_source_receipt.json` | URLs and hashes for all 64 archived daily reports. |
| `swpc_full_audit_summary.json`, `swpc_event_checks.csv` | Exact-peak XRA evidence for the 90-event queue, with one regionless XRA event kept unresolved at this stage. |
| `swpc_raw_line_verification.json` | Separate raw-line replay and the explicitly recorded optical/XRA event-bin resolution for the last window. |
| `source_supported_corrections_48h.csv` | All 1,033 supported correction proposals, including 798 matched cases; none applied to upstream labels. |

The difference between 797 directly supported matched cases in the full-audit summary and 798 in the final verification is the documented same-event-bin optical region association, not a hidden change of matching tolerance. Event/case review queues retain the earlier diagnostic state for provenance.

Raw annual and daily sources and large per-case outputs remain Git-ignored. A Git clone alone is insufficient to rerun the full build. Read [the reconciliation report](../../docs/EVENT_RECONCILIATION_20261001.md) for reconstruction requirements and limitations.
