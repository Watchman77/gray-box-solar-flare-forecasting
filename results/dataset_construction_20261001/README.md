# Executed candidate dataset evidence

These files describe the 1 October 2026 inventory and provisional-label build. They are not training labels, model predictions, or validation of an operational forecasting system.

| File | Content |
|---|---|
| `artifact_manifest.json` | Canonical object generation, local output locations, scope and validation status. |
| `inventory_build_summary.json` | Case counts, source hashes, component multiplicity and timing checks. |
| `yearly_inventory.csv` | Candidate and matched-input counts by UTC issue year. |
| `outcome_source_receipt.json` | Exact hashes and public URLs for 17 NOAA annual files plus metadata. |
| `outcomes_build_summary.json` | Provisional label rules, nominal bounds, support and output hashes. |
| `candidate_support.csv` | Provisional positive/negative/unresolved counts by horizon, input cohort and association scope. |
| `event_source_profile.csv` | Event counts and missing region associations by peak-time year. |
| `computational_cross_check.json` | All-row invariants and separate direct-filter checks on 1,596 sampled rows. |

The source-retrieval timestamp records staging, not when historical measurements became available to a forecaster. Reused files retain their exact upstream hashes. The M/X event count covers the full downloaded 2010–September 2026 source partitions, not only events in the candidate forecast windows.

The `different_from_upstream_48h_among_resolved` field compares the reconstructed 48-hour candidate with the preserved upstream 48-hour label only where the candidate is non-missing. It is blank for 72 hours. It is not a count of proven mistakes: source, event timing and association conventions still need adjudication. Candidate negatives are not certified non-flare outcomes.

See [the full build note](../../docs/DATASET_BUILD_20261001.md) for interpretation, limitations and reproduction commands.
