# Candidate dataset build — 1 October 2026

**Executed:** a versioned HARP-record inventory, linked verified AIA–SHARP inputs, and provisional flare-outcome tables. **Not completed:** final training labels, an operational availability contract, a 72-hour split, or a trained Gray-Box model.

## What now exists

| Item | Verified build output |
|---|---:|
| Native candidate records retained | 153,366 |
| Candidates linked to verified three-slot AIA–SHARP inputs | 113,433 |
| Candidates retained without that accepted input package | 39,933 |
| Candidates without a region-component mapping | 0 |
| NOAA annual event files staged | 17, covering 2010–2026 |
| Events in those files | 44,016 |
| M/X entries in those files | 3,412 |
| Provisional case/horizon rows | 306,732: every case at 48 and 72 hours |

The first issue is **21 May 2010, 03:47:26 UTC**; the last is **17 August 2026, 23:11:23 UTC**. The latest 72-hour window ends **20 August 2026, 23:11:23 UTC**. The staged catalogue metadata extends to 29 September; conservatively using midnight on that date places every required window inside its nominal span. This does not prove continuous flare-detection coverage.

SHARP has already been extended upstream for the accepted AIA cohorts. Its 16-feature arrays have shape `(cases, 3, 16)` with histories at native issue time minus 288, 192 and 96 minutes. This build verifies their hashes and row mappings and references them directly. AIA image URIs and accepted history membership are reused; this run does not re-read every image pixel. These histories differ from the ASR model's seven daily states; they are not interchangeable inputs to that checkpoint.

| Existing input cohort | Matched cases |
|---|---:|
| Cycle 24 | 55,871 |
| Primary Cycle 25 | 45,403 |
| Supplementary 2026 | 12,159 |

These are inherited input-cohort names, not newly assigned training/test roles. The source is a curated candidate universe, not a census of every observable solar region. The 39,933 unmatched cases include protocol exclusions and unassembled inputs, not just outages. For example, 2020 has 2,089 candidates but no accepted input package in these three cohorts. No operational availability percentage is inferred from these counts.

## Findings that affect scientific validity

1. **Region association changes the target.** There are 37,925 records listing multiple NOAA regions. The outcome builder separately records (a) flares matched to the source's primary NOAA ID and (b) flares matched to any NOAA ID listed on that particular HARP record. These conventions change recorded M/X presence in 4,329 of the 72-hour cases. The component's eventual membership is not used for event assignment. Choose the target after verifying what region the SHARP parameters and AIA crop represent; do not choose it because one convention scores better.
2. **Some M/X events lack region IDs.** Of 3,412 catalogue M/X entries, 412 have no region association. The missingness is uneven: 61 of 65 M/X entries in the 2017 file lack region IDs. An unassociated M/X entry within a window masks an otherwise-negative candidate for every target, as a conservative screening rule. It does not imply that all those regions flared. Recorded positives remain positive. Unmasked zeros are still provisional pending observing-coverage verification.
3. **Forecast grain needs a decision.** There are 988 HARP records in 494 groups sharing the same component and issue time, including 895 matched-input records. No rows were dropped or averaged. Component-level forecasts require an explicit consolidation rule; HARP-level forecasts require that distinct target unit to be declared and dependence handled during evaluation.
4. **A new event source is not interchangeable with the old labels.** NOAA reports that science-calibrated classes for GOES-8 through GOES-15 can differ from earlier operational flare lists. Its event table is indexed by peak time and also supplies start time. We explicitly use reported start time for these provisional labels. See the [NOAA product guide](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES_Flare_Report_ReadMe.pdf). The 48-hour reconstruction isolates source/convention differences before the horizon is changed. Disagreements remain unadjudicated and cannot all be attributed to calibration.
5. **GOES outcomes and GOES predictors are separate products.** These files supply future-event candidates for labels. No continuous GOES/XRS predictor has been assembled here. Input quality, satellite transitions, observation cutoffs and historical availability remain work owned by the upstream AIA/GOES pipeline. Definitive products and later reprocessing do not prove that the same values were available operationally at issuance.

The first four findings are high-priority threats to target validity or evaluation independence. The fifth limits operational claims. Counts and computational behavior are verified; their scientific resolution is outstanding.

## Provisional outcome contract

- **Timing:** event start in `(issue, issue + horizon]`; add physical elapsed hours in TAI, then convert to UTC using the pinned IERS table. Preserve native timestamps. Minute-resolution reported starts do not provide sub-minute event-time accuracy.
- **Class:** combined M or X according to the pinned science catalogue. Do not silently replace an operational-class target with this version.
- **Association:** store both primary-NOAA and all-currently-listed-NOAA scopes with matched event IDs. Four-digit source region suffixes are expanded only within the checked NOAA 10000–19999 epoch for 2010–2026.
- **Missingness:** unknown-region events mask otherwise-negative candidates. Windows beyond nominal source bounds remain unlabeled. Missing event end times do not invalidate this start-time target; all staged events have recorded starts and peaks.
- **Readiness:** every output has `training_ready=False`; all new roles are `unassigned`. Preserve the original 48-hour labels and roles. No final 72-hour split is inferred from them.

Unique source IDs do not establish unique physical flares: closely related or separately reported detections still need reconciliation before event-disjoint evaluation. Catalogue metadata and the absence of events do not establish negative truth. The current native issuance grid is retained for inventory purposes; the proposed daily 12:00 UTC experiment has not been constructed.

## Validation and evidence

All accepted case IDs, region identifiers, original labels and historical UTC times match the pinned upstream records. Nine upstream index/ledger/tensor hashes were checked; arrays have the declared shape and finite values. All canonical IDs and HARP/native-time keys are unique.

Twenty focused tests pass, including eleven new tests for leap-second transitions, physical 72-hour duration, `(t,end]` inclusion, uncertain negatives, multiple region scopes, missing follow-up, invalid region epochs and duplicate keys. A separate computational cross-check independently implements time conversion for this bounded epoch, checks all 306,732 outcome rows, and compares direct event filtering for 1,596 deterministically sampled rows. This checks implementation; it is not independent scientific validation of the source catalogue.

Compact evidence is in [results/dataset_construction_20261001](../results/dataset_construction_20261001/): source hashes, build summaries, yearly inventory, provisional support, event missingness and the cross-check receipt. Large per-case files are Git-ignored. No model performance is reported by this build.

## Reproduction

Use an environment with [requirements-audit.txt](../requirements-audit.txt). The commands below assume a fresh output location and read access to the pinned upstream AIA files. Builders reject existing output directories. The earlier local inventory `v1` was preserved; `v2` adds component multiplicity and source-provenance fields without changing its case population.

```sh
export AIA_INPUT_ROOT="/path/to/AIA Solar flare Project"
mkdir -p data/raw/gray_box_inventory_v1
gcloud storage cp 'gs://suryabench-sharp-pipeline-bamidele/metadata/curated_sharp_suryabench_true96min_48h_AR_SPECIFIC_2010_2026.csv#1790079135531166' data/raw/gray_box_inventory_v1/canonical_1790079135531166.csv
python scripts/prepare_dataset_sources.py --aia-root "$AIA_INPUT_ROOT" --output-dir data/raw/gray_box_inventory_v1 --expected-receipt results/dataset_construction_20261001/outcome_source_receipt.json
python scripts/build_dataset_inventory.py --aia-root "$AIA_INPUT_ROOT" --source-dir data/raw/gray_box_inventory_v1 --output-dir data/processed/gray_box_inventory_v2
python scripts/build_candidate_outcomes.py --source-dir data/raw/gray_box_inventory_v1 --inventory-dir data/processed/gray_box_inventory_v2 --output-dir data/processed/gray_box_outcomes_v1
python scripts/verify_candidate_dataset.py --inventory-dir data/processed/gray_box_inventory_v2 --outcome-dir data/processed/gray_box_outcomes_v1 --output outputs/dataset_verification_rerun.json
python -m unittest discover -s tests -v
```

The canonical object generation and all staged NOAA hashes are recorded. NOAA annual files can be revised; `--expected-receipt` rejects changed contents rather than silently reproducing a different dataset. Recover archived source files if necessary. Upstream SHARP indices, ledgers and tensors must match their accepted receipt; their paths are recorded relative to the explicitly supplied upstream root.

## Next executable milestone

Reconcile event identity, missing NOAA associations and class conventions against the existing HEK/SWPC evidence, alongside the upstream continuous-XRS quality/coverage audit. Then freeze the forecast unit, issue cadence, historical input-availability assumptions and final 72-hour labels. Generate chronologically ordered development, calibration, policy and evaluation manifests with matured outcomes and required region/event separation before fitting any model.

Already-inspected 2021–2026 AIA results support retrospective analysis; they do not become untouched confirmation merely by creating a new dataset file. A later uninspected period or logged future forecasts is needed for that claim.
