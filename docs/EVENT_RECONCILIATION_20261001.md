# Event and label reconciliation — 1 October 2026

**The audit produced source-supported positive-label correction proposals for 1,033 existing 48-hour windows, including 798 matched AIA–SHARP cases.** The proposals are saved separately. Original labels, predictions, training jobs and published/reported metrics are unchanged. This is not yet a complete repaired dataset or a final 72-hour target.

## Established label lineage

The earlier HEK object, generation `1778961084955762`, contains 2,185 M/X entries. Combining it with the pinned 2026 SWPC extension reproduces **all 153,366 original labels exactly** when using the original native-clock interpretation. Replaying with physically converted UTC changes one unmatched-input case, dated 14 July 2024. It changes none of the 113,433 accepted input cases.

This separates computational reproducibility from source completeness. The label-generating calculation can reproduce the old result while its event list omits relevant observations. The September HEK object has 3,462 rows and substantially different event support, but the extension audit explicitly preserved historical labels. Duplicate detections and provider differences mean those row totals must not be interpreted as distinct physical-flare counts.

## Positive evidence missing from the earlier labels

| Issue year | Matched cases requiring the source-supported 0 → 1 proposal |
|---|---:|
| 2021 | 35 |
| 2022 | 316 |
| 2023 | 446 |
| 2024 | 1 |
| **Total** | **798** |

Another 235 affected cases are outside the accepted matched-input packages. All 1,033 windows were originally negative, yet have same-region M/X evidence in their future 48-hour intervals. They link to 90 science-catalogue events; overlapping forecast windows are not independent flare events.

The first comparison used exact peak matches between the science catalogue and SolarSoft/HER annual records, requiring the same region and an M/X annual classification. A separate implementation checked the event linkage and future interval for every flagged window. We then obtained **64 archived NOAA daily reports** and checked the entire event queue:

- **1,032 windows / 797 matched cases:** an exact-peak M/X X-ray report names the forecast region and starts inside its future window.
- **One window / one matched case:** the X-ray report has no region number. The same day's SWPC event group `3510` links it to an optical flare report naming NOAA 13784. This separate association is retained explicitly; the source X-ray field remains missing. NOAA describes the event number as a forecaster-defined grouping of related reports in its [event-report specification](https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_event_reports/docs/SolarEventReport.pdf).

Examples:

| Forecast issue (UTC) | NOAA region | Original label | Archived positive evidence inside 48 h |
|---|---:|---:|---|
| 16 January 2022, 17:47:23 | 12929 | 0 | M1.5 starts 18 January at 17:01; peak 17:44. [Daily report](https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_event_reports/2022/01/20220118events.txt). |
| 7 January 2023, 12:23:23 | 13181 | 0 | M2.1 starts 9 January at 08:45; peak 09:01. [Daily report](https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_event_reports/2023/01/20230109events.txt). |
| 14 August 2024, 14:11:23 | 13784 | 0 | M5.3 starts at 15:39; its region is supported through the same-day optical/X-ray event group. [Daily report](https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_event_reports/2024/08/20240814events.txt). |

The correction file is [source_supported_corrections_48h.csv](../results/event_reconciliation_20261001/source_supported_corrections_48h.csv). Each proposed change records its case ID, input cohort, original label, supported label, evidence type and exact daily-report line references. None has been applied to a frozen upstream experiment.

This is a bounded audit of missing positive evidence. It does not identify every possible label defect, validate all negatives, or determine how much any model metric changes. HEK, HER and SWPC can share underlying observations; these checks improve source traceability rather than supplying independent physical measurements. The cause of the older catalogue's omissions still requires its original extraction/query history. No conclusion about the user's 2024-versus-2025 distribution-shift hypothesis follows from this audit alone.

## Missing-region reconciliation for the new science target

The reconciliation examines 40,477 reference rows from the older and September HEK files, the 2026 extension, and sixteen annual HER files. It preserves duplicate reference evidence rather than counting duplicate rows as independent votes.

For a proposed region fill, the science event must have a unique peak across all science classes; a reference must match its exact start and peak; and all known reference-region IDs at that peak must agree. References with inconsistent timing or assumed annual clock rollover are excluded from matching. Existing science-region assignments are never overwritten.

| Originally unassociated science M/X events | Count |
|---|---:|
| Exact-start/exact-peak region-fill proposals | 113 |
| Peak-only proposals, not applied | 71 |
| Conflicting reference regions | 4 |
| No exact reference with a usable region | 224 |
| **Total originally missing a region** | **412** |

There are also 239 science M/X events with an existing region that conflicts with at least one exact-peak reference. The [651-event review queue](../results/event_reconciliation_20261001/event_review_queue.csv) contains those conflicts and all 412 originally unassociated events, including the 113 proposals. These totals cover the downloaded source period; not every event affects a forecast case.

Applying **only the 113 proposals as a sensitivity analysis** reduces unresolved primary-region 72-hour candidates from 19,332 to 13,916. It yields 104 additional candidate positives and 5,312 additional candidate negatives. These are provisional source-association consequences, not independently validated outcomes. All previously resolved candidates are preserved. The original science classes and event times are unchanged.

The source audit additionally records 21 timing-problem rows, 575 assumed rollover rows, and two four-digit region IDs inside the older HEK file's otherwise five-digit field. Recorded IDs remain available for exact legacy replay; normalized IDs are separately marked. Annual lists lack observed 31 December entries, and the supplied 2025 list stops in April. The inspected [SolarSoft generator](https://soho.nascom.nasa.gov/solarsoft/gen/idl/synoptic/goes/goes_make_yearly_eventlist.pro) sets its year-end query to 31 December without an explicit end-of-day time. This supports a boundary-risk interpretation, not a declaration of annual completeness.

Science and historical operational class conventions remain distinct. For example, the 14 February 2011 reference check supports NOAA 11158 at the same peak, but the archived report calls it C7.0 and the science catalogue calls it M1.2. Region agreement does not establish class agreement. The [NOAA science-product guide](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES_Flare_Report_ReadMe.pdf) documents differences from older classifications. No class rescaling was inferred or applied in this audit.

## Research consequence and next work

1. **Complete the event adjudication and declare the target.** Resolve the remaining region conflicts and missing associations, decide primary-NOAA versus HARP-patch scope from the actual image/feature footprint, and state whether the target uses science-calibrated or historical operational classes. A convenient majority vote or better model score is not a valid source decision.
2. **Build a complete new label version.** Use reconciled events and documented observing coverage; do not patch only the detected positives and call all remaining zeros verified. Keep 48-hour source diagnostics distinct from the new 72-hour experiment.
3. **Re-evaluate appropriately.** Existing frozen AIA predictions can later be rescored against a declared corrected 48-hour label version, with original scores retained. Report the full evaluation population, support, TSS and PR-AUC/AP together. No revised metrics are available now. Any model refit or calibration update needs its own earlier-data rules and version; retrospective corrections do not create an untouched test set.
4. **Freeze Gray-Box splits only after those decisions.** Preserve original input cohort names as provenance, not as automatic 72-hour experimental roles. Negative-window completeness, historical input availability and the operational issuance schedule remain open.

This finding should be considered when interpreting the existing cross-cycle evaluation. It does not authorize changing another chat's active training run or overwriting its evidence. A local handoff note is available in [AIA_LABEL_REVIEW_HANDOFF.md](AIA_LABEL_REVIEW_HANDOFF.md).

## Validation and reproduction

All **36 focused tests pass**. Separate computational checks verify every proposed region fill, all 306,732 candidate case/horizon rows, the original 153,366-label replay and every flagged case/event link. A second parser checks the raw daily-report lines for all 1,032 direct windows; the final event-bin association is pinned and checked separately. All 64 daily-file hashes are verified. These are implementation and provenance checks, not final scientific clearance of the full dataset.

The full local outputs remain under Git-ignored `data/processed/event_reconciliation_v1`, `data/processed/swpc_label_audit_v1` and `data/processed/swpc_label_verification_v1`. Small evidence tables, source hashes, correction proposals and verification receipts are tracked in [results/event_reconciliation_20261001](../results/event_reconciliation_20261001/).

In an environment with `requirements-audit.txt`, use fresh output directories:

```sh
python scripts/reconcile_event_sources.py --aia-root "$AIA_INPUT_ROOT" --legacy-hek /path/to/hek_mx_ar_flares_2010_2026.csv --source-dir data/raw/event_reconciliation_v1 --inventory-dir data/processed/gray_box_inventory_v2 --previous-outcomes-dir data/processed/gray_box_outcomes_v1 --output-dir data/processed/event_reconciliation_v1
python scripts/verify_event_reconciliation.py --reconciliation-dir data/processed/event_reconciliation_v1 --previous-outcomes-dir data/processed/gray_box_outcomes_v1 --output outputs/event_reconciliation_cross_check.json
python scripts/audit_swpc_label_disagreements.py --reconciliation-dir data/processed/event_reconciliation_v1 --source-dir data/raw/event_reconciliation_v1/swpc_label_audit --reuse-dir data/raw/event_reconciliation_v1/swpc_daily_spot_checks --output-dir data/processed/swpc_label_audit_v1
python scripts/verify_swpc_label_audit.py --source-dir data/raw/event_reconciliation_v1/swpc_label_audit --audit-dir data/processed/swpc_label_audit_v1 --output-dir data/processed/swpc_label_verification_v1
python -m unittest discover -s tests -v
```

The source directory must contain the pinned September object `hek_sep_1790075866552766.csv`. Its generation and hash, the older HEK object's identity, and the extension's identity are in the artifact manifest. Exact daily-source reproduction requires the saved files and matching receipt; new downloads may reflect later archive corrections and must be treated as a new source version. Source staging reads the upstream project without modifying it. The review queue and evidence exports are derived from the named output tables; no data is inferred from model predictions.
