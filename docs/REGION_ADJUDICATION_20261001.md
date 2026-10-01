# Daily-source region adjudication and candidate labels v2

The 651-event review queue has been checked against archived NOAA daily reports. A separate candidate label version now covers all 153,366 forecast cases at both 48 and 72 hours. This is an executed source-association and label-construction milestone; it does not establish training-ready truth, continuous event detection, historical operational availability, or new model performance.

## Result

| Decision | Events | Treatment in candidate v2 |
|---|---:|---|
| Missing science region supported by direct daily XRA evidence | 55 | Fill the candidate region; preserve the original missing field. |
| Existing disputed region supported by the daily event group | 193 | Retain the original science region; retain the competing references in the audit. |
| Originally missing region still unresolved | 357 | Keep the candidate region unknown. |
| Existing disputed region without sufficient consistent daily support | 46 | Quarantine the candidate association as unknown; preserve the original region. |
| Total reviewed | 651 | 248 accepted associations, 403 unresolved. |

Of the 248 accepted associations, 247 have direct XRA region evidence and one retains an existing region using an overlapping optical flare in the same daily event group. No existing science region was replaced with a different region. “Quarantined” means insufficient support under this rule, not proof that the original assignment was wrong.

The 403 unresolved entries comprise 250 without an exact eligible XRA peak, 134 without qualified region evidence, 13 with conflicting daily-group regions, three with multiple matching daily groups, two with reference/daily disagreement, and one affected by unavailable daily reports. These counts describe the entire review catalogue, including events outside the forecast-case time range.

The source search requested 698 dates, covering the science start day, peak day and day before the peak. It obtained **696 reports, 3,053,251 bytes**, containing 30,553 parsed report rows. The NOAA archive returned 404 for **30 and 31 December 2025**; its [December directory](https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_event_reports/2025/12/) lists files only through 29 December. The affected science event is `noaa-science-v1-0-1:202512311311`. This source gap remains recorded rather than interpreted as no flares.

## Reproducible decision rule

The implemented policy is `swpc_exact_peak_region_v1`. NOAA's [report specification](https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_event_reports/docs/SolarEventReport.pdf) defines the daily event grouping and clock conventions. The additional acceptance restrictions below are research choices, not guarantees supplied by NOAA.

1. Match a science peak to exactly timed, quality-5 XRA reports at the same UTC minute. Do not accept a nearest-time match. Require the science peak to be unique across the full science catalogue, including lower classes.
2. Require exactly one matching **date plus SWPC event-number** group. Event numbers are not globally unique. Preserve all reports, including the representative-report `+` flag; a selected row does not erase contradictory rows.
3. Require the known region fields across that group to agree. Accept a directly assigned matching XRA region, or an optical FLA region with quality 3–5 and an exactly timed interval overlapping the matched XRA interval. Radio/CME/SXI assignments alone are insufficient for a fill, but a conflicting group region prevents acceptance.
4. Fill a missing science region only if the available exact-peak reference regions also do not contradict the daily candidate. For an existing disputed region, retain it only when daily evidence agrees. Preserve all original values and disagreements.
5. Require all requested dates for an accepted association to be present and checksum-verified. Otherwise leave it unresolved.

The parser preserves raw row identifiers and text, including bounded/uncertain clocks. It implements the documented next-day rollover from an exact begin clock. It also handles the archived expanded CME layout and its occasional full five-digit NOAA IDs. Sixteen report rows have invalid or inconsistent timing: ten FLA, three XRA, two RNS and one RBR. They are retained in the evidence and excluded from exact timing support. One XRA end clock is literally `0075`; it was not silently corrected.

## Class definitions remain separate

The science catalogue's reported start times and classes remain unchanged. **Sixteen of the 55 newly assigned science M/X events have only lower-class matching historical XRA reports.** For example, science event `201102141345` is M1.2 with peak 14:27 UTC; the daily report gives C7.0 at that peak and region 1158, normalized to 11158. This supports a region association under the declared rule, not historical M-class status.

The earlier reference-only analysis proposed 113 fills. Only 12 of those meet this daily-source policy; the other 43 accepted fills come from events outside that earlier strict-proposal set. The earlier outputs remain preserved as a sensitivity version. No majority vote or union of the proposed fills was applied.

Before final evaluation, freeze a named science-class target and, where adequately supported, a separate historical operational-class sensitivity target. Do not blend these conventions or attribute their differences to solar-cycle physics without a controlled comparison. These findings alone do not explain the reported 2024-versus-2025 model behaviour.

## Candidate label version

The new target identifier is `candidate_mx_start_science_swpc_regions_v2`. The source is still the same 44,016-event NOAA science catalogue, with 3,412 M/X entries. Only candidate region association fields change. The original 412 missing M/X regions become **403 unknown candidate regions: 412 − 55 + 46**.

Outcome membership remains `(issue_time, issue_time + horizon]`, using the science **start time**. Physical elapsed 48/72 hours are calculated in TAI before converting endpoints to UTC. Both the source-primary NOAA target and the union of NOAA IDs listed on the current HARP record are retained separately. Future component membership is not used.

| Horizon | Association scope | Candidate positive | Provisional zero | Unresolved | Total cases |
|---|---|---:|---:|---:|---:|
| 48 h | Primary NOAA | 8,095 | 128,953 | 16,318 | 153,366 |
| 48 h | Any NOAA listed on current HARP | 11,506 | 126,525 | 15,335 | 153,366 |
| 72 h | Primary NOAA | 10,515 | 121,080 | 21,771 | 153,366 |
| 72 h | Any NOAA listed on current HARP | 14,759 | 118,331 | 20,276 | 153,366 |

The 113,433 cases with matched upstream AIA–SHARP inputs have 6,017 positive, 95,112 provisional-zero and 12,304 unresolved primary-NOAA outcomes at 48 hours; at 72 hours these counts are 7,781, 89,190 and 16,462. These are candidate support counts, not final sample sizes or validated flare prevalence.

Unknown-region M/X events conservatively mask otherwise-negative windows for every target. They do not mean that every target flared. Despite the small net decrease in unknown event count, unresolved primary-NOAA windows increase compared with the original science-source build: 14,471 to 16,318 at 48 hours, and 19,332 to 21,771 at 72 hours. The timing of the newly quarantined associations matters more than their count. No definite 0-to-1 or 1-to-0 flips occur between those two science versions: changes pass through an unresolved state. The full transition table records all cases and scopes.

This build is separate from the previous **1,033 source-supported corrections to inherited AIA 48-hour labels**. Those proposals concern missing historical positive evidence. This build concerns science-class targets and stricter region associations. Neither has overwritten upstream experiments, prediction arrays, labels or scores.

## Verification and artifacts

All **51 unit tests pass**, including scientific window boundaries, TAI conversion, region ambiguity, uncertain clocks, cross-day event-number reuse, overlapping optical support and quarantine propagation. The separate verifier uses token-based raw parsing and assigns each event to forecast windows, rather than calling the builder's parsing or label functions. It checked:

- Checksums and all 30,553 parsed daily rows, including times and region fields.
- All 248 accepted associations and all 651 applied decisions.
- Preservation of every original science field and unchanged regions outside the queue.
- All 306,732 case/horizon rows, both target scopes, every event link, counts, labels and elapsed-hour endpoints.
- Preservation of original 48-hour labels, unassigned new experiment roles, and `training_ready=False`.

This is a computational cross-check by a separate code path, not external scientific validation or independent observations. NOAA daily reports, HER and HEK may share source lineage.

The compact [evidence package](../results/region_adjudication_20261001/README.md) contains decisions, raw accepted evidence, source receipts, candidate support, transitions and verification. Large local artifacts are Git-ignored:

| Artifact | Local repository-relative path |
|---|---|
| Pinned daily reports | `data/raw/swpc_region_adjudication_v1/` |
| Parsed daily rows and decisions | `data/processed/swpc_region_adjudication_v1/` |
| Versioned science events and 48/72-hour outcomes | `data/processed/gray_box_outcomes_v2/` |

Candidate outcome SHA-256: `37ca5f2094d6581e2b41dc282da8526cbb108dc559a9a827e47bf1dd6407c4ac`.

Reproduce from the pinned earlier inputs, using new output directories if the named versions already exist:

```bash
python scripts/adjudicate_swpc_regions.py \
  --queue results/event_reconciliation_20261001/event_review_queue.csv \
  --source-dir data/raw/swpc_region_adjudication_v1 \
  --output-dir data/processed/swpc_region_adjudication_v1

python scripts/build_adjudicated_outcomes.py \
  --inventory-dir data/processed/gray_box_inventory_v2 \
  --original-outcomes-dir data/processed/gray_box_outcomes_v1 \
  --adjudication-dir data/processed/swpc_region_adjudication_v1 \
  --leap-file data/raw/gray_box_inventory_v1/Leap_Second.dat \
  --output-dir data/processed/gray_box_outcomes_v2

python scripts/verify_adjudicated_outcomes.py \
  --source-dir data/raw/swpc_region_adjudication_v1 \
  --adjudication-dir data/processed/swpc_region_adjudication_v1 \
  --outcome-dir data/processed/gray_box_outcomes_v2 \
  --original-outcome-dir data/processed/gray_box_outcomes_v1 \
  --inventory-dir data/processed/gray_box_inventory_v2 \
  --output data/processed/gray_box_outcomes_v2/verification.json

python -m unittest discover -s tests -v
```

The acquisition script additionally accepts `--fetch` and repeated `--reuse-dir` arguments for the earlier `swpc_label_audit` and `swpc_daily_spot_checks` caches. It bounds the request to this queue, uses three concurrent downloads, pins each successful report, and records failures. For exact reproduction, obtain bytes matching the committed receipt; a fresh archive response may represent a later revision.

## Remaining dataset gates

The next work is to establish GOES observation/detection coverage over each required outcome interval, freeze the class convention and target region/crop scope, and record defensible availability assumptions. Gaps and unresolved associations must remain visible in the master table. Only after those choices are established should new training/calibration/policy/test roles be frozen and final comparisons run. Previously evaluated 2026 cases are not an untouched prospective test set.
