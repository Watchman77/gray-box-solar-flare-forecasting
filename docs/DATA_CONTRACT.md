# Forecast data contract — draft

## Unit of prediction

One row per `region_component_id`, `issue_time_utc`, `horizon_hours`, and `target_definition`, per experiment/model version. Region forecasts and full-disk forecasts are distinct tasks.

Proposed horizons: 72 hours first; then 24 and 3 hours. The existing 48-hour experiments retain their original scope. The initial target is a combined M-or-X event in the same region within `(issue_time, issue_time + horizon]`; the choice of event start versus peak time must be explicitly frozen before labels are generated. Separate exceedance targets require their own named definition.

## Required inputs

| Source | Required content | Checks |
|---|---|---|
| SHARP | Feature values/units, HARP and NOAA associations, actual times, product series/version and quality | History completeness; timestamp semantics; NRT versus definitive provenance; availability before issue time. |
| AIA | Channel order, observation times/exposure, spatial mapping, source and processing version | Aligned six-channel contract (94/131/171/193/211/335 Å), channel-specific quality, missingness and latency. |
| Flare catalogue | Event identifiers, class, timestamps with time scale, region association, provider/version, coverage interval | Completeness, duplicate/ambiguous events, association consistency and follow-up through the entire horizon. |
| Availability | Expected and received observations, unavailable channels, outage blocks and latency | Preserve the intended forecast population, including cases with missing inputs. |

Continuous GOES/XRS or flare-history predictors are optional additional inputs. A catalogue used to label future events is not by itself a GOES predictor branch.

## Master prediction table

| Field group | Minimum fields |
|---|---|
| Identity | `forecast_id`, `experiment_id`, `region_component_id`, HARP/NOAA IDs, `issue_time_utc`, `horizon_hours`, `target_definition` |
| Source provenance | source URIs/IDs, object generation or checksum, product series/version, preprocessing version |
| Time and availability | original timestamp and time scale, observation start/end UTC, availability time or explicit latency assumption |
| Quality | channel/feature missingness masks, quality flags, history completeness, source-regime identifier |
| Outcomes | label, matched event IDs, label version, outcome time definition, follow-up end, completeness/ambiguity flag |
| Experimental role | split/fold, training/calibration/policy/test role, role-assignment version |
| Forecast | raw score, calibrated probability, model/calibrator versions, uncertainty method/output |
| Decision | policy version, normal/degraded/abstain state, reason codes, fallback model/calibrator identity |

## Rules

- Convert time scales; do not obtain UTC by deleting a `TAI` suffix.
- Keep observations and their historical availability separate. A modern cloud upload is not evidence of historical operational availability.
- Preserve original labels; corrections create a new version with an audit trail.
- An absent event entry is not a verified negative when catalogue coverage or region association is incomplete.
- Training, calibration and policy fitting use only their designated earlier data. Keep evaluation prevalence natural.
- Give all compared branches the same primary forecast cases; report availability-based exclusions separately.
- Record unmatched or missing cases; do not silently select complete cases as the sole operational population.
- Retain quality/OOD/abstention outcomes even when no probability is issued.

## 2026 extension — updated 1 October 2026

The upstream AIA project records its extension through 17 August 2026 as sealed, with 12,159 eligible 48-hour forecast cases and 803 positive windows in the supplementary evaluation. The extension-specific label work supersedes the earlier warning about relying on the May catalogue snapshot. These are upstream 48-hour outcomes, not validated 72-hour labels. Forecasts near 17 August require complete follow-up through their corresponding times on 20 August at a 72-hour horizon, or an earlier issue cutoff. The latest event in a catalogue does not itself prove its coverage interval. The retained population has already been evaluated and cannot be presented as untouched confirmation for a new policy motivated by those results.

## Historical 72-hour input

The 1 October [baseline audit](BASELINE_72H_AUDIT.md) reproduces cohort support and recovered headline predictions, but finds unresolved region/time continuity and availability issues. Treat the historical CSV and its embedded labels as preserved source evidence. The new operational case table must establish source identities, physical time, label coverage and missing-input accounting independently.

## Executed candidate inventory — 1 October 2026

The [candidate dataset build](DATASET_BUILD_20261001.md) is a staging table at **native HARP-record grain**, keyed by the pinned source sample ID. It is not yet the component-level master prediction table defined above. Multiple HARP records share a component and issue time in 494 groups; retain these records and resolve the forecast unit before aggregation or evaluation.

Original 48-hour labels and roles are retained as upstream evidence. New experiment roles are unassigned. Provisional 48/72-hour outcomes use the explicitly named start-time interval `(t, t+h]`, with physical elapsed hours calculated in TAI before converting endpoints to UTC. The 48-hour reconstruction is a source/convention audit and does not replace the AIA experiment's outcomes.

Two association scopes are recorded separately: the source's primary NOAA ID, and the union of NOAA IDs listed on that HARP record. A component's future membership is not used to assign events. Missing-region M/X events mask otherwise-negative candidates conservatively. Nominal catalogue dates are not evidence of continuous observing coverage, so even unmasked zeros remain provisional. No build output is marked training-ready.
