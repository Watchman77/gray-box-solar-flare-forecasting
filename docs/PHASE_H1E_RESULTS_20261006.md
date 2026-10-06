# Phase H1e results — source-series harmonization audit

Status: **EXECUTED — 6 October 2026**

H1e mapped the supplementary-2026 SHARP source series at slot and forecast-case level and compared gradient-family behaviour across the source transition. It made no policy change.

## 1. Exact source transition

The supplementary-2026 cohort switches almost completely from `hmi.sharp_720s` to `hmi.sharp_cea_720s` at 4 April 2026:

- last CCD-series slot: `2026.04.03_23:12:00_TAI`
- first CEA-series slot: `2026.04.04_00:48:00_TAI`
- last pure-CCD forecast issue: `2026-04-04 00:47:23 UTC`
- first case containing a CEA slot: `2026-04-04 02:23:23 UTC`
- first pure-CEA forecast issue: `2026-04-04 05:35:23 UTC`

Case composition:
- pure CCD: 2,077
- mixed three-slot history: 4
- pure CEA: 10,078

Thus source series and calendar time are almost perfectly confounded after 4 April.

## 2. Gradient distributions differ strongly across the two temporal/source strata

Latest-slot raw medians:

| Series | MEANGBZ | MEANGBH | MEANGBT |
|---|---:|---:|---:|
| `hmi.sharp_720s` | 123.008 | 58.410 | 123.150 |
| `hmi.sharp_cea_720s` | 102.248 | 48.242 | 99.810 |

Training-coordinate medians are correspondingly much lower in the CEA-era records.

The H1b applicability summary follows the same split:

| Composition | Cases | Median MD² | q99 OOD | Median gradients-latest |
|---|---:|---:|---:|---:|
| pure CCD | 2,077 | 13.083 | 1.11% | -0.060 |
| pure CEA | 10,078 | 17.671 | 3.20% | -0.897 |
| mixed | 4 | 11.439 | 0.0% | -0.473 |

This establishes a strong temporal/source-series association, not causality.

## 3. The exact transition does not show a large discontinuity in the tiny direct overlap

Only two consecutive same-HARP cross-series record pairs occur within six hours of the source transition. Their median changes are:

- MEANGBZ: -1.84
- MEANGBH: -2.68
- MEANGBT: -1.99

These changes are comparable to ordinary same-series consecutive-record absolute changes. Therefore the two direct transition pairs do **not** show evidence of a catastrophic instantaneous keyword jump.

This sample is far too small to establish parity or non-parity between the two series.

## 4. Same-HARP/month overlap remains temporally confounded

Four HARP-month strata contain both source series. Their median CEA-minus-CCD gradient differences are large:

- MEANGBZ: -16.70
- MEANGBH: -24.67
- MEANGBT: -20.30

However, the CCD observations occur before the 4 April switch and the CEA observations after it. These are not simultaneous paired records, so active-region evolution and viewing geometry remain confounded with series.

## 5. Processing metadata

Both series use the same recorded `CODEVER7` and `CALVER64` values in the retained native-slot ledger, but they differ in:

- `source_origin`: `locked_sharp96` versus `definitive_extension`
- DRMS series: `hmi.sharp_720s` versus `hmi.sharp_cea_720s`
- query lineage / response provenance.

This is a material provenance boundary, but not proof of a changed numerical definition for the 16 SHARP summary keywords.

## 6. Important documentation nuance from the HMI/SHARP literature

The HMI vector-pipeline documentation describes `hmi.sharp_720s` and `hmi.sharp_cea_720s` as different map-coordinate products, but also states that the SHARP active-region indices are generally computed from high-confidence values in the remapped CEA representation and stored as SHARP keywords.

Therefore the presence of `hmi.sharp_cea_720s` after April does **not by itself prove** that the summary-keyword definitions changed. Direct same-record keyword parity is required.

## 7. H1e interpretation category

**SOURCE_SERIES_PARTIAL_CONFOUND / INSUFFICIENT SAME-TIME OVERLAP TO RESOLVE**

The current archive cannot distinguish cleanly between:
- genuine temporal evolution in the active-region population,
- viewing/selection effects,
- and source-series/provenance effects,

because the series transition is almost perfectly aligned with time.

The two immediate cross-series pairs do not show a large discontinuity, which argues against a simple catastrophic source switch. But the sample is insufficient.

## 8. Consequence for the Gray-Box framework

The physical/provenance layer has successfully detected an input-domain change that statistical confidence signals did not expose. However, physical MD² remains a monitoring diagnostic rather than a routing gate.

The next step is H1f: query both SHARP series for the **same HARP and same T_REC** and compare all 16 summary keywords directly.
