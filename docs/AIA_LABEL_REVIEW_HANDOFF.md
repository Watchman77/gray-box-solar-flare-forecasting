# AIA label review handoff — prepared locally, not sent to another chat

The Gray-Box source audit found 1,033 existing negative 48-hour windows with source-supported M/X events from their forecast region inside the future interval. This includes **798 matched primary Cycle-25 cases: 35 in 2021, 316 in 2022, 446 in 2023 and one in 2024**.

All original 153,366 labels reproduce exactly from the older HEK catalogue plus the 2026 extension under the original clock convention. The source issue therefore needs separate attention from implementation reproducibility. Converting to physical UTC causes only one additional disagreement outside matched input support.

All 90 underlying science events were checked against 64 archived NOAA daily reports. Of the 1,033 windows, 1,032 have directly associated M/X X-ray reports; one uses the same-day SWPC optical/X-ray event group to obtain the region. The reports have preserved hashes and raw-line verification. HER and SWPC share source lineage, so this is not independent observational validation.

Use [the correction proposal table](../results/event_reconciliation_20261001/source_supported_corrections_48h.csv) to join on the **source sample ID**: `forecast_case_id` has the explicit prefix `gb72-native-v1:` followed by the unchanged upstream ID. Despite that inventory prefix, these corrections concern **48-hour** labels only. Match by ID, not row position. All original labels in this bounded table are 0 and all source-supported proposals are 1.

Do not silently replace frozen labels, restart training, fit a calibrator on corrected test outcomes, or recompute only selected cases. First reconcile the whole label source and target convention, create a versioned label manifest, then re-evaluate all frozen predictions on the complete applicable population. Keep original label-version results alongside any revised ones. Training or calibration changes require a separate protocol and version.

The full event review still has unresolved region/class/coverage questions. No new TSS/AP, calibration result, corrected model, or final label release is claimed. The broader [reconciliation report](EVENT_RECONCILIATION_20261001.md) documents scope, evidence and reproduction.
