"""Verify saved region proposals and label lineage through direct source filtering."""

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from scripts.prepare_dataset_sources import digest
except ModuleNotFoundError:
    from prepare_dataset_sources import digest


def check(value, message):
    if not value:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconciliation-dir", required=True, type=Path)
    parser.add_argument("--previous-outcomes-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Verification receipt already exists")
    root = args.reconciliation_dir
    summary = json.loads((root / "reconciliation_summary.json").read_text())
    for name, expected in summary["output_sha256"].items():
        check(digest(root / name) == expected, f"Output hash changed: {name}")
    refs = pd.read_csv(root / "reference_events.csv.gz")
    for field in ["start_utc", "peak_utc", "end_utc"]:
        refs[field] = pd.to_datetime(refs[field], utc=True)
    old_events = pd.read_csv(args.previous_outcomes_dir / "normalized_events.csv.gz")
    new_events = pd.read_csv(root / "source_enriched_events.csv.gz")
    check(old_events.event_id.equals(new_events.event_id), "Event population/order changed")
    check(old_events.flare_class.equals(new_events.flare_class), "Science classes changed")
    for field in ["start_time", "time", "end_time"]:
        check(pd.to_datetime(old_events[field], utc=True).equals(pd.to_datetime(new_events[field], utc=True)), f"Science timing changed: {field}")
    present = old_events.noaa_full_id_candidate.notna()
    check(old_events.loc[present, "noaa_full_id_candidate"].equals(new_events.loc[present, "noaa_full_id_candidate"]), "Existing science association overwritten")
    audit = pd.read_csv(root / "event_reconciliation.csv.gz")
    proposed = audit.loc[audit.proposed_region_fill.notna()]
    all_peaks = pd.to_datetime(old_events.time, utc=True)
    for event in proposed.itertuples():
        peak, start = pd.Timestamp(event.peak_utc), pd.Timestamp(event.start_utc)
        check(int((all_peaks == peak).sum()) == 1, "A proposed fill has a nonunique science peak")
        candidates = refs.loc[refs.peak_utc.eq(peak) & ~refs.clock_rollover_assumed & ~refs.source_time_problem]
        known = set(candidates.noaa_region.dropna().astype(int))
        check(known == {int(event.proposed_region_fill)}, "A proposed fill has conflicting known regions")
        check(((candidates.start_utc == start) & candidates.noaa_region.eq(event.proposed_region_fill)).any(), "No exact-start counterpart for proposed fill")
        check(pd.isna(event.original_noaa_region), "Proposed fill was not originally missing")
    previous = pd.read_csv(args.previous_outcomes_dir / "candidate_outcomes.csv.gz", low_memory=False)
    current = pd.read_csv(root / "reference_enriched_candidate_outcomes.csv.gz", low_memory=False)
    check(not current.duplicated(["forecast_case_id", "horizon_hours"]).any(), "Duplicate case/horizon")
    check(not current.training_ready.any() and current.new_experiment_role.eq("unassigned").all(), "Readiness guard changed")
    joined = previous.merge(current, on=["forecast_case_id", "horizon_hours"], how="outer", validate="one_to_one", indicator=True, suffixes=("_old", "_new"))
    check(joined._merge.eq("both").all(), "Case population changed")
    for scope in ["primary", "patch"]:
        old = joined[f"candidate_{scope}_label_old"]
        new = joined[f"candidate_{scope}_label_new"]
        check((old.isna() | old.eq(new)).all(), "Already-resolved candidate changed")
    diagnostics = pd.read_csv(root / "label_lineage_diagnostics.csv.gz", low_memory=False)
    check(diagnostics.legacy_combined_presence_48h_native_clock.eq(diagnostics.upstream_label_48h).all(), "Original label replay failed")
    flagged = diagnostics.loc[diagnostics.old_negative_with_corroborated_science_positive]
    check(flagged.HER_annual_recorded_presence_48h_utc.eq(1).all(), "A flagged window lacks annual positive evidence within its interval")
    # Independently match the corroborating events to each flagged case and its
    # original NOAA ID, rather than checking only aggregate counts.
    corroborating_event_ids = set()
    for row in flagged.itertuples():
        start = pd.Timestamp(row.issue_utc)
        end = start + pd.Timedelta(hours=48)  # Flagged years 2021–2024 contain no leap insertion.
        check(2021 <= start.year <= 2024, "Flagged verification needs a different time-epoch implementation")
        ids = set(row.primary_event_ids.split(";"))
        events = audit.loc[audit.event_id.isin(ids) & audit.original_noaa_region.eq(row.NOAA_AR_clean)
            & audit.same_region_mx_annual_reference_ids.notna() & ~audit.same_region_mx_base_hek_reference]
        found = False
        for event in events.itertuples():
            annual_ids = event.same_region_mx_annual_reference_ids.split(";")
            annual = refs.loc[refs.reference_id.isin(annual_ids)]
            support = annual.loc[(annual.start_utc > start) & (annual.start_utc <= end)
                & annual.noaa_region.eq(row.NOAA_AR_clean) & annual.is_mx
                & annual.peak_utc.eq(pd.Timestamp(event.peak_utc))]
            if len(support):
                found = True
                corroborating_event_ids.add(event.event_id)
        check(found, f"No direct paired event support for flagged case: {row.forecast_case_id}")
    report = {"status": "computational_cross_check_passed_not_final_label_clearance",
        "proposal_events_checked": len(proposed), "case_horizon_rows_checked": len(current),
        "all_case_label_replay_checked": len(diagnostics), "flagged_case_event_links_checked": len(flagged),
        "distinct_corroborating_science_events_in_flagged_cases": len(corroborating_event_ids),
        "checks": ["output hashes", "science event population/classes/times preserved", "existing science region IDs preserved",
            "every proposed fill checked by direct reference filtering", "all cases retained at both horizons",
            "previously resolved candidates preserved", "original recorded-clock labels reproduced for every case",
            "every flagged case has same-region M/X annual evidence inside the same future interval at the same science peak"],
        "limitations": ["Source consistency is not an independent observation or complete physical-event validation.",
            "No catalogue coverage clearance, final labels, split freeze or model rerun is supplied."],
        "reconciliation_summary_sha256": digest(root / "reconciliation_summary.json")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
