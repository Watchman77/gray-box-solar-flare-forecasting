"""Cross-check saved candidates without calling the builder's label/time routines.

Checks all rows for key, count, timing and readiness invariants, and compares a
deterministic stratified sample against direct event-table filtering. This is a
computational cross-check, not independent scientific validation of NOAA events.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-dir", required=True, type=Path)
    parser.add_argument("--outcome-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Cross-check receipt exists; choose a new output")
    inventory_path = args.inventory_dir / "forecast_case_inventory.csv.gz"
    inventory_receipt = json.loads((args.inventory_dir / "build_summary.json").read_text())
    outcome_receipt = json.loads((args.outcome_dir / "build_summary.json").read_text())
    check(sha(inventory_path) == inventory_receipt["manifest_sha256"] == outcome_receipt["inventory_sha256"], "Inventory checksum")
    for name, expected in outcome_receipt["output_sha256"].items():
        check(sha(args.outcome_dir / name) == expected, f"Output checksum: {name}")
    cases = pd.read_csv(inventory_path, low_memory=False)
    labels = pd.read_csv(args.outcome_dir / "candidate_outcomes.csv.gz", low_memory=False)
    events = pd.read_csv(args.outcome_dir / "normalized_events.csv.gz", low_memory=False)
    check(not cases.forecast_case_id.duplicated().any(), "Case key")
    check(not labels.duplicated(["forecast_case_id", "horizon_hours"]).any(), "Case/horizon key")
    check(set(labels.horizon_hours) == {48, 72}, "Horizon set")
    for horizon, part in labels.groupby("horizon_hours"):
        check(set(part.forecast_case_id) == set(cases.forecast_case_id), f"Case preservation at {horizon}")
    check(cases.region_component_id.notna().all(), "Component mapping")
    check(not cases.operational_training_ready.any() and not labels.training_ready.any(), "Readiness guard")
    check(labels.new_experiment_role.eq("unassigned").all(), "No inherited evaluation roles")
    native = pd.to_datetime(cases.issue_tai, format="%Y.%m.%d_%H:%M:%S_TAI")
    check(native.min() >= pd.Timestamp("2010-01-01") and native.max() < pd.Timestamp("2027-01-01"), "Independent time-conversion epoch")

    def known_epoch_utc(tai):
        # Fixed published transitions for this bounded epoch, independent of the
        # builder's parsed IERS/searchsorted implementation.
        offsets = np.full(len(tai), 34)
        offsets[tai >= pd.Timestamp("2012-07-01T00:00:35")] = 35
        offsets[tai >= pd.Timestamp("2015-07-01T00:00:36")] = 36
        offsets[tai >= pd.Timestamp("2017-01-01T00:00:37")] = 37
        return (tai - pd.to_timedelta(offsets, unit="s")).dt.tz_localize("UTC")

    check(known_epoch_utc(native).equals(pd.to_datetime(cases.issue_utc, utc=True)), "All issue UTC conversions")
    joined = labels.merge(cases[["forecast_case_id", "issue_tai", "NOAA_AR_clean", "NOAA_ARS", "source_noaa_multiple", "upstream_label_48h"]],
        on="forecast_case_id", how="left", validate="many_to_one", suffixes=("", "_inventory"))
    end_tai = pd.to_datetime(joined.issue_tai, format="%Y.%m.%d_%H:%M:%S_TAI") + pd.to_timedelta(joined.horizon_hours, unit="h")
    check(known_epoch_utc(end_tai).equals(pd.to_datetime(joined.outcome_end_utc, utc=True)), "All physical horizon endpoints")
    check(joined.upstream_label_48h.eq(joined.upstream_label_48h_inventory).all(), "Original labels preserved")
    for scope in ("primary", "patch"):
        positive = joined[f"candidate_{scope}_label"].eq(1)
        negative = joined[f"candidate_{scope}_label"].eq(0)
        missing = joined[f"candidate_{scope}_label"].isna()
        check((~positive | (joined[f"{scope}_event_count"] > 0)).all(), f"Positive event evidence: {scope}")
        check((~negative | ((joined[f"{scope}_event_count"] == 0) & (joined.unassigned_region_mx_count == 0))).all(), f"No ambiguous negative: {scope}")
        check((~missing | (~joined.nominal_span_contains_window | (joined.unassigned_region_mx_count > 0))).all(), f"Missing label reason: {scope}")
    mx = events.loc[events.flare_class.str.startswith(("M", "X"))].copy()
    mx["start"] = pd.to_datetime(mx.start_time, utc=True)
    selected = set(joined.sample(min(800, len(joined)), random_state=7201).index)
    # Sample the difficult states deliberately, not just common negatives.
    for mask in [joined.source_noaa_multiple, joined.unassigned_region_mx_count > 0,
                 joined.primary_event_count > 0, joined.association_convention_changes_recorded_event_presence]:
        pool = joined.loc[mask]
        selected.update(pool.sample(min(200, len(pool)), random_state=7202).index)
    for row in joined.loc[sorted(selected)].itertuples():
        window = mx.loc[(mx.start > pd.Timestamp(row.issue_utc)) & (mx.start <= pd.Timestamp(row.outcome_end_utc))]
        primary = set(window.loc[window.noaa_full_id_candidate == row.NOAA_AR_clean, "event_id"])
        patch_regions = {int(x) for x in row.NOAA_ARS.split(",")}
        patch = set(window.loc[window.noaa_full_id_candidate.isin(patch_regions), "event_id"])
        unknown = set(window.loc[window.noaa_full_id_candidate.isna(), "event_id"])
        for col, expected in [("primary_event_ids", primary), ("patch_event_ids", patch), ("unassigned_region_mx_event_ids", unknown)]:
            stored = getattr(row, col)
            actual = set() if pd.isna(stored) else set(stored.split(";"))
            check(actual == expected, f"Direct event-filter mismatch: {row.forecast_case_id}/{row.horizon_hours}/{col}")
    report = {"status": "computational_cross_check_passed_provisional_only", "case_rows": len(cases),
        "case_horizon_rows": len(labels), "direct_filter_sample_rows": len(selected),
        "sampling_seed_random": 7201, "sampling_seed_strata": 7202,
        "inventory_sha256": sha(inventory_path), "outcomes_sha256": sha(args.outcome_dir / "candidate_outcomes.csv.gz"),
        "checks": ["saved artifact hashes", "complete population at both horizons", "unique case/horizon IDs",
            "all component IDs present", "all issue/endpoint UTC conversions", "original 48h labels retained",
            "positive event support", "uncertain negatives remain missing", "no final training eligibility or split roles",
            "stratified direct event-filter comparison"],
        "scientific_validation_not_established": ["catalogue observing completeness", "event/region/class adjudication",
            "unique physical-event identity", "historical input availability", "model performance"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
