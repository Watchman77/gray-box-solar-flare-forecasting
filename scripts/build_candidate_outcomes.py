"""Audit NOAA science events and construct provisional 48/72-hour start-time labels.

This is a source/convention comparison, not final training truth. The science
catalogue does not establish uninterrupted observing coverage. Unknown-region
events mask otherwise-negative candidates; all output labels remain provisional.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

try:
    from scripts.build_dataset_inventory import require_unique, tai_to_utc, write_csv
    from scripts.prepare_dataset_sources import digest
except ModuleNotFoundError:
    from build_dataset_inventory import require_unique, tai_to_utc, write_csv
    from prepare_dataset_sources import digest


def parse_regions(value, primary):
    """Only expand the four-digit catalogue suffix within this verified NOAA epoch."""
    if not re.fullmatch(r"\d+(?:,\d+)*", str(value)):
        raise ValueError(f"Unresolved NOAA association: {value}")
    regions = tuple(sorted(set(int(x) for x in str(value).split(","))))
    if any(x < 10000 or x >= 20000 for x in regions) or int(primary) != primary or primary not in regions:
        raise ValueError("NOAA epoch or primary association requires explicit reconciliation")
    return regions


def normalize_events(raw):
    events = raw.copy(deep=True)
    require_unique(events, ["flare_id"], "flare event ID")
    if events.flare_id.astype(str).str.fullmatch(r"\d+").eq(False).any():
        raise ValueError("Invalid flare ID")
    for col in ["time", "start_time", "end_time"]:
        events[col] = pd.to_datetime(events[col], utc=True, errors="raise")
    if events[["time", "start_time"]].isna().any().any() or (events.start_time > events.time).any():
        raise ValueError("Missing or inconsistent event timing")
    if (events.end_time.notna() & (events.end_time < events.time)).any():
        raise ValueError("Event end precedes peak")
    if not events.flare_class.astype(str).str.fullmatch(r"[ABCMX]\d+(?:\.\d+)?").all():
        raise ValueError("Unknown flare class convention")
    present = events.active_region.dropna()
    if not ((present >= 1) & (present <= 9999) & (present % 1 == 0)).all():
        raise ValueError("Invalid four-digit NOAA region suffix")
    if not events.peak_saturated.isin([0, 1]).all():
        raise ValueError("Invalid saturation flag")
    if (events.start_time < pd.Timestamp("2010-01-01", tz="UTC")).any() or (events.time >= pd.Timestamp("2027-01-01", tz="UTC")).any():
        raise ValueError("Event time outside the supported 2010-2026 normalization epoch")
    events["noaa_full_id_candidate"] = (events.active_region + 10000).astype("Int64")
    events["is_mx"] = events.flare_class.str.startswith(("M", "X"))
    events["event_id"] = "noaa-science-v1-0-1:" + events.flare_id.astype(str)
    return events.sort_values(["start_time", "event_id"], kind="stable").reset_index(drop=True)


def interval_ids(index, start_ns, end_ns):
    """Return event IDs in the exact open-left, closed-right interval (t, end]."""
    if end_ns <= start_ns:
        raise ValueError("Outcome interval must have positive duration")
    times, ids = index
    left, right = np.searchsorted(times, [start_ns, end_ns], side="right")
    return tuple(ids[left:right])


def provisional_label(ids, unknown_ids, nominal_span):
    # Do not assign even a partial-window candidate outside the source envelope.
    if not nominal_span:
        return pd.NA
    if ids:
        return 1
    return pd.NA if unknown_ids else 0


def label_cases(cases, events, ends, nominal_start, nominal_end):
    """Keep both source-primary and all-listed-NOAA conventions as separate fields."""
    mx = events.loc[events.is_mx].copy()

    def event_index(frame):
        ordered = frame.sort_values(["start_time", "event_id"])
        return (ordered.start_time.astype("int64").to_numpy(), ordered.event_id.to_numpy())

    by_region = {int(region): event_index(part) for region, part in mx.dropna(subset=["noaa_full_id_candidate"]).groupby("noaa_full_id_candidate")}
    unknown = event_index(mx.loc[mx.noaa_full_id_candidate.isna()])
    empty = (np.array([], dtype="int64"), np.array([], dtype=str))
    starts = pd.to_datetime(cases.issue_utc, utc=True).astype("int64").to_numpy()
    end_ns = pd.to_datetime(ends, utc=True).astype("int64").to_numpy()
    rows = []
    for case, start, end in zip(cases.itertuples(), starts, end_ns):
        regions = parse_regions(case.NOAA_ARS, case.NOAA_AR_clean)
        primary = interval_ids(by_region.get(int(case.NOAA_AR_clean), empty), start, end)
        patch = sorted({event for region in regions for event in interval_ids(by_region.get(region, empty), start, end)})
        unassigned = interval_ids(unknown, start, end)
        within = start >= nominal_start.value and end <= nominal_end.value
        rows.append({
            "forecast_case_id": case.forecast_case_id,
            "primary_event_ids": ";".join(primary), "patch_event_ids": ";".join(patch),
            "unassigned_region_mx_event_ids": ";".join(unassigned),
            "primary_event_count": len(primary), "patch_event_count": len(patch),
            "unassigned_region_mx_count": len(unassigned),
            "candidate_primary_label": provisional_label(primary, unassigned, within),
            "candidate_patch_label": provisional_label(patch, unassigned, within),
            "nominal_span_contains_window": within,
            "association_convention_changes_recorded_event_presence": bool(primary) != bool(patch),
        })
    result = pd.DataFrame(rows)
    for col in ("candidate_primary_label", "candidate_patch_label"):
        result[col] = result[col].astype("Int64")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--inventory-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Output version exists; preserve it and choose a new directory")
    receipt = json.loads((args.source_dir / "outcome_source_receipt.json").read_text())
    if receipt["status"] != "complete":
        raise ValueError("Incomplete source receipt")
    names = {x["filename"] for x in receipt["files"]}
    expected = {f"sci_xrsf-l2-flrpt_geo_y{y}_v1-0-1.csv" for y in range(2010, 2027)}
    expected.add("sci_xrsf-l2-flrpt_geo_metadata.json")
    if names != expected or len(receipt["files"]) != len(expected):
        raise ValueError("Source annual partition set changed")
    root = args.source_dir / "noaa_flare_report_v1_0_1"
    for item in receipt["files"]:
        if digest(root / item["filename"]) != item["sha256"]:
            raise ValueError(f"Source hash changed: {item['filename']}")
    raw = pd.concat([pd.read_csv(root / name, dtype={"flare_id": str}).assign(source_file=name)
                     for name in sorted(expected) if name.endswith(".csv")], ignore_index=True)
    events = normalize_events(raw)
    metadata = json.loads((root / "sci_xrsf-l2-flrpt_geo_metadata.json").read_text())["global_attributes"]
    if metadata["flrpt_algorithm_version"] != "1-0-1":
        raise ValueError("Unexpected event processing version")
    # Metadata end is date-only. Midnight on that date is a conservative bound,
    # not evidence of continuous detection or of complete region association.
    nominal_start = max(pd.Timestamp(metadata["time_coverage_start"], tz="UTC"), pd.Timestamp("2010-01-01", tz="UTC"))
    nominal_end = min(pd.Timestamp(metadata["time_coverage_end"], tz="UTC"), pd.Timestamp("2027-01-01", tz="UTC"))
    inventory = args.inventory_dir / "forecast_case_inventory.csv.gz"
    inventory_receipt = json.loads((args.inventory_dir / "build_summary.json").read_text())
    if digest(inventory) != inventory_receipt["manifest_sha256"] or digest(args.source_dir / "Leap_Second.dat") != inventory_receipt["leap_table_sha256"]:
        raise ValueError("Inventory or leap-table provenance mismatch")
    cases = pd.read_csv(inventory, low_memory=False)
    require_unique(cases, ["forecast_case_id"], "forecast case")
    output, support = [], []
    native = pd.to_datetime(cases.issue_tai, format="%Y.%m.%d_%H:%M:%S_TAI")
    for horizon in (48, 72):
        ends = tai_to_utc((native + pd.Timedelta(hours=horizon)).dt.strftime("%Y.%m.%d_%H:%M:%S_TAI"), args.source_dir / "Leap_Second.dat")
        labelled = label_cases(cases, events, ends, nominal_start, nominal_end)
        labelled["horizon_hours"] = horizon
        labelled["issue_utc"] = cases.issue_utc
        labelled["outcome_end_utc"] = ends
        labelled["upstream_label_48h"] = cases.upstream_label_48h
        labelled["upstream_matched_three_slot_inputs"] = cases.upstream_matched_three_slot_inputs
        labelled["input_cohort"] = cases.input_cohort.fillna("not_in_matched_inputs")
        labelled["target_version"] = "candidate_mx_start_science_v1_0_1"
        labelled["continuous_catalogue_coverage_status"] = "not_established"
        labelled["label_status"] = "provisional_not_training_truth"
        labelled["new_experiment_role"] = "unassigned"
        labelled["training_ready"] = False
        if horizon == 72 and not (ends.to_numpy() == pd.to_datetime(cases.outcome_end_utc, utc=True).to_numpy()).all():
            raise ValueError("72-hour endpoint differs from inventory")
        for cohort, part in labelled.groupby("input_cohort", sort=True):
            for scope in ("primary", "patch"):
                label = part[f"candidate_{scope}_label"]
                resolved = label.notna()
                support.append({"horizon_hours": horizon, "input_cohort": cohort, "association_scope": scope,
                    "cases": len(part), "candidate_positive": int((label == 1).sum()),
                    "candidate_negative": int((label == 0).sum()), "unresolved": int(label.isna().sum()),
                    "different_from_upstream_48h_among_resolved": int((label[resolved] != part.loc[resolved, "upstream_label_48h"]).sum()) if horizon == 48 else None})
        output.append(labelled)
    labels = pd.concat(output, ignore_index=True)
    require_unique(labels, ["forecast_case_id", "horizon_hours"], "case/horizon")
    mx = events.loc[events.is_mx]
    args.output_dir.mkdir(parents=True)
    write_csv(events, args.output_dir / "normalized_events.csv.gz")
    write_csv(labels, args.output_dir / "candidate_outcomes.csv.gz")
    pd.DataFrame(support).to_csv(args.output_dir / "candidate_support.csv", index=False)
    event_profile = events.assign(year=events.time.dt.year, missing_ar=events.active_region.isna(),
        mx_missing_ar=events.is_mx & events.active_region.isna()).groupby("year").agg(
        events=("event_id", "size"), mx_events=("is_mx", "sum"), missing_region=("missing_ar", "sum"),
        mx_missing_region=("mx_missing_ar", "sum")).reset_index()
    event_profile.to_csv(args.output_dir / "event_source_profile.csv", index=False)
    summary = {
        "built_utc": datetime.now(timezone.utc).isoformat(), "status": "candidate_outcomes_built_not_training_ready",
        "inventory_sha256": digest(inventory), "source_receipt_sha256": digest(args.source_dir / "outcome_source_receipt.json"),
        "catalogue_events": len(events), "mx_events": len(mx), "mx_events_missing_region": int(mx.active_region.isna().sum()),
        "events_missing_end": int(events.end_time.isna().sum()), "events_missing_start_or_peak": 0,
        "duplicate_event_ids": 0, "nominal_start_utc": nominal_start.isoformat(), "conservative_nominal_end_utc": nominal_end.isoformat(),
        "candidate_case_horizon_rows": len(labels), "windows_outside_nominal_span": int((~labels.nominal_span_contains_window).sum()),
        "scope_changes_by_horizon": {str(h): int(p.association_convention_changes_recorded_event_presence.sum()) for h, p in labels.groupby("horizon_hours")},
        "support": support, "output_sha256": {name: digest(args.output_dir / name) for name in
            ["normalized_events.csv.gz", "candidate_outcomes.csv.gz", "candidate_support.csv", "event_source_profile.csv"]},
        "limitations": [
            "All labels are provisional; no training or final split assignment is authorized by this build.",
            "Annual files and nominal coverage do not establish uninterrupted observing/detection coverage.",
            "Source-primary and any-NOAA-listed-on-the-current-HARP-record targets are distinct conventions, not interchangeable labels.",
            "Unknown-region M/X events conservatively mask otherwise-negative windows for every target; these are screening flags, not evidence that each region flared.",
            "Science-calibrated classes can differ from historical operational classes. Label disagreements have not been adjudicated.",
            "This source contains minute-resolution reported starts; no uncertainty in the event boundary is modeled.",
            "Region suffix expansion is limited to the verified 2010-2026 NOAA 10000-19999 epoch; no future component membership is used for event matching.",
        ],
    }
    (args.output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("support", "output_sha256")}, indent=2))


if __name__ == "__main__":
    main()
