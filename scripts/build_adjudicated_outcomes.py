"""Build separate science-class candidate labels with daily-source region decisions.

Unresolved conflicting known regions are quarantined to unknown. No catalogue
class, time, original region, upstream label, or experiment role is overwritten.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

try:
    from scripts.prepare_dataset_sources import digest
    from scripts.build_dataset_inventory import require_unique, tai_to_utc, write_csv
    from scripts.build_candidate_outcomes import label_cases
except ModuleNotFoundError:
    from prepare_dataset_sources import digest
    from build_dataset_inventory import require_unique, tai_to_utc, write_csv
    from build_candidate_outcomes import label_cases


def apply_decisions(science, decisions):
    require_unique(science, ["event_id"], "science event")
    require_unique(decisions, ["event_id"], "adjudication event")
    if not set(decisions.event_id).issubset(science.event_id):
        raise ValueError("Decision references an unknown science event")
    events = science.copy(deep=True).set_index("event_id", drop=False)
    events["original_noaa_full_id"] = events.noaa_full_id_candidate.astype("Int64")
    events["noaa_full_id_candidate"] = events.noaa_full_id_candidate.astype("Int64")
    events["region_decision"] = "original_science_not_in_review_queue"
    for decision in decisions.itertuples():
        original = events.loc[decision.event_id, "original_noaa_full_id"]
        if pd.isna(original) != pd.isna(decision.original_region) or (pd.notna(original) and original != decision.original_region):
            raise ValueError("Decision original region differs from source")
        accepted = decision.status.startswith("filled_missing_") or decision.status == "retained_original_daily_supported"
        if accepted != pd.notna(decision.candidate_region):
            raise ValueError("Decision status/region mismatch")
        if decision.status.startswith("filled_missing_") and pd.notna(original):
            raise ValueError("A fill cannot replace an existing region")
        if decision.status == "retained_original_daily_supported" and original != decision.candidate_region:
            raise ValueError("A retained original must match")
        events.loc[decision.event_id, "noaa_full_id_candidate"] = decision.candidate_region if accepted else pd.NA
        events.loc[decision.event_id, "region_decision"] = decision.status
    return events.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--original-outcomes-dir", type=Path, required=True)
    parser.add_argument("--adjudication-dir", type=Path, required=True)
    parser.add_argument("--leap-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Output version already exists")
    prior = json.loads((args.original_outcomes_dir / "build_summary.json").read_text())
    inv = json.loads((args.inventory_dir / "build_summary.json").read_text())
    adj = json.loads((args.adjudication_dir / "summary.json").read_text())
    if digest(args.leap_file) != inv["leap_table_sha256"]:
        raise ValueError("Leap table changed")
    inventory_path = args.inventory_dir / "forecast_case_inventory.csv.gz"
    if digest(inventory_path) != inv["manifest_sha256"] or digest(inventory_path) != prior["inventory_sha256"]:
        raise ValueError("Inventory lineage changed")
    for name in ["normalized_events.csv.gz", "candidate_outcomes.csv.gz"]:
        if digest(args.original_outcomes_dir / name) != prior["output_sha256"][name]:
            raise ValueError("Original outcomes changed")
    if digest(args.adjudication_dir / "event_decisions.csv") != adj["output_sha256"]["event_decisions.csv"]:
        raise ValueError("Adjudication changed")
    cases = pd.read_csv(inventory_path, low_memory=False)
    decisions = pd.read_csv(args.adjudication_dir / "event_decisions.csv")
    science = pd.read_csv(args.original_outcomes_dir / "normalized_events.csv.gz", low_memory=False)
    events = apply_decisions(science, decisions)
    for col in ["time", "start_time", "end_time"]:
        events[col] = pd.to_datetime(events[col], utc=True)
    original = pd.read_csv(args.original_outcomes_dir / "candidate_outcomes.csv.gz", low_memory=False)
    native = pd.to_datetime(cases.issue_tai, format="%Y.%m.%d_%H:%M:%S_TAI")
    outputs, support, transitions = [], [], []
    for horizon in (48, 72):
        ends = tai_to_utc((native + pd.Timedelta(hours=horizon)).dt.strftime("%Y.%m.%d_%H:%M:%S_TAI"), args.leap_file)
        result = label_cases(cases, events, ends, pd.Timestamp(prior["nominal_start_utc"]), pd.Timestamp(prior["conservative_nominal_end_utc"]))
        result["horizon_hours"] = horizon
        result["issue_utc"] = cases.issue_utc
        result["outcome_end_utc"] = ends
        result["upstream_label_48h"] = cases.upstream_label_48h
        result["upstream_matched_three_slot_inputs"] = cases.upstream_matched_three_slot_inputs
        result["input_cohort"] = cases.input_cohort.fillna("not_in_matched_inputs")
        result["target_version"] = "candidate_mx_start_science_swpc_regions_v2"
        result["continuous_catalogue_coverage_status"] = "not_established"
        result["label_status"] = "provisional_not_training_truth"
        result["new_experiment_role"] = "unassigned"
        result["training_ready"] = False
        earlier = original.loc[original.horizon_hours.eq(horizon)].set_index("forecast_case_id").loc[result.forecast_case_id]
        for scope in ["primary", "patch"]:
            col = f"candidate_{scope}_label"
            before = earlier[col].astype("Int64").astype(str).to_numpy()
            after = result[col].astype(str).to_numpy()
            comparison = pd.DataFrame({"old": before, "new": after, "cohort": result.input_cohort})
            for (cohort, old, new), count in comparison.groupby(["cohort", "old", "new"]).size().items():
                transitions.append({"horizon_hours": horizon, "scope": scope, "input_cohort": cohort,
                    "original_science_label": old, "adjudicated_candidate_label": new, "cases": int(count)})
            for cohort, part in result.groupby("input_cohort"):
                support.append({"horizon_hours": horizon, "scope": scope, "input_cohort": cohort,
                    "cases": len(part), "positive": int(part[col].eq(1).sum()), "provisional_negative": int(part[col].eq(0).sum()),
                    "unresolved": int(part[col].isna().sum())})
        outputs.append(result)
    labels = pd.concat(outputs, ignore_index=True)
    require_unique(labels, ["forecast_case_id", "horizon_hours"], "case/horizon")
    args.output_dir.mkdir(parents=True)
    write_csv(events, args.output_dir / "adjudicated_events.csv.gz")
    write_csv(labels, args.output_dir / "candidate_outcomes.csv.gz")
    pd.DataFrame(support).to_csv(args.output_dir / "candidate_support.csv", index=False)
    pd.DataFrame(transitions).to_csv(args.output_dir / "label_transitions.csv", index=False)
    year = pd.to_datetime(decisions.science_peak_utc, utc=True).dt.year
    decisions.assign(year=year).groupby(["year", "status"]).size().rename("events").reset_index().to_csv(args.output_dir / "decisions_by_year.csv", index=False)
    summary = {"built_utc": datetime.now(timezone.utc).isoformat(), "target_version": "candidate_mx_start_science_swpc_regions_v2",
        "inventory_sha256": digest(inventory_path), "original_summary_sha256": digest(args.original_outcomes_dir / "build_summary.json"),
        "adjudication_summary_sha256": digest(args.adjudication_dir / "summary.json"),
        "catalogue_events": len(events), "mx_events": int(events.is_mx.sum()),
        "original_mx_missing_region": int((science.is_mx & science.noaa_full_id_candidate.isna()).sum()),
        "filled_missing_regions": int(decisions.status.str.startswith("filled_missing_").sum()),
        "retained_reviewed_original_regions": int(decisions.status.eq("retained_original_daily_supported").sum()),
        "quarantined_original_regions": int((decisions.original_region.notna() & decisions.candidate_region.isna()).sum()),
        "candidate_mx_unknown_region": int((events.is_mx & events.noaa_full_id_candidate.isna()).sum()),
        "forecast_cases": len(cases), "case_horizon_rows": len(labels), "training_ready": False,
        "upstream_labels_modified": False, "output_sha256": {p.name: digest(p) for p in args.output_dir.iterdir()},
        "limitations": ["Region decisions are conservative retrospective source adjudications, not definitive physical identification.",
            "Science-calibrated classes and reported start times are preserved; this is not an operational-class label rebuild.",
            "Other known-region events outside the queue are unchanged and have not all received daily-report review.",
            "Unknown-region M/X events mask otherwise-negative windows globally; these flags do not mean each target flared.",
            "Nominal archive spans do not establish continuous observation or event-detection completeness.",
            "Training, calibration, test splits, final target scope and historical availability are not certified."]}
    (args.output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
