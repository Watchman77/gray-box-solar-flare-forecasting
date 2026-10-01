"""Build the canonical candidate universe and join verified upstream input pointers.

This manifest retains excluded cases. It assigns no new train/test roles, fits no
model, and does not certify historical operational availability.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.prepare_dataset_sources import CANONICAL_SHA256, digest
except ModuleNotFoundError:
    from prepare_dataset_sources import CANONICAL_SHA256, digest

COHORTS = ("cycle24", "primary_cycle25", "supplementary_2026")


def tai_to_utc(values, leap_file):
    """Convert native TAI with pinned IERS epochs; reject unrepresentable leap instants."""
    table = pd.read_csv(leap_file, sep=r"\s+", comment="#", header=None,
                        names=["mjd", "day", "month", "year", "offset"])
    text = Path(leap_file).read_text()
    expiry_line = next(line for line in text.splitlines() if "File expires on" in line)
    expiry = pd.Timestamp(expiry_line.split("File expires on", 1)[1].strip())
    utc_epochs = pd.to_datetime(table[["year", "month", "day"]])
    tai_epochs = utc_epochs + pd.to_timedelta(table.offset, unit="s")
    native = pd.to_datetime(values, format="%Y.%m.%d_%H:%M:%S_TAI", errors="raise")
    if native.isna().any():
        raise ValueError("Missing native TAI time")
    integers = native.to_numpy(dtype="datetime64[ns]").astype("int64")
    starts = tai_epochs.to_numpy(dtype="datetime64[ns]").astype("int64")
    pos = np.searchsorted(starts, integers, side="right") - 1
    if (pos < 0).any() or (native >= expiry).any():
        raise ValueError("Native time outside the pinned leap-table scope")
    for i in range(1, len(starts)):
        jump = int(table.offset.iloc[i] - table.offset.iloc[i - 1]) * 1_000_000_000
        if jump > 0 and ((integers >= starts[i] - jump) & (integers < starts[i])).any():
            raise ValueError("An inserted leap second needs a leap-capable time representation")
    return pd.Series(pd.to_datetime(integers - table.offset.to_numpy()[pos] * 1_000_000_000, utc=True), index=values.index)


def require_unique(frame, columns, label):
    if frame[columns].isna().any().any() or frame.duplicated(columns).any():
        raise ValueError(f"Missing or duplicate {label}: {columns}")


def verified_upstream(aia_root):
    root = aia_root / "inputs/sharp16_matched_v1_20261001"
    receipt = json.loads((root / "build_receipt.json").read_text())
    review = json.loads((root / "independent_review.json").read_text())
    if review["status"] != "SHARP16_FULL_INDEPENDENTLY_VERIFIED" or digest(root / "build_receipt.json") != review["build_receipt_sha256"]:
        raise ValueError("Upstream acceptance receipt mismatch")
    inventories, parts = [], []
    for cohort in COHORTS:
        for suffix in ("index.csv.gz", "native_slot_ledger.csv.gz", "sharp16_raw_three_slot.npy"):
            rel = f"{cohort}/{suffix}"
            path = root / rel
            actual = digest(path)
            if actual != receipt["output_hashes"][rel]:
                raise ValueError(f"Upstream content changed: {rel}")
            inventories.append({"relative_to_aia_root": str(path.relative_to(aia_root)), "sha256": actual, "bytes": path.stat().st_size})
        index = pd.read_csv(root / cohort / "index.csv.gz", float_precision="round_trip")
        require_unique(index, ["target_sample_id"], f"{cohort} target")
        require_unique(index, ["sharp16_row"], f"{cohort} tensor row")
        data = np.load(root / cohort / "sharp16_raw_three_slot.npy", mmap_mode="r", allow_pickle=False)
        if data.shape != (len(index), 3, 16) or not np.isfinite(data).all():
            raise ValueError("SHARP tensor shape or finite-value failure")
        if not np.array_equal(index.sharp16_row.to_numpy(), np.arange(len(index))):
            raise ValueError("Tensor row mapping is not the accepted compact order")
        index["input_cohort"] = cohort
        parts.append(index)
    merged = pd.concat(parts, ignore_index=True)
    require_unique(merged, ["target_sample_id"], "combined target")
    return merged, inventories, receipt["feature_order"]


def write_csv(frame, path):
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aia-root", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists():
        raise ValueError("Output version already exists; use a new directory to preserve evidence")
    source = args.source_dir / "canonical_1790079135531166.csv"
    if digest(source) != CANONICAL_SHA256:
        raise ValueError("Unexpected canonical snapshot")
    raw = pd.read_csv(source, float_precision="round_trip", low_memory=False)
    require_unique(raw, ["sample_id"], "canonical sample")
    require_unique(raw, ["HARPNUM", "T_REC"], "canonical physical record")
    if not raw.label_48h_final.isin([0, 1]).all():
        raise ValueError("Invalid legacy labels; never coerce them silently")
    accepted, source_inventory, features = verified_upstream(args.aia_root)
    if not set(accepted.target_sample_id).issubset(set(raw.sample_id)):
        raise ValueError("Accepted upstream IDs absent from canonical snapshot")

    frame = raw[["sample_id", "HARPNUM", "NOAA_AR_clean", "NOAA_ARS", "T_REC", "QUALITY", "label_48h_final"]].copy()
    frame = frame.rename(columns={"sample_id": "source_sample_id", "T_REC": "issue_tai", "label_48h_final": "upstream_label_48h"})
    frame.insert(0, "forecast_case_id", "gb72-native-v1:" + frame.source_sample_id)
    frame["issue_utc"] = tai_to_utc(frame.issue_tai, args.source_dir / "Leap_Second.dat")
    native_end = pd.to_datetime(frame.issue_tai, format="%Y.%m.%d_%H:%M:%S_TAI") + pd.Timedelta(hours=72)
    frame["outcome_end_utc"] = tai_to_utc(native_end.dt.strftime("%Y.%m.%d_%H:%M:%S_TAI"), args.source_dir / "Leap_Second.dat")
    frame["horizon_hours"] = 72
    frame["new_experiment_role"] = "unassigned"
    frame["forecast_schedule_status"] = "native_source_grid_candidate_not_daily_schedule"
    frame["historical_availability_status"] = "not_established"
    frame["goes_predictor_status"] = "pending_source_quality_and_cutoff_contract"
    frame["operational_training_ready"] = False

    cols = ["target_sample_id", "HARPNUM", "NOAA_AR_clean", "region_component_id", "label_48h_final", "role", "issue_utc", "sharp16_row", "input_cohort"]
    cols += [f"history_uri_tminus{lag}" for lag in (288, 192, 96)]
    accepted = accepted[cols].rename(columns={"target_sample_id": "source_sample_id", "HARPNUM": "matched_harp", "NOAA_AR_clean": "matched_noaa", "label_48h_final": "matched_label48", "role": "upstream_role_48h", "issue_utc": "matched_issue_utc"})
    frame = frame.merge(accepted, on="source_sample_id", how="left", validate="one_to_one", sort=False)
    matched = frame.input_cohort.notna()
    for canonical_col, joined_col in [("HARPNUM", "matched_harp"), ("NOAA_AR_clean", "matched_noaa"), ("upstream_label_48h", "matched_label48")]:
        if not (frame.loc[matched, canonical_col].to_numpy() == frame.loc[matched, joined_col].to_numpy()).all():
            raise ValueError(f"Accepted/canonical discrepancy: {canonical_col}")
    if not (frame.loc[matched, "issue_utc"].to_numpy() == pd.to_datetime(frame.loc[matched, "matched_issue_utc"], utc=True, format="mixed").to_numpy()).all():
        raise ValueError("UTC conversion differs from accepted upstream issue times")
    frame["upstream_matched_three_slot_inputs"] = matched
    frame["region_mapping_origin"] = np.where(matched, "accepted_input_index", "unresolved")

    # Preserve known component identities for candidates excluded from fusion.
    reservations_path = args.aia_root / "inputs/2026_readiness/snapshot/region_reservations.csv"
    reservations = pd.read_csv(reservations_path, dtype=str)
    harp_map = {}
    for row in reservations.itertuples():
        for value in row.HARPNUM_values.split(";"):
            harp = int(value)
            if harp in harp_map and harp_map[harp] != row.component_id:
                raise ValueError("Conflicting HARP/component reservation")
            harp_map[harp] = row.component_id
    proposed = frame.HARPNUM.map(harp_map)
    comparable = matched & proposed.notna()
    if (proposed[comparable] != frame.loc[comparable, "region_component_id"]).any():
        raise ValueError("Reservation component differs from accepted index")
    use = ~matched & proposed.notna()
    frame.loc[use, "region_component_id"] = proposed[use]
    frame.loc[use, "region_mapping_origin"] = "upstream_harp_reservation"
    reasons_path = args.aia_root / "archives/2026_readiness_20260930/reconciled_v2/candidate_exclusions.csv.gz"
    reasons = pd.read_csv(reasons_path).rename(columns={"target_sample_id": "source_sample_id", "region_component_id": "extension_component", "exclusion_reasons": "upstream_exclusion_reasons"})
    require_unique(reasons, ["source_sample_id"], "extension exclusion")
    frame = frame.merge(reasons[["source_sample_id", "extension_component", "upstream_exclusion_reasons"]], on="source_sample_id", how="left", validate="one_to_one", sort=False)
    both = frame.region_component_id.notna() & frame.extension_component.notna()
    if (frame.loc[both, "region_component_id"] != frame.loc[both, "extension_component"]).any():
        raise ValueError("Extension component identity conflict")
    use = frame.region_component_id.isna() & frame.extension_component.notna()
    frame.loc[use, "region_component_id"] = frame.loc[use, "extension_component"]
    frame.loc[use, "region_mapping_origin"] = "upstream_extension_exclusion_manifest"

    mapping_path = args.aia_root / "inputs/goes_cutoff_time_audit_20261001/target_native_utc_map.csv.gz"
    mapping = pd.read_csv(mapping_path).rename(columns={"target_sample_id": "source_sample_id"})
    require_unique(mapping, ["source_sample_id"], "time map")
    if set(mapping.source_sample_id) != set(accepted.source_sample_id):
        raise ValueError("Time map and accepted input population differ")
    for lag in (288, 192, 96):
        converted = tai_to_utc(mapping[f"history_{lag}_TAI"], args.source_dir / "Leap_Second.dat")
        if not (converted.to_numpy() == pd.to_datetime(mapping[f"history_{lag}_UTC"], utc=True).to_numpy()).all():
            raise ValueError("Historical UTC time replay differs")
    timecols = ["source_sample_id"] + [f"history_{lag}_UTC" for lag in (288, 192, 96)]
    frame = frame.merge(mapping[timecols], on="source_sample_id", how="left", validate="one_to_one", sort=False)
    frame["matched_history_contract"] = np.where(frame.upstream_matched_three_slot_inputs, "three_native_slots_at_minus288_minus192_minus96_minutes", "not_assembled")
    frame["source_noaa_multiple"] = frame.NOAA_ARS.astype(str).str.contains(",|;", regex=True)
    frame["component_issue_multiplicity"] = frame.groupby(["region_component_id", "issue_utc"])["source_sample_id"].transform("size")
    frame["unmatched_reason_status"] = np.where(frame.upstream_matched_three_slot_inputs, "not_applicable",
        np.where(frame.upstream_exclusion_reasons.notna(), "upstream_extension_reasons_recorded", "not_yet_reconciled"))
    frame = frame.drop(columns=["matched_harp", "matched_noaa", "matched_label48", "matched_issue_utc", "extension_component"])
    frame["sharp16_row"] = frame.sharp16_row.astype("Int64")
    frame = frame.sort_values(["issue_utc", "HARPNUM", "source_sample_id"], kind="stable").reset_index(drop=True)
    yearly = frame.assign(year=frame.issue_utc.dt.year).groupby("year").agg(
        canonical_candidates=("source_sample_id", "size"),
        existing_matched_inputs=("upstream_matched_three_slot_inputs", "sum"),
        components_mapped=("region_component_id", "count"),
    ).reset_index()
    summary = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_inventory_built_not_training_ready",
        "canonical_candidates": len(frame), "existing_matched_inputs": int(frame.upstream_matched_three_slot_inputs.sum()),
        "retained_without_matched_inputs": int((~frame.upstream_matched_three_slot_inputs).sum()),
        "unmapped_components": int(frame.region_component_id.isna().sum()),
        "multi_noaa_candidate_rows": int(frame.source_noaa_multiple.sum()),
        "shared_component_issue_rows": int((frame.component_issue_multiplicity > 1).sum()),
        "shared_component_issue_groups": int(frame.loc[frame.component_issue_multiplicity > 1, ["region_component_id", "issue_utc"]].drop_duplicates().shape[0]),
        "first_issue_utc": frame.issue_utc.min().isoformat(), "last_issue_utc": frame.issue_utc.max().isoformat(),
        "latest_required_72h_outcome_utc": frame.outcome_end_utc.max().isoformat(),
        "feature_order": features, "source_sha256": CANONICAL_SHA256,
        "leap_table_sha256": digest(args.source_dir / "Leap_Second.dat"),
        "upstream_build_receipt_sha256": digest(args.aia_root / "inputs/sharp16_matched_v1_20261001/build_receipt.json"),
        "upstream_review_sha256": digest(args.aia_root / "inputs/sharp16_matched_v1_20261001/independent_review.json"),
        "verified_upstream_files": source_inventory,
        "yearly": yearly.to_dict(orient="records"),
        "checks": {"canonical_ids_unique": True, "no_accepted_cases_lost": True,
            "matched_identifiers_and_48h_labels_preserved": True, "accepted_issue_and_history_utc_replayed": True},
        "limitations": [
            "Canonical candidates are an upstream-curated universe, not all observable solar regions.",
            "The native issue grid is retained; a daily operational issuance schedule is not frozen.",
            "Matched histories contain three past states, not the old paper's seven daily states.",
            "Existing labels/roles remain 48-hour upstream evidence; new 72-hour roles are unassigned.",
            "Distinct HARP records can share a component and issue time; a component-level prediction requires an explicit consolidation rule.",
            "Unmatched candidates include protocol exclusions as well as missing inputs; they are not an outage-rate estimate.",
            "No continuous GOES features or operational availability clearance is supplied.",
        ],
    }
    out.mkdir(parents=True)
    write_csv(frame, out / "forecast_case_inventory.csv.gz")
    yearly.to_csv(out / "yearly_inventory.csv", index=False)
    source_inventory += [{"relative_to_aia_root": str(p.relative_to(args.aia_root)), "sha256": digest(p), "bytes": p.stat().st_size}
                         for p in (reservations_path, reasons_path, mapping_path)]
    summary["manifest_sha256"] = digest(out / "forecast_case_inventory.csv.gz")
    (out / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("verified_upstream_files", "yearly", "feature_order")}, indent=2))


if __name__ == "__main__":
    main()
