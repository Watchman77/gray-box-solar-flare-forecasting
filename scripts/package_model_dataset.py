"""Assemble existing verified SHARP, AIA references and GOES-derived labels.

No image download or event adjudication. Unknown labels use -1 in arrays and
remain nullable in the master table. All candidate cases are retained.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

try:
    from scripts.prepare_dataset_sources import digest
    from scripts.build_dataset_inventory import require_unique, write_csv
except ModuleNotFoundError:
    from prepare_dataset_sources import digest
    from build_dataset_inventory import require_unique, write_csv

COHORTS = ("cycle24", "primary_cycle25", "supplementary_2026")
LAGS = (288, 192, 96)


def merge_labels(cases, labels):
    require_unique(cases, ["forecast_case_id"], "forecast case")
    require_unique(labels, ["forecast_case_id", "horizon_hours"], "case/horizon")
    if set(labels.horizon_hours) != {48, 72}:
        raise ValueError("Expected both 48 and 72 hours")
    master = cases.drop(columns=["horizon_hours", "outcome_end_utc"]).copy()
    for horizon in (48, 72):
        part = labels.loc[labels.horizon_hours.eq(horizon)]
        if set(part.forecast_case_id) != set(master.forecast_case_id):
            raise ValueError("Every case must be present at both horizons")
        columns = ["forecast_case_id", "outcome_end_utc", "candidate_primary_label", "candidate_patch_label",
                   "primary_event_ids", "patch_event_ids", "unassigned_region_mx_event_ids", "nominal_span_contains_window"]
        renamed = part[columns].rename(columns={c: f"{c}_{horizon}h" for c in columns if c != "forecast_case_id"})
        master = master.merge(renamed, on="forecast_case_id", how="left", validate="one_to_one", sort=False)
        for scope in ("primary", "patch"):
            col = f"candidate_{scope}_label_{horizon}h"
            if not master[col].dropna().isin([0, 1]).all():
                raise ValueError("Invalid target label")
            master[col] = master[col].astype("Int64")
            master[f"label_known_{scope}_{horizon}h"] = master[col].notna()
    return master


def package(aia_root, inventory_dir, outcome_dir, output_dir):
    if output_dir.exists():
        raise ValueError("Package version exists; choose a new output directory")
    root = aia_root / "inputs/sharp16_matched_v1_20261001"
    receipt = json.loads((root / "build_receipt.json").read_text())
    review = json.loads((root / "independent_review.json").read_text())
    if review["status"] != "SHARP16_FULL_INDEPENDENTLY_VERIFIED" or review["build_receipt_sha256"] != digest(root / "build_receipt.json"):
        raise ValueError("SHARP source review/receipt mismatch")
    inv = json.loads((inventory_dir / "build_summary.json").read_text())
    outcome = json.loads((outcome_dir / "build_summary.json").read_text())
    inventory_path = inventory_dir / "forecast_case_inventory.csv.gz"
    if digest(inventory_path) != inv["manifest_sha256"] or digest(inventory_path) != outcome["inventory_sha256"]:
        raise ValueError("Inventory provenance mismatch")
    for name in ("candidate_outcomes.csv.gz", "adjudicated_events.csv.gz"):
        if digest(outcome_dir / name) != outcome["output_sha256"][name]:
            raise ValueError(f"Outcome provenance mismatch: {name}")
    cases = pd.read_csv(inventory_path, low_memory=False)
    labels = pd.read_csv(outcome_dir / "candidate_outcomes.csv.gz", low_memory=False)
    master = merge_labels(cases, labels)
    source_indices, tensors = [], []
    input_hashes = {"inventory": digest(inventory_path), "outcomes": digest(outcome_dir / "candidate_outcomes.csv.gz"),
                    "sharp_receipt": digest(root / "build_receipt.json"), "sharp_review": digest(root / "independent_review.json")}
    offset = 0
    for cohort in COHORTS:
        for filename in ("index.csv.gz", "sharp16_raw_three_slot.npy"):
            relative = f"{cohort}/{filename}"
            actual = digest(root / relative)
            if actual != receipt["output_hashes"][relative]:
                raise ValueError(f"SHARP source changed: {relative}")
            input_hashes[relative] = actual
        index = pd.read_csv(root / cohort / "index.csv.gz", low_memory=False)
        array = np.load(root / cohort / "sharp16_raw_three_slot.npy", mmap_mode="r", allow_pickle=False)
        require_unique(index, ["target_sample_id"], "SHARP target")
        if array.shape != (len(index), 3, len(receipt["feature_order"])) or not np.array_equal(np.sort(index.sharp16_row), np.arange(len(index))):
            raise ValueError("SHARP shape/row mapping mismatch")
        # Explicit compact row mapping; never use the earlier sparse sharp_row.
        tensors.append(np.asarray(array[index.sharp16_row.to_numpy(dtype=int)]))
        index["tensor_row"] = np.arange(offset, offset + len(index))
        index["source_cohort"] = cohort
        source_indices.append(index)
        offset += len(index)
    sources = pd.concat(source_indices, ignore_index=True).set_index("target_sample_id", drop=False)
    require_unique(sources, ["target_sample_id"], "combined SHARP target")
    matched = master.loc[master.upstream_matched_three_slot_inputs].copy()
    if set(matched.source_sample_id) != set(sources.index):
        raise ValueError("Matched inventory and tensor population differ")
    aligned = sources.loc[matched.source_sample_id]
    for col in ["HARPNUM", "NOAA_AR_clean", "region_component_id"]:
        if not np.array_equal(matched[col], aligned[col]):
            raise ValueError(f"Input identity mismatch: {col}")
    if not np.array_equal(pd.to_datetime(matched.issue_utc, utc=True), pd.to_datetime(aligned.issue_utc, utc=True, format="mixed")):
        raise ValueError("Issue time mismatch")
    for lag in LAGS:
        col = f"history_uri_tminus{lag}"
        if not np.array_equal(matched[col], aligned[col]) or not aligned[col].str.startswith("gs://").all():
            raise ValueError("AIA reference mismatch")
    master["tensor_row"] = master.source_sample_id.map(sources.tensor_row).fillna(-1).astype("int64")
    master["inputs_available"] = master.tensor_row.ge(0)
    master["source_cohort"] = master.input_cohort.fillna("unmatched")
    master["goes_use"] = "event_outcomes_only"
    master["dataset_status"] = "exploratory_candidate_labels"
    master["target_version"] = outcome["target_version"]
    master["continuous_outcome_coverage_verified"] = False
    # Temporal predictor matrix, in cohort/index order, independent of CSV order.
    sharp = np.concatenate(tensors, axis=0)
    missing = ~np.isfinite(sharp)
    output_dir.mkdir(parents=True)
    np.save(output_dir / "sharp.npy", sharp, allow_pickle=False)
    np.save(output_dir / "sharp_missing.npy", missing, allow_pickle=False)
    ordered = master.loc[master.inputs_available].sort_values("tensor_row")
    if not np.array_equal(ordered.tensor_row, np.arange(len(sharp))):
        raise ValueError("Final compact tensor mapping is not bijective")
    counts = []
    for horizon in (48, 72):
        for scope in ("primary", "patch"):
            col = f"candidate_{scope}_label_{horizon}h"
            y = ordered[col].fillna(-1).to_numpy(dtype=np.int8)
            np.save(output_dir / f"y_{scope}_{horizon}h.npy", y, allow_pickle=False)
            np.save(output_dir / f"known_{scope}_{horizon}h.npy", y != -1, allow_pickle=False)
            for cohort, part in ordered.groupby("source_cohort"):
                counts.append({"horizon_hours": horizon, "scope": scope, "cohort": cohort, "cases": len(part),
                    "positive": int(part[col].eq(1).sum()), "provisional_zero": int(part[col].eq(0).sum()),
                    "unknown": int(part[col].isna().sum())})
    write_csv(master, output_dir / "master_cases.csv.gz")
    write_csv(ordered[["forecast_case_id", "source_sample_id", "tensor_row", "source_cohort", "issue_utc",
        "region_component_id"] + [f"history_uri_tminus{lag}" for lag in LAGS] + [f"history_{lag}_UTC" for lag in LAGS]],
        output_dir / "input_index.csv.gz")
    shutil.copyfile(outcome_dir / "adjudicated_events.csv.gz", output_dir / "goes_events.csv.gz")
    pd.DataFrame(counts).to_csv(output_dir / "support.csv", index=False)
    # Small, executable reader included inside the portable archive.
    shutil.copyfile(Path(__file__).with_name("dataset_io.py"), output_dir / "dataset_io.py")
    (output_dir / "README.txt").write_text(
        "Gray-Box aligned dataset v1\n\n"
        "master_cases.csv.gz retains every candidate; tensor_row=-1 means inputs are not assembled.\n"
        "sharp.npy is raw float64 (case, history, feature); history order is t-288, t-192, t-96 native minutes.\n"
        "input_index.csv.gz maps every tensor row to its case ID and existing AIA GCS objects.\n"
        "AIA pixels remain in the existing bucket and are NOT copied into this package.\n"
        "y_*.npy uses -1 for unknown labels. known_*.npy is the corresponding outcome mask.\n"
        "sharp_missing.npy marks nonfinite features; no imputation or scaling has been fitted.\n"
        "goes_events.csv.gz contains the event catalogue used for labels, not a continuous XRS predictor.\n"
        "Labels are science-class start-time candidates; zeros have not been certified against continuous coverage.\n"
        "Use only for explicitly exploratory work until the remaining protocol choices are frozen.\n\n"
        "Quick start (numpy/pandas required):\n"
        "from dataset_io import load_dataset\n"
        "data = load_dataset('.', horizon=72, scope='primary')\n"
        "X, y, known = data['sharp'], data['labels'], data['known']\n"
        "# Apply known BEFORE passing y to any classifier.\n")
    outputs = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in sorted(output_dir.iterdir())}
    manifest = {"dataset_version": "gray_box_aligned_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(master), "tensor_case_count": len(sharp), "unmatched_cases_retained": int((~master.inputs_available).sum()),
        "sharp_shape": list(sharp.shape), "sharp_dtype": str(sharp.dtype), "sharp_missing_values": int(missing.sum()),
        "feature_order": receipt["feature_order"], "history_lag_native_minutes": list(LAGS),
        "first_issue_utc": str(pd.to_datetime(master.issue_utc, utc=True).min()),
        "last_issue_utc": str(pd.to_datetime(master.issue_utc, utc=True).max()),
        "target_version": outcome["target_version"], "unknown_label_value": -1,
        "aia_storage": "existing_gcs_references_no_pixel_copy", "goes_use": "future_event_labels_no_continuous_predictor",
        "new_experiment_roles": "unassigned", "operational_training_ready": False,
        "input_sha256": input_hashes, "outputs": outputs, "support": counts}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aia-root", "inventory-dir", "outcome-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = package(args.aia_root, args.inventory_dir, args.outcome_dir, args.output_dir)
    print(json.dumps({k: result[k] for k in ("dataset_version", "case_count", "tensor_case_count", "sharp_shape", "sharp_missing_values")}, indent=2))
