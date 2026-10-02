"""Expose the packaged arrays as a readable CSV, with a compact stratified preview."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def export(root):
    if (root / "dataset.csv").exists():
        raise ValueError("Readable export already exists")
    manifest = json.loads((root / "manifest.json").read_text())
    table = pd.read_csv(root / "master_cases.csv.gz", low_memory=False)
    sharp = np.load(root / "sharp.npy", mmap_mode="r", allow_pickle=False)
    matched = table.tensor_row.ge(0).to_numpy()
    row_indices = table.loc[matched, "tensor_row"].to_numpy(dtype=int)
    features = {}
    for slot, lag in enumerate(manifest["history_lag_native_minutes"]):
        for position, feature in enumerate(manifest["feature_order"]):
            values = np.full(len(table), np.nan)
            values[matched] = sharp[row_indices, slot, position]
            features[f"{feature}_tminus{lag}"] = values
    table = pd.concat([table, pd.DataFrame(features)], axis=1)
    table.to_csv(root / "dataset.csv", index=False)
    # At most two rows per year/input/label state; the preview is not a random
    # evaluation sample and must not be used to calculate prevalence.
    strata = pd.DataFrame({"year": pd.to_datetime(table.issue_utc, utc=True).dt.year,
        "inputs": table.inputs_available, "label": table.candidate_primary_label_72h.fillna(-1)})
    selected = strata.groupby(["year", "inputs", "label"], sort=True).head(2).index
    preview = table.loc[selected].sort_values("issue_utc")
    columns = ["source_sample_id", "issue_utc", "HARPNUM", "NOAA_AR_clean", "source_cohort", "inputs_available",
        "candidate_primary_label_48h", "candidate_primary_label_72h", "label_known_primary_72h"]
    columns += [f"{f}_tminus96" for f in manifest["feature_order"]]
    columns += ["history_uri_tminus288", "history_uri_tminus192", "history_uri_tminus96"]
    preview[columns].to_csv(root / "preview.csv", index=False)
    dictionary = [
        ("tensor_row", "Row in sharp.npy and label arrays; -1 means inputs unavailable in this package."),
        ("*_tminus288 / *_tminus192 / *_tminus96", "Raw SHARP parameters at the three native-time history offsets; 16 features per slot."),
        ("history_uri_tminus*", "Existing AIA image object in the user's Google Cloud bucket; pixels are not duplicated here."),
        ("candidate_primary_label_48h / 72h", "Science M/X event start within (issue, end] for primary NOAA region; 1 event, 0 provisional no-event, blank unresolved."),
        ("candidate_patch_label_48h / 72h", "Same convention over all NOAA IDs listed on the current HARP record."),
        ("label_known_*", "True when a candidate outcome is 0/1; does not certify continuous outcome coverage."),
        ("inputs_available", "True when the matched three-slot SHARP tensor and AIA references are assembled."),
        ("upstream_label_48h", "Preserved earlier AIA label; use explicit candidate target columns for this version."),
        ("unassigned_region_mx_event_ids_*", "Unresolved M/X associations inside the outcome window."),
        ("operational_training_ready", "False: historical delivery and definitive final labels are not certified."),
        ("new_experiment_role", "Unassigned: exploratory runner roles are stored separately."),
    ]
    pd.DataFrame(dictionary, columns=["field", "meaning"]).to_csv(root / "data_dictionary.csv", index=False)
    for name in ["dataset.csv", "preview.csv", "data_dictionary.csv"]:
        p = root / name
        manifest["outputs"][name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    manifest["readable_table"] = {"file": "dataset.csv", "rows": len(table), "columns": len(table.columns),
        "preview_file": "preview.csv", "preview_rows": len(preview), "preview_sampling": "up to two per year/input/primary72-label state"}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["readable_table"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    export(parser.parse_args().dataset)
