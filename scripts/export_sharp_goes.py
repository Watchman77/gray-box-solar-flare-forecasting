"""Export a SHARP/GOES view of the aligned cohort without AIA columns or pixels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.dataset_io import file_sha256, preflight_dataset
except ModuleNotFoundError:
    from dataset_io import file_sha256, preflight_dataset


def export(dataset, output):
    if output.exists():
        raise ValueError("Choose a new output directory; existing exports are preserved")
    data = preflight_dataset(dataset)
    source, manifest = data["master"], data["manifest"]
    columns = ["forecast_case_id", "issue_utc", "issue_tai", "HARPNUM",
               "NOAA_AR_clean", "NOAA_ARS", "region_component_id", "source_cohort",
               "QUALITY", "operational_training_ready", "continuous_outcome_coverage_verified"]
    for horizon in (48, 72):
        columns.append(f"outcome_end_utc_{horizon}h")
        for scope in ("primary", "patch"):
            columns += [f"candidate_{scope}_label_{horizon}h", f"label_known_{scope}_{horizon}h"]
    table = source[columns].copy()
    # This is a view of the existing matched cohort, not new SHARP acquisition.
    table["sharp_history_available"] = source.tensor_row.ge(0)
    matched = table.sharp_history_available.to_numpy()
    rows = source.loc[matched, "tensor_row"].to_numpy(dtype=int)
    features = {}
    for slot, lag in enumerate(manifest["history_lag_native_minutes"]):
        table[f"history_{lag}_UTC"] = source[f"history_{lag}_UTC"]
        for j, name in enumerate(manifest["feature_order"]):
            values = np.full(len(table), np.nan)
            values[matched] = data["sharp"][rows, slot, j]
            features[f"{name}_tminus{lag}"] = values
    table = pd.concat([table, pd.DataFrame(features)], axis=1)
    output.mkdir(parents=True)
    table.to_csv(output / "sharp_goes.csv", index=False)

    # A viewing sample spanning years, availability and primary 72h outcomes.
    # It is deliberately stratified and must not be used for prevalence estimates.
    strata = pd.DataFrame({
        "year": pd.to_datetime(table.issue_utc, utc=True, format="ISO8601").dt.year,
        "available": matched, "target": table.candidate_primary_label_72h.fillna(-1)})
    selected = strata.groupby(["year", "available", "target"], sort=True).head(2).index
    latest_lag = manifest["history_lag_native_minutes"][-1]
    preview_columns = ["issue_utc", f"history_{latest_lag}_UTC", "HARPNUM", "NOAA_ARS",
                       "source_cohort", "sharp_history_available",
                       "candidate_primary_label_48h", "label_known_primary_48h",
                       "candidate_primary_label_72h", "label_known_primary_72h"]
    preview_columns += [f"{name}_tminus{latest_lag}" for name in manifest["feature_order"]]
    table.loc[selected, preview_columns].sort_values("issue_utc").to_csv(output / "sample.csv", index=False)

    readback = pd.read_csv(output / "sharp_goes.csv", low_memory=False)
    pd.testing.assert_frame_equal(table, readback, check_dtype=False, rtol=1e-12, atol=1e-12)
    (output / "README.txt").write_text(
        "SHARP-GOES view, without AIA image paths or pixels\n\n"
        "sharp_goes.csv: all forecast candidates and all three SHARP history slots.\n"
        "sample.csv: viewing sample with the most recent SHARP history slot only; not a prevalence sample.\n"
        "Features are raw and have explicit native-minute offsets (288, 192, 96).\n"
        "issue_utc is forecast issue time; history_*_UTC records physical observation times.\n"
        "GOES supplies event outcomes, not a continuous X-ray flux predictor.\n"
        "Targets: 1 = M/X start in (issue, outcome end], 0 = provisional no-event, blank = unknown.\n"
        "primary = primary NOAA region; patch = union of NOAA regions on the current HARP record.\n"
        "Known-label flags do not certify complete outcome observation coverage.\n"
        "Blank features mean histories were not assembled. No values or labels are invented.\n"
        "This preserves the aligned cohort's sampling and missingness; it is not an independently expanded SHARP-only cohort.\n"
        "The uploaded daily 2010-2025 sample is a reference only; it has not been merged or overwritten.\n"
        "Its FLARE_BINARY horizon is unspecified; this export uses explicit 48h/72h targets.\n"
        "The current feature set includes MEANJZH but not AREA_ACR, unlike that reference.\n"
        "These remain candidate research data for exploratory use.\n")
    receipt = {
        "source_manifest_sha256": file_sha256(dataset / "manifest.json"),
        "export_script_sha256": file_sha256(Path(__file__)),
        "rows": len(table), "columns": len(table.columns), "sample_rows": len(selected),
        "sharp_history_available": int(matched.sum()), "missing_history_cases": int((~matched).sum()),
        "issue_utc_min": table.issue_utc.min(), "issue_utc_max": table.issue_utc.max(),
        "contains_aia_columns_or_pixels": False,
        "population": "unchanged aligned cohort; no independent SHARP-only population expansion",
        "roundtrip_verification": "all cells compared with source projection",
        "outputs": {name: {"bytes": (output / name).stat().st_size,
                           "sha256": file_sha256(output / name)}
                    for name in ("sharp_goes.csv", "sample.csv", "README.txt")}}
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.dataset, args.output_dir)
