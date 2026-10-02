"""Independently reconcile saved preparation tables without calling their builder."""

import argparse
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd


def verify(root, output):
    root, output = Path(root), Path(output)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = json.loads((output / "notebook_receipt.json").read_text())
    for name, digest in receipt["artifacts"].items():
        assert sha(output / name) == digest, name
    cases = pd.read_csv(output / "multimodal_case_manifest.csv.gz", low_memory=False)
    selected = pd.read_csv(output / "candidate_comparison_cases.csv.gz", low_memory=False)
    parent = pd.read_csv(root / "outputs/sharp72_notebook_v1_20261002T080801721986Z/split_manifest.csv.gz")
    base = pd.read_csv(root / "data/processed/training_snapshot_20261002/gray_box_aligned_v1/master_cases.csv.gz", low_memory=False)
    assert len(cases) == len(base) and cases.forecast_case_id.tolist() == base.forecast_case_id.tolist()
    assert cases.label.tolist() == base.candidate_primary_label_72h.fillna(-1).astype(int).tolist()
    assert cases.label_known.tolist() == base.label_known_primary_72h.tolist()
    assert cases.issue_tai.tolist() == base.issue_tai.tolist()
    matched = cases[cases.tensor_row.ge(0)].sort_values("tensor_row")
    for field in ["forecast_case_id", "role", "label", "label_known"]:
        assert matched[field].tolist() == parent[field].tolist(), field
    config = json.loads((root / "configs/sharp72_temporal_v1.json").read_text())
    issue = pd.to_datetime(cases.issue_utc, utc=True, format="ISO8601")
    end = pd.to_datetime(cases.outcome_end_utc_72h, utc=True, format="ISO8601")
    leap_path = root / "data/raw/gray_box_inventory_v1/Leap_Second.dat"
    leaps = pd.read_csv(leap_path, sep=r"\s+", comment="#", header=None,
                        names=["mjd", "day", "month", "year", "offset"])
    epochs = pd.to_datetime(leaps[["year", "month", "day"]], utc=True).astype("int64").to_numpy()
    offsets = leaps.offset.to_numpy()
    def utc_to_tai_ns(stamps):
        values = stamps.astype("int64").to_numpy()
        pos = np.searchsorted(epochs, values, side="right") - 1
        assert (pos >= 0).all()
        return values + offsets[pos] * 1_000_000_000
    native = pd.to_datetime(cases.issue_tai, format="%Y.%m.%d_%H:%M:%S_TAI").astype("int64").to_numpy()
    assert np.array_equal(utc_to_tai_ns(issue), native)
    assert ((utc_to_tai_ns(end) - native) == 72 * 3600 * 1_000_000_000).all()
    # The reporting-delay convention is inherited exactly from the frozen parent.
    expected = np.full(len(cases), "outside_blocks", dtype=object)
    for block in config["blocks"]:
        stop = pd.Timestamp(block["end"], tz="UTC")
        mask = (issue >= pd.Timestamp(block["start"], tz="UTC")) & (issue < stop)
        expected[mask] = "purged_outcome_boundary"
        expected[mask & ((end + pd.Timedelta(hours=config["reporting_delay_hours"])) <= stop)] = block["role"]
    assert expected.tolist() == cases.temporal_role.tolist()
    wanted = (cases.label_known & cases.sharp_input_finite & cases.aia_references_present &
              cases.aia_history_past_and_ordered & cases.temporal_role.isin([b["role"] for b in config["blocks"]]))
    assert wanted.tolist() == cases.candidate_sharp_aia_comparison.tolist()
    assert cases.loc[wanted, "forecast_case_id"].tolist() == selected.forecast_case_id.tolist()
    images = pd.read_csv(output / "aia_object_requests.csv.gz")
    refs = selected[["history_uri_tminus288", "history_uri_tminus192", "history_uri_tminus96"]].stack()
    assert images.uri.is_unique and set(images.uri) == set(refs)
    assert images.set_index("uri").case_uses.to_dict() == refs.value_counts().to_dict()
    goes = pd.read_csv(output / "goes_case_requests.csv.gz")
    assert goes.forecast_case_id.tolist() == selected.forecast_case_id.tolist()
    assert goes.maximum_input_observation_utc.tolist() == selected.issue_utc.tolist()
    notebook_path = root / "notebooks/07_SHARP_AIA_GOES_72h_Integration.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(cells) == 8 and [c.execution_count for c in cells] == list(range(1, 9))
    assert not any(o.output_type == "error" for c in cells for o in c.outputs)
    report = {
        "status": "independent_numeric_verification_passed", "output_relative_to_repository": str(output.relative_to(root)),
        "population_rows": len(cases), "candidate_rows": len(selected), "parent_rows": len(parent),
        "aia_unique_objects": len(images), "aia_references": len(refs), "goes_case_requests": len(goes),
        "leap_second_crossing_windows": int(((end - issue) != pd.Timedelta(hours=72)).sum()),
        "notebook_cells_executed": len(cells), "notebook_sha256": sha(notebook_path),
        "contract_sha256": sha(output / "experiment_contract.json"), "summary_sha256": sha(output / "summary.json"),
        "leap_table_sha256": sha(leap_path), "verifier_sha256": sha(Path(__file__)),
        "checks": ["saved artifact hashes", "full-population identity and labels", "parent SHARP roles/labels",
                   "native TAI 72h endpoints using pinned leap table", "independent temporal role arithmetic",
                   "candidate inclusion flags", "deduplicated AIA counts", "GOES cutoff identity", "notebook schema/execution"],
        "limitations": ["Preparation only; no new model training", "AIA reference presence is not pixel verification",
                        "GOES requests are not predictor measurements", "Historical availability and continuous outcome coverage unverified"]}
    result = root / "results/multimodal72_preparation_20261002"
    result.mkdir(exist_ok=True)
    (result / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in ["experiment_contract.json", "role_support.csv", "annual_support.csv", "upstream_source_snapshot.json"]:
        (result / name).write_bytes((output / name).read_bytes())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parents[1], args.output.resolve()), indent=2))
