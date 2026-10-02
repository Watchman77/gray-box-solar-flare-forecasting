"""Load the aligned package without coercing unknown outcomes into non-events."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path, horizon=72, scope="primary", verify=True):
    root = Path(path)
    if horizon not in (48, 72) or scope not in ("primary", "patch"):
        raise ValueError("Choose horizon 48/72 and scope primary/patch")
    manifest = json.loads((root / "manifest.json").read_text())
    required = {"sharp.npy", "sharp_missing.npy", "input_index.csv.gz", "master_cases.csv.gz",
                f"y_{scope}_{horizon}h.npy", f"known_{scope}_{horizon}h.npy"}
    if not required.issubset(manifest["outputs"]):
        raise ValueError("Manifest does not pin every required model input")
    for name in required:
        if not (root / name).is_file():
            raise ValueError(f"Required dataset file missing: {name}")
    if verify:
        for name, item in manifest["outputs"].items():
            if not (root / name).is_file():
                raise ValueError(f"Required dataset file missing: {name}")
            actual = file_sha256(root / name)
            if actual != item["sha256"]:
                raise ValueError(f"Package checksum mismatch: {name}")
    sharp = np.load(root / "sharp.npy", mmap_mode="r", allow_pickle=False)
    missing = np.load(root / "sharp_missing.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(root / f"y_{scope}_{horizon}h.npy", allow_pickle=False)
    known = np.load(root / f"known_{scope}_{horizon}h.npy", allow_pickle=False)
    index = pd.read_csv(root / "input_index.csv.gz")
    if tuple(manifest["sharp_shape"]) != sharp.shape or missing.shape != sharp.shape:
        raise ValueError("Tensor dimensions do not match the manifest")
    if sharp.ndim != 3 or sharp.shape[1:] != (len(manifest["history_lag_native_minutes"]), len(manifest["feature_order"])):
        raise ValueError("Feature/history dimensions differ from their declared order")
    if str(sharp.dtype) != manifest["sharp_dtype"] or not np.issubdtype(sharp.dtype, np.floating):
        raise ValueError("Unexpected SHARP numeric type")
    if missing.dtype != np.bool_ or not np.array_equal(missing, ~np.isfinite(sharp)):
        raise ValueError("Feature missingness mask differs from actual values")
    if len(index) != len(sharp) or index.forecast_case_id.isna().any() or index.forecast_case_id.duplicated().any() or not np.array_equal(index.tensor_row, np.arange(len(sharp))):
        raise ValueError("Case/tensor row alignment mismatch")
    if known.dtype != np.bool_ or y.shape != (len(sharp),) or not np.isin(y, [-1, 0, 1]).all() or not np.array_equal(known, y != -1):
        raise ValueError("Label/mask mismatch")
    return {"sharp": sharp, "missing": missing, "labels": y, "known": known,
            "index": index, "manifest": manifest}


def preflight_dataset(path, horizon=72, scope="primary", purpose="exploratory"):
    """Validate all target variants once; uncertainty remains explicit, not an error."""
    if purpose not in ("exploratory", "confirmatory"):
        raise ValueError("Unknown experiment purpose")
    root = Path(path)
    data = load_dataset(root, horizon, scope, verify=True)
    manifest, index = data["manifest"], data["index"]
    if purpose == "confirmatory" and not manifest["operational_training_ready"]:
        raise ValueError("This candidate package is exploratory; final label/availability clearance is not established")
    master = pd.read_csv(root / "master_cases.csv.gz", low_memory=False)
    if len(master) != manifest["case_count"] or master.forecast_case_id.isna().any() or master.forecast_case_id.duplicated().any():
        raise ValueError("Master-case keys/counts are inconsistent")
    if not master.inputs_available.eq(master.tensor_row.ge(0)).all():
        raise ValueError("Input availability differs from tensor pointers")
    matched = master.loc[master.tensor_row.ge(0)].sort_values("tensor_row")
    if not np.array_equal(matched.tensor_row, np.arange(len(index))) or not np.array_equal(matched.forecast_case_id, index.forecast_case_id):
        raise ValueError("Master-case and tensor-index mapping differ")
    for col in ("source_sample_id", "region_component_id"):
        if matched[col].isna().any() or not np.array_equal(matched[col], index[col]):
            raise ValueError(f"Master/index identity mismatch: {col}")
    issue = pd.to_datetime(index.issue_utc, utc=True, format="ISO8601")
    if issue.isna().any() or not np.array_equal(issue, pd.to_datetime(matched.issue_utc, utc=True, format="ISO8601")):
        raise ValueError("Missing or mismatched issue times")
    history = []
    for lag in manifest["history_lag_native_minutes"]:
        uri = f"history_uri_tminus{lag}"
        if index[uri].isna().any() or not index[uri].str.startswith("gs://").all() or not np.array_equal(index[uri], matched[uri]):
            raise ValueError(f"Missing or mismatched AIA reference: {uri}")
        times = pd.to_datetime(index[f"history_{lag}_UTC"], utc=True, format="ISO8601")
        if times.isna().any() or not times.lt(issue).all():
            raise ValueError("Missing or future predictor timestamp")
        if not np.array_equal(times, pd.to_datetime(matched[f"history_{lag}_UTC"], utc=True, format="ISO8601")):
            raise ValueError("Master/index history timestamp mismatch")
        history.append(times.astype("int64").to_numpy())
    if not (np.diff(np.column_stack(history), axis=1) > 0).all():
        raise ValueError("History slots are not chronologically ordered")
    targets = []
    for h in (48, 72):
        end = pd.to_datetime(matched[f"outcome_end_utc_{h}h"], utc=True, format="ISO8601").to_numpy()
        if pd.isna(end).any() or not (end > issue.to_numpy()).all():
            raise ValueError("Invalid outcome endpoint")
        for s in ("primary", "patch"):
            y = np.load(root / f"y_{s}_{h}h.npy", allow_pickle=False)
            known = np.load(root / f"known_{s}_{h}h.npy", allow_pickle=False)
            col = matched[f"candidate_{s}_label_{h}h"]
            if not col.dropna().isin([0, 1]).all() or not np.array_equal(col.fillna(-1).to_numpy(), y):
                raise ValueError(f"Master/array outcome mismatch: {h}/{s}")
            if known.dtype != np.bool_ or not np.array_equal(known, y != -1):
                raise ValueError(f"Unknown outcome mask mismatch: {h}/{s}")
            targets.append({"horizon": h, "scope": s, "known": int(known.sum()), "unknown": int((~known).sum())})
    data["master"] = master
    data["preflight"] = {"status": "passed", "purpose": purpose, "case_count": len(master),
        "tensor_count": len(index), "manifest_sha256": file_sha256(root / "manifest.json"),
        "checks": ["all package hashes", "master/tensor/label alignment", "four label masks", "feature masks",
                   "identity and AIA references", "past-only ordered histories", "outcome endpoints"],
        "targets": targets, "aia_remote_object_reads": 0,
        "remaining_limits": ["AIA pixel files are referenced, not downloaded or decoded in this preflight.",
            "Candidate zeros do not certify continuous observation coverage or historical delivery."]}
    return data
