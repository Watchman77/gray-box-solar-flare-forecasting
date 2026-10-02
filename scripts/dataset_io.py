"""Load the aligned package without coercing unknown outcomes into non-events."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(path, horizon=72, scope="primary", verify=False):
    root = Path(path)
    if horizon not in (48, 72) or scope not in ("primary", "patch"):
        raise ValueError("Choose horizon 48/72 and scope primary/patch")
    manifest = json.loads((root / "manifest.json").read_text())
    if verify:
        for name, item in manifest["outputs"].items():
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if actual != item["sha256"]:
                raise ValueError(f"Package checksum mismatch: {name}")
    sharp = np.load(root / "sharp.npy", mmap_mode="r", allow_pickle=False)
    missing = np.load(root / "sharp_missing.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(root / f"y_{scope}_{horizon}h.npy", allow_pickle=False)
    known = np.load(root / f"known_{scope}_{horizon}h.npy", allow_pickle=False)
    index = pd.read_csv(root / "input_index.csv.gz")
    if tuple(manifest["sharp_shape"]) != sharp.shape or missing.shape != sharp.shape:
        raise ValueError("Tensor dimensions do not match the manifest")
    if len(index) != len(sharp) or index.forecast_case_id.duplicated().any() or not np.array_equal(index.tensor_row, np.arange(len(sharp))):
        raise ValueError("Case/tensor row alignment mismatch")
    if y.shape != (len(sharp),) or not np.isin(y, [-1, 0, 1]).all() or not np.array_equal(known, y != -1):
        raise ValueError("Label/mask mismatch")
    return {"sharp": sharp, "missing": missing, "labels": y, "known": known,
            "index": index, "manifest": manifest}
