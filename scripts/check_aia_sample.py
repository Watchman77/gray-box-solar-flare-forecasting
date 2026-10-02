"""Bounded AIA I/O check using three existing objects, one per input cohort."""

import argparse
import base64
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

try:
    from scripts.dataset_io import file_sha256
    from scripts.aia_io import CHANNELS, load_aia_frame
except ModuleNotFoundError:
    from dataset_io import file_sha256
    from aia_io import CHANNELS, load_aia_frame


def check_frame(path, shape=(512, 512, 6)):
    return load_aia_frame(path, expected_shape=shape)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gcloud", default="gcloud")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.resume:
        raise ValueError("I/O check output exists")
    index = pd.read_csv(args.dataset / "input_index.csv.gz")
    # Latest case per cohort includes the actual 2026 extension endpoint.
    sample = index.sort_values("issue_utc").groupby("source_cohort", sort=True).tail(1)
    if len(sample) != 3:
        raise ValueError("Expected three inherited input cohorts")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    rows = []
    for row in sample.itertuples():
        uri = row.history_uri_tminus96
        metadata = subprocess.run([args.gcloud, "storage", "objects", "describe", uri, "--format=json"],
            capture_output=True, text=True, check=True, timeout=60)
        object_info = json.loads(metadata.stdout)
        generation = str(object_info["generation"])
        destination = args.output_dir / f"{row.source_cohort}.npz"
        if destination.exists():
            expected_md5 = object_info.get("md5_hash", object_info.get("md5Hash"))
            actual_md5 = base64.b64encode(hashlib.md5(destination.read_bytes()).digest()).decode()
            if expected_md5 != actual_md5:
                raise ValueError("Cached AIA file differs from the described cloud generation; choose a fresh output")
        else:
            subprocess.run([args.gcloud, "storage", "cp", f"{uri}#{generation}", str(destination)],
                capture_output=True, text=True, check=True, timeout=120)
        rows.append({"case_id": row.forecast_case_id, "cohort": row.source_cohort, "issue_utc": row.issue_utc,
            "uri": uri, "generation": generation, "bytes": destination.stat().st_size,
            "sha256": file_sha256(destination), **check_frame(destination)})
    result = {"status": "three_cohort_aia_io_check_passed", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "cases": rows, "scope": "Three latest-case history frames only; not a full AIA data or fusion-model validation."}
    (args.output_dir / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
