"""Stage a bounded, label-independent AIA check from existing cloud objects."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

import pandas as pd

from scripts.dataset_io import file_sha256


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def select_cases(cases):
    """First/last issue in each declared role; labels never determine selection."""
    chosen = []
    for _, group in cases.groupby("role", sort=False):
        ordered = group.sort_values(["issue_utc", "forecast_case_id"])
        chosen.append(pd.concat([ordered.head(1), ordered.tail(1)]).drop_duplicates("forecast_case_id"))
    return pd.concat(chosen, ignore_index=True)


def prepare(root, parent, output):
    root, parent, output = map(Path, [root, parent, output])
    if output.exists():
        raise ValueError("Canary staging directory already exists")
    contract = json.loads((parent / "experiment_contract.json").read_text())
    cases_file = parent / "candidate_comparison_cases.csv.gz"
    if file_sha256(cases_file) != contract["split_manifest_sha256"]:
        raise ValueError("Frozen candidate case manifest changed")
    cases = pd.read_csv(cases_file, low_memory=False)
    if set(cases.role) != {b["role"] for b in contract["blocks"]}:
        raise ValueError("Unexpected roles in canary source")
    selected = select_cases(cases)
    uris = sorted(set(selected[[f"history_uri_tminus{x}" for x in [288, 192, 96]]].to_numpy().ravel()))
    if len(selected) != 14 or len(uris) > 42:
        raise ValueError("Canary selection exceeds its declared bound")
    output.mkdir(parents=True)
    selected.to_csv(output / "cases.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    plan = {"status": "prepared_not_downloaded", "prepared_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Technical loader/gradient/checkpoint canary, not scientific model evaluation",
            "selection": "First and last eligible issue per role, independent of outcomes",
            "parent_contract_sha256": file_sha256(parent / "experiment_contract.json"),
            "parent_cases_sha256": file_sha256(cases_file), "cases_sha256": file_sha256(output / "cases.csv.gz"),
            "cases": len(selected), "uris": uris, "max_files": 42, "max_object_bytes": 16 * 1024**2,
            "max_download_bytes": 512 * 1024**2, "max_seconds": 1200, "workers": 3,
            "minimum_free_bytes": 4 * 1024**3, "source_bucket": "suryabench-sharp-pipeline-bamidele",
            "gpu_used": False, "old_embedded_labels_used": False}
    write_json(output / "plan.json", plan)
    return plan


def stage(root, output, gcloud):
    root, output = Path(root), Path(output)
    plan = json.loads((output / "plan.json").read_text())
    if (output / "complete.json").exists() or (output / "failure.json").exists():
        raise ValueError("Existing completion/failure retained; no automatic overwrite or retry")
    if shutil.disk_usage(output).free < plan["minimum_free_bytes"] + plan["max_download_bytes"]:
        raise ValueError("Insufficient local disk reserve")
    started = time.monotonic()
    # Keep the token in process memory; never log, serialize or put it in a URL.
    token = subprocess.run([gcloud, "auth", "print-access-token"], capture_output=True,
                           text=True, check=True, timeout=60).stdout.strip()
    def get(url):
        if time.monotonic() - started > plan["max_seconds"]:
            raise TimeoutError("Canary staging time bound exceeded")
        return urlopen(Request(url, headers={"Authorization": "Bearer " + token}), timeout=90)
    def metadata(uri):
        parsed = urlsplit(uri)
        if parsed.scheme != "gs" or parsed.netloc != plan["source_bucket"]:
            raise ValueError("Object is outside the declared source bucket")
        key = quote(parsed.path.lstrip("/"), safe="")
        endpoint = f"https://storage.googleapis.com/storage/v1/b/{parsed.netloc}/o/{key}"
        with get(endpoint) as stream:
            record = json.loads(stream.read(1 << 20))
        size = int(record["size"])
        if not 0 < size <= plan["max_object_bytes"] or not record.get("md5Hash"):
            raise ValueError("Unbounded object or missing source checksum")
        return {"uri": uri, "generation": str(record["generation"]), "bytes": size,
                "md5_base64": record["md5Hash"], "endpoint": endpoint}
    try:
        with ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
            objects = list(pool.map(metadata, plan["uris"]))
        if len(objects) > plan["max_files"] or sum(r["bytes"] for r in objects) > plan["max_download_bytes"]:
            raise ValueError("Canary object/byte cap exceeded before downloads")
        write_json(output / "pinned_objects.json", objects)
        cache = {}
        receipt = root / "outputs/aia_pipeline_sample_v1/receipt.json"
        if receipt.is_file():
            for record in json.loads(receipt.read_text())["cases"]:
                path = receipt.parent / (record["cohort"] + ".npz")
                if path.is_file() and file_sha256(path) == record["sha256"]:
                    cache[(record["uri"], record["generation"])] = path
        (output / "objects").mkdir()
        def fetch(record):
            cached = cache.get((record["uri"], record["generation"]))
            if cached is not None:
                path = cached
                reused = True
            else:
                key = hashlib.sha256((record["uri"] + "#" + record["generation"]).encode()).hexdigest()
                path = output / "objects" / (key + ".npz")
                partial = path.with_suffix(".partial")
                with get(record["endpoint"] + "?alt=media&generation=" + record["generation"]) as stream:
                    payload = stream.read(record["bytes"] + 1)
                if len(payload) != record["bytes"]:
                    raise ValueError("Downloaded byte count differs from pinned object")
                partial.write_bytes(payload)
                if base64.b64encode(hashlib.md5(payload).digest()).decode() != record["md5_base64"]:
                    raise ValueError("Pinned object checksum differs; partial retained")
                partial.rename(path)
                reused = False
            payload = path.read_bytes()
            if len(payload) != record["bytes"] or base64.b64encode(hashlib.md5(payload).digest()).decode() != record["md5_base64"]:
                raise ValueError("Cached object checksum differs")
            return {k: v for k, v in record.items() if k != "endpoint"} | {
                "path_relative_to_repository": str(path.relative_to(root)),
                "sha256": hashlib.sha256(payload).hexdigest(), "reused_local": reused}
        with ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
            records = list(pool.map(fetch, objects))
        result = {"status": "bounded_objects_staged_not_model_checked", "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "plan_sha256": file_sha256(output / "plan.json"), "objects": records,
                  "local_reused": sum(r["reused_local"] for r in records),
                  "downloaded_bytes": sum(r["bytes"] for r in records if not r["reused_local"]),
                  "seconds": time.monotonic() - started}
        write_json(output / "complete.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "objects"}), flush=True)
        return result
    except Exception as error:
        write_json(output / "failure.json", {"status": "stopped", "error_type": type(error).__name__,
                                              "message": str(error), "partial_files_preserved": True})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--gcloud", default="/Users/mac/google-cloud-sdk/bin/gcloud")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.prepare:
        plan = prepare(root, args.parent.resolve(), args.output.resolve())
        print(json.dumps({k: v for k, v in plan.items() if k != "uris"}))
    if args.stage:
        stage(root, args.output.resolve(), args.gcloud)
