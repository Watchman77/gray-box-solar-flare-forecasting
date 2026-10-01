"""Stage a pinned metadata snapshot and NOAA outcome-source candidates only.

No images, training jobs, source edits or continuous-XRS downloads are performed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv/"
CANONICAL_SHA256 = "c28623e7577447ccf245303b90b45ed455720a70ab81c826a48f2c483266cd41"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aia-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-receipt", type=Path, help="Require exact NOAA hashes from an earlier build")
    args = parser.parse_args()
    root = args.output_dir
    canonical = root / "canonical_1790079135531166.csv"
    if digest(canonical) != CANONICAL_SHA256:
        raise ValueError("Canonical metadata does not match the previously verified generation")
    existing = args.aia_root / "inputs/goes_primary_source_audit_20261001"
    receipt = json.loads((existing / "source_receipt.json").read_text())["sources"]
    noaa = root / "noaa_flare_report_v1_0_1"
    noaa.mkdir(parents=True, exist_ok=True)
    rows = []
    metadata_name = "sci_xrsf-l2-flrpt_geo_metadata.json"
    names = [metadata_name] + [f"sci_xrsf-l2-flrpt_geo_y{year}_v1-0-1.csv" for year in range(2010, 2027)]
    previous = root / "outcome_source_receipt.json"
    known = {x["filename"]: x for x in json.loads(previous.read_text())["files"]} if previous.exists() else {}
    pinned = {x["filename"]: x for x in json.loads(args.expected_receipt.read_text())["files"]} if args.expected_receipt else {}
    if pinned and set(pinned) != set(names):
        raise ValueError("Pinned receipt does not cover exactly the required source files")

    def stage(name):
        destination = noaa / name
        url = BASE + name
        if destination.exists():
            expected = pinned.get(name, known.get(name, receipt.get(name, {}))).get("sha256")
            if expected is None or digest(destination) != expected:
                raise ValueError(f"Existing file has no matching source receipt: {name}")
            origin = "existing_verified_stage"
        elif name in receipt:
            source = existing / name
            if digest(source) != receipt[name]["sha256"]:
                raise ValueError(f"Upstream source hash mismatch: {name}")
            if name in pinned and digest(source) != pinned[name]["sha256"]:
                raise ValueError(f"Upstream no longer matches pinned source version: {name}")
            shutil.copyfile(source, destination)
            origin = "reused_upstream_snapshot"
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "GrayBoxResearch/0.1"})
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read(2 * 1024 * 1024 + 1)
            if len(data) > 2 * 1024 * 1024 or not data.startswith(b"time,start_time,"):
                raise ValueError(f"Unexpected or oversized annual CSV: {name}")
            if name in pinned and hashlib.sha256(data).hexdigest() != pinned[name]["sha256"]:
                raise ValueError(f"Remote source changed since the pinned build; recover the archived version: {name}")
            destination.write_bytes(data)
            origin = "downloaded_missing_annual_source"
        return {"filename": name, "url": url, "sha256": digest(destination),
                "bytes": destination.stat().st_size, "origin": origin}

    with ThreadPoolExecutor(max_workers=3) as executor:
        # Record successful futures even if another request fails; a retry must
        # not orphan already-completed downloads without their hashes.
        pending = {executor.submit(stage, name): name for name in names}
        failures = []
        for future in as_completed(pending):
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append(f"{pending[future]}: {exc}")
                continue
            rows.sort(key=lambda row: row["filename"])
            previous.write_text(json.dumps({"retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "status": "complete" if len(rows) == len(names) else "partial",
                "files": rows}, indent=2) + "\n")
        if failures:
            raise RuntimeError("Source staging incomplete: " + "; ".join(failures))
    for name in ["Leap_Second.dat"]:
        source = args.aia_root / "inputs/goes_cutoff_time_audit_20261001" / name
        destination = root / name
        if destination.exists() and digest(destination) != digest(source):
            raise ValueError("Staged time table differs; create a new source version")
        if not destination.exists():
            shutil.copyfile(source, destination)
    print(json.dumps({"canonical_sha256": digest(canonical), "outcome_source_files": len(rows),
        "downloaded_files": sum(x["origin"] == "downloaded_missing_annual_source" for x in rows),
        "total_outcome_source_bytes": sum(x["bytes"] for x in rows)}, indent=2))


if __name__ == "__main__":
    main()
