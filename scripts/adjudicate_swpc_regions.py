"""Conservative, versioned region adjudication for the locked 651-event queue.

Exact peak matches identify candidate daily event groups, never nearest-time
matches. Archived reports support retrospective associations, not data latency.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import urllib.error
import urllib.request

import pandas as pd

try:
    from scripts.prepare_dataset_sources import digest
    from scripts.build_dataset_inventory import write_csv
    from scripts.audit_swpc_label_disagreements import BASE
except ModuleNotFoundError:
    from prepare_dataset_sources import digest
    from build_dataset_inventory import write_csv
    from audit_swpc_label_disagreements import BASE

POLICY = "swpc_exact_peak_region_v1"
COLUMNS = ["row_id", "date", "group_id", "event_bin", "selected", "type", "observatory",
           "quality", "start_raw", "peak_raw", "end_raw", "start_utc", "peak_utc", "end_utc",
           "exact_start_peak", "source_time_problem", "flare_class", "region", "raw_line"]


def clock(day, value):
    if not re.fullmatch(r"\d{4}", value):
        return pd.NaT
    hh, mm = int(value[:2]), int(value[2:])
    if hh >= 24 or mm >= 60:
        raise ValueError(f"Invalid clock {value}")
    return day + pd.Timedelta(hours=hh, minutes=mm)


def parse_report(path):
    day = pd.Timestamp(path.name[:8], tz="UTC")
    text = path.read_text()
    if f":Date: {day:%Y %m %d}" not in text:
        raise ValueError(f"Report date mismatch: {path.name}")
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not re.match(r"^\d{4}\s", line):
            continue
        expanded_cme = (line[43:46] == "CME" and line[48:58] == "XUV,EUV,UV" and len(line) in (82, 83))
        if (len(line) > 80 and not expanded_cme) or not re.fullmatch(r"[A-Z]{3}", line[43:46]):
            raise ValueError(f"Unexpected fixed-width layout: {path.name}:{number}")
        line80 = line.ljust(80)
        start_raw, peak_raw, end_raw = (line80[a:b].strip() for a, b in ((10, 15), (17, 22), (27, 32)))
        time_problem = False
        times = []
        for value in (start_raw, peak_raw, end_raw):
            try:
                times.append(clock(day, value))
            except ValueError:
                times.append(pd.NaT)
                time_problem = True
        start, peak, end = times
        if pd.notna(start):
            if pd.notna(peak) and peak < start:
                peak += pd.Timedelta(days=1)
            if pd.notna(end) and end < start:
                end += pd.Timedelta(days=1)
        else:
            # Without an exact begin clock the day of max/end is not determined.
            peak = end = pd.NaT
        if pd.notna(peak) and pd.notna(end) and end < peak:
            end = pd.NaT
            time_problem = True
        # Archived CME rows expand Loc/Frq by two characters; several also
        # contain full five-digit NOAA IDs. Do not truncate their region field.
        suffix = (line[78:] if expanded_cme else line80[76:80]).strip()
        if suffix and not re.fullmatch(r"\d{4}|1\d{4}", suffix):
            raise ValueError(f"Invalid region field: {path.name}:{number}")
        region = (int(suffix) + 10000 if int(suffix) < 10000 else int(suffix)) if suffix and int(suffix) > 0 else None
        kind = line80[43:46]
        cls = line80[58:76].split()[0] if kind == "XRA" else ""
        if kind == "XRA" and not re.fullmatch(r"[ABCMX]\d+(?:\.\d+)?", cls):
            raise ValueError(f"Invalid XRA class: {path.name}:{number}")
        rows.append({"row_id": f"{path.name}:{number}", "date": f"{day:%Y%m%d}",
            "group_id": f"{day:%Y%m%d}:{line80[:4]}", "event_bin": line80[:4],
            "selected": line80[5] == "+", "type": kind, "observatory": line80[34:37],
            "quality": line80[39].strip(), "start_raw": start_raw, "peak_raw": peak_raw,
            "end_raw": end_raw, "start_utc": start, "peak_utc": peak, "end_utc": end,
            "exact_start_peak": pd.notna(start) and pd.notna(peak) and not time_problem,
            "source_time_problem": time_problem, "flare_class": cls,
            "region": region, "raw_line": line})
    frame = pd.DataFrame(rows, columns=COLUMNS)
    for col in ("start_utc", "peak_utc", "end_utc"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    frame["region"] = frame.region.astype("Int64")
    return frame


def required_dates(queue):
    starts = pd.to_datetime(queue.start_utc, utc=True)
    peaks = pd.to_datetime(queue.peak_utc, utc=True)
    # A report is filed on the XRA begin date, which may precede its peak date.
    return sorted(set(starts.dt.strftime("%Y%m%d")) | set(peaks.dt.strftime("%Y%m%d")) |
                  set((peaks.dt.normalize() - pd.Timedelta(days=1)).dt.strftime("%Y%m%d")))


def acquire(queue, source_dir, reuse_dirs, queue_sha256):
    dates = required_dates(queue)
    if len(queue) != 651 or len(dates) > 900:
        raise ValueError("Acquisition scope differs from the locked 651-event queue")
    source_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = source_dir / "receipt.json"
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"sources": []}
    known = {x["file"]: x for x in previous["sources"]}
    reuse = {}
    for root in reuse_dirs:
        for item in json.loads((root / "receipt.json").read_text())["sources"]:
            if item["file"] in reuse and reuse[item["file"]][1]["sha256"] != item["sha256"]:
                raise ValueError("Conflicting cached daily-source versions")
            reuse[item["file"]] = (root, item)

    def fetch(date):
        name = f"{date}events.txt"
        url = f"{BASE}{date[:4]}/{date[4:6]}/{name}"
        path = source_dir / name
        if path.exists():
            if name not in known or digest(path) != known[name]["sha256"]:
                raise ValueError(f"Unpinned or changed existing file {name}")
            return known[name]
        if name in reuse:
            root, original = reuse[name]
            if digest(root / name) != original["sha256"]:
                raise ValueError(f"Changed cache {name}")
            shutil.copyfile(root / name, path)
            method = "reused_pinned_report"
        else:
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read(512001)
            if len(data) > 512000 or b":Product:" not in data or f":Date: {date[:4]} {date[4:6]} {date[6:]}".encode() not in data:
                raise ValueError(f"Unexpected report response {name}")
            path.write_bytes(data)
            method = "public_archive_download"
        return {"file": name, "url": url, "sha256": digest(path), "bytes": path.stat().st_size,
                "method": method, "retrieved_utc": datetime.now(timezone.utc).isoformat()}

    # Keep earlier pinned successes in intermediate receipts during a retry.
    # Otherwise interruption while revisiting the cache could orphan later files.
    sources = {name: item for name, item in known.items() if name[:8] in dates}
    failures = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        pending = {executor.submit(fetch, date): date for date in dates}
        for completed, future in enumerate(as_completed(pending), 1):
            try:
                item = future.result()
                sources[item["file"]] = item
            except Exception as exc:
                failures.append({"date": pending[future], "error": str(exc)})
                sources.pop(f"{pending[future]}events.txt", None)
            receipt = {"policy": POLICY, "requested_dates": dates, "queue_rows": len(queue),
                "queue_sha256": queue_sha256,
                "sources": sorted(sources.values(), key=lambda x: x["file"]), "failures": failures,
                "status": "complete" if len(sources) == len(dates) and completed == len(dates) and not failures else "incomplete"}
            temporary = source_dir / "receipt.tmp.json"
            temporary.write_text(json.dumps(receipt, indent=2) + "\n")
            temporary.replace(receipt_path)
            if completed % 50 == 0:
                print(f"Reports processed {completed}/{len(dates)}; failures {len(failures)}", flush=True)
    return receipt


def region_set(series):
    return sorted(int(x) for x in series.dropna().unique())


def adjudicate(queue, reports, available_dates):
    groups = {key: part for key, part in reports.groupby("group_id")}
    exact = reports.loc[reports.type.eq("XRA") & reports.exact_start_peak & reports.quality.eq("5")]
    by_peak = {key: part for key, part in exact.groupby("peak_utc")}
    rows = []
    for event in queue.itertuples():
        peak = pd.Timestamp(event.peak_utc)
        matches = by_peak.get(peak, exact.iloc[:0])
        group_ids = sorted(matches.group_id.unique())
        group = groups[group_ids[0]] if len(group_ids) == 1 else reports.iloc[:0]
        all_regions = region_set(group.region)
        direct = matches.loc[matches.region.notna()]
        optical = group.loc[group.type.eq("FLA") & group.region.notna() & ~group.source_time_problem & group.quality.isin(["3", "4", "5"])]
        # Optical support additionally requires an exactly timed overlap with XRA.
        optical_ids = []
        for optical_row in optical.itertuples():
            overlap = matches.start_utc.le(optical_row.end_utc) & matches.end_utc.ge(optical_row.start_utc)
            if pd.notna(optical_row.start_utc) and pd.notna(optical_row.end_utc) and overlap.any():
                optical_ids.append(optical_row.row_id)
        original = int(event.original_noaa_region) if pd.notna(event.original_noaa_region) else None
        reference_regions = {int(x) for x in str(event.exact_peak_known_regions).split(";") if x.isdigit()}
        candidate = all_regions[0] if len(all_regions) == 1 else None
        supporting = direct.row_id.tolist() or optical_ids
        evidence = "direct_xra" if len(direct) else "same_day_event_bin_optical" if optical_ids else "none"
        needed = required_dates(pd.DataFrame([event._asdict()]))
        complete = set(needed).issubset(available_dates)
        if not complete:
            status = "unresolved_missing_daily_source"
        elif event.science_peak_multiplicity != 1:
            status = "unresolved_nonunique_science_peak"
        elif not group_ids:
            status = "unresolved_no_exact_xra_peak"
        elif len(group_ids) != 1:
            status = "unresolved_multiple_swpc_groups"
        elif len(all_regions) > 1:
            status = "unresolved_conflicting_swpc_regions"
        elif candidate is None or not supporting:
            status = "unresolved_no_qualified_region_evidence"
        elif original is not None:
            status = "retained_original_daily_supported" if candidate == original else "quarantined_science_daily_conflict"
        elif reference_regions and reference_regions != {candidate}:
            status = "unresolved_reference_daily_conflict"
        else:
            status = "filled_missing_direct_xra" if evidence == "direct_xra" else "filled_missing_optical_event_bin"
        accepted = status.startswith("filled_missing_") or status == "retained_original_daily_supported"
        rows.append({"event_id": event.event_id, "science_start_utc": event.start_utc,
            "science_peak_utc": event.peak_utc, "science_class": event.science_class,
            "original_region": original, "reference_status": event.reference_status,
            "reference_regions": ";".join(map(str, sorted(reference_regions))),
            "daily_candidate_region": candidate, "candidate_region": candidate if accepted else None,
            "status": status, "evidence_tier": evidence, "daily_source_search_complete": complete,
            "exact_xra_rows": ";".join(matches.row_id), "swpc_group_ids": ";".join(group_ids),
            "swpc_group_regions": ";".join(map(str, all_regions)),
            "supporting_region_rows": ";".join(supporting) if len(group_ids) == 1 else "",
            "swpc_xra_classes": ";".join(sorted(set(matches.flare_class))),
            "historical_mx_class_present": bool(matches.flare_class.str.startswith(("M", "X")).any()),
            "policy": POLICY})
    result = pd.DataFrame(rows)
    for col in ("original_region", "daily_candidate_region", "candidate_region"):
        result[col] = result[col].astype("Int64")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--reuse-dir", type=Path, action="append", default=[])
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    queue = pd.read_csv(args.queue)
    if args.fetch:
        receipt = acquire(queue, args.source_dir, args.reuse_dir, digest(args.queue))
        print(json.dumps({"sources": len(receipt["sources"]), "failures": receipt["failures"]}, indent=2))
    if not args.output_dir:
        return
    if args.output_dir.exists():
        raise ValueError("Output version exists; choose a new directory")
    receipt = json.loads((args.source_dir / "receipt.json").read_text())
    if receipt["queue_sha256"] != digest(args.queue) or receipt["requested_dates"] != required_dates(queue):
        raise ValueError("Queue/source scope mismatch")
    for item in receipt["sources"]:
        if digest(args.source_dir / item["file"]) != item["sha256"]:
            raise ValueError("Pinned report changed")
    reports = pd.concat([parse_report(args.source_dir / item["file"]) for item in receipt["sources"]], ignore_index=True)
    decisions = adjudicate(queue, reports, {x["file"][:8] for x in receipt["sources"]})
    args.output_dir.mkdir(parents=True)
    write_csv(reports, args.output_dir / "parsed_daily_rows.csv.gz")
    decisions.to_csv(args.output_dir / "event_decisions.csv", index=False)
    summary = {"policy": POLICY, "queue_sha256": digest(args.queue),
        "source_receipt_sha256": digest(args.source_dir / "receipt.json"),
        "daily_reports": len(receipt["sources"]), "report_bytes": sum(x["bytes"] for x in receipt["sources"]),
        "failed_sources": receipt["failures"], "reviewed_events": len(decisions),
        "decisions": decisions.status.value_counts().to_dict(),
        "output_sha256": {p.name: digest(p) for p in args.output_dir.iterdir()},
        "training_ready": False, "upstream_labels_modified": False}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
