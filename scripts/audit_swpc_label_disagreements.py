"""Check every flagged science event against pinned SWPC daily XRA reports.

This audits positive evidence for a bounded review queue, not the completeness
of non-flare labels. It preserves every disagreement and source response.
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
except ModuleNotFoundError:
    from prepare_dataset_sources import digest

BASE = "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_event_reports/"


def parse_xra(path):
    day = pd.Timestamp(path.name[:8], tz="UTC")
    text = path.read_text()
    if f":Date: {day:%Y %m %d}" not in text:
        raise ValueError("SWPC report date does not match its filename")
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        tokens = line.split()
        if "XRA" not in tokens or line.startswith(("#", ":")):
            continue
        pos = tokens.index("XRA")
        if pos < 6 or len(tokens) <= pos + 2:
            raise ValueError(f"Unexpected XRA row: {path.name}:{line_number}")
        start, peak = tokens[pos - 5], tokens[pos - 4]
        cls = tokens[pos + 2]
        if not re.fullmatch(r"[ABCMX]\d+(?:\.\d+)?", cls):
            raise ValueError("Unparsed XRA class")
        # Do not treat uncertain/bounded or missing clocks as exact timestamps.
        exact = bool(re.fullmatch(r"\d{4}", start) and re.fullmatch(r"\d{4}", peak))
        start_utc = peak_utc = pd.NaT
        if exact:
            clocks = []
            for clock in [start, peak]:
                hh, mm = int(clock[:2]), int(clock[2:])
                if hh >= 24 or mm >= 60:
                    raise ValueError("Invalid SWPC clock")
                clocks.append(day + pd.Timedelta(hours=hh, minutes=mm))
            start_utc, peak_utc = clocks
            if peak_utc < start_utc:
                peak_utc += pd.Timedelta(days=1)
        suffix = tokens[-1] if len(tokens) > pos + 3 and tokens[-1].isdigit() else None
        noaa = int(suffix) + 10000 if suffix and 0 < int(suffix) < 10000 else None
        rows.append({"swpc_row_id": f"{path.name}:{line_number}", "date": str(day.date()),
            "swpc_event_bin": tokens[0], "start_utc": start_utc, "peak_utc": peak_utc,
            "exact_clocks": exact, "flare_class": cls, "is_mx": cls.startswith(("M", "X")),
            "noaa_region": noaa, "observatory": tokens[pos - 2], "quality": tokens[pos - 1]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconciliation-dir", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--reuse-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Output audit version already exists")
    summary = json.loads((args.reconciliation_dir / "reconciliation_summary.json").read_text())
    for name in ["event_reconciliation.csv.gz", "label_lineage_diagnostics.csv.gz"]:
        if digest(args.reconciliation_dir / name) != summary["output_sha256"][name]:
            raise ValueError("Reconciliation input changed")
    events = pd.read_csv(args.reconciliation_dir / "event_reconciliation.csv.gz")
    cases = pd.read_csv(args.reconciliation_dir / "label_lineage_diagnostics.csv.gz", low_memory=False)
    flagged = cases.loc[cases.old_negative_with_corroborated_science_positive].copy()
    used = set(";".join(flagged.primary_event_ids).split(";"))
    selected = events.loc[events.event_id.isin(used) & events.original_noaa_region.notna()
        & events.same_region_mx_annual_reference_ids.notna() & ~events.same_region_mx_base_hek_reference].copy()
    dates = sorted(set(pd.to_datetime(selected.start_utc, utc=True).dt.strftime("%Y%m%d")) |
        set(pd.to_datetime(selected.peak_utc, utc=True).dt.strftime("%Y%m%d")))
    if len(dates) > 100:
        raise ValueError("Bounded daily-source audit requires a new acquisition scope")
    args.source_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.source_dir / "receipt.json"
    previous = json.loads(receipt_path.read_text())["sources"] if receipt_path.exists() else []
    known = {x["file"]: x for x in previous}
    reused = {x["file"]: x for x in json.loads((args.reuse_dir / "receipt.json").read_text())["sources"]}

    def fetch(date):
        name = f"{date}events.txt"
        url = f"{BASE}{date[:4]}/{date[4:6]}/{name}"
        destination = args.source_dir / name
        if destination.exists():
            if name not in known or digest(destination) != known[name]["sha256"]:
                raise ValueError(f"Unpinned daily source: {name}")
        elif name in reused:
            source = args.reuse_dir / name
            if digest(source) != reused[name]["sha256"]:
                raise ValueError("Spot-check source changed")
            shutil.copyfile(source, destination)
        else:
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read(512001)
            if len(data) > 512000 or b":Product:" not in data:
                raise ValueError("Unexpected daily report response")
            destination.write_bytes(data)
        return {"file": name, "url": url, "sha256": digest(destination), "bytes": destination.stat().st_size}

    sources, failures = [], []
    with ThreadPoolExecutor(max_workers=3) as executor:
        pending = {executor.submit(fetch, date): date for date in dates}
        for future in as_completed(pending):
            try:
                sources.append(future.result())
            except Exception as exc:
                failures.append({"date": pending[future], "error": str(exc)})
            receipt_path.write_text(json.dumps({"retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "sources": sorted(sources, key=lambda x: x["file"]), "failures": failures}, indent=2) + "\n")
    if failures:
        raise ValueError(f"{len(failures)} daily sources failed; successes are pinned for resumption")
    reports = pd.concat([parse_xra(args.source_dir / item["file"]) for item in sources], ignore_index=True)
    event_rows = []
    for event in selected.itertuples():
        matches = reports.loc[reports.exact_clocks & reports.is_mx
            & reports.peak_utc.eq(pd.Timestamp(event.peak_utc)) & reports.noaa_region.eq(event.original_noaa_region)]
        near = reports.loc[reports.exact_clocks & reports.is_mx
            & ((reports.peak_utc - pd.Timestamp(event.peak_utc)).abs() <= pd.Timedelta(minutes=4))]
        event_rows.append({"event_id": event.event_id, "science_peak_utc": event.peak_utc,
            "science_noaa_region": event.original_noaa_region, "science_class": event.science_class,
            "exact_peak_same_region_swpc_mx_rows": ";".join(matches.swpc_row_id),
            "swpc_classes": ";".join(sorted(set(matches.flare_class))),
            "nearby_mx_rows_for_review": ";".join(near.swpc_row_id),
            "status": "direct_xra_positive_evidence" if len(matches) else "needs_review_no_exact_same_region_xra"})
    event_result = pd.DataFrame(event_rows).set_index("event_id", drop=False)
    case_rows = []
    for case in flagged.itertuples():
        start = pd.Timestamp(case.issue_utc)
        if not 2021 <= start.year <= 2024:
            raise ValueError("This audit's elapsed-hour shortcut is bounded to a period without leap insertions")
        end = start + pd.Timedelta(hours=48)
        linked = []
        for event_id in case.primary_event_ids.split(";"):
            if event_id not in event_result.index:
                continue
            report_ids = event_result.loc[event_id, "exact_peak_same_region_swpc_mx_rows"].split(";")
            supported = reports.loc[reports.swpc_row_id.isin(report_ids) & (reports.start_utc > start)
                & (reports.start_utc <= end) & reports.noaa_region.eq(case.NOAA_AR_clean)]
            linked.extend(supported.swpc_row_id.tolist())
        case_rows.append({"forecast_case_id": case.forecast_case_id, "issue_utc": case.issue_utc,
            "input_cohort": case.input_cohort, "matched_inputs": case.upstream_matched_three_slot_inputs,
            "upstream_label_48h": case.upstream_label_48h, "NOAA_AR_clean": case.NOAA_AR_clean,
            "direct_swpc_mx_window_evidence": bool(linked), "supporting_swpc_rows": ";".join(sorted(set(linked)))})
    case_result = pd.DataFrame(case_rows)
    args.output_dir.mkdir(parents=True)
    for name, frame in [("parsed_xra_rows.csv", reports), ("event_checks.csv", event_result), ("case_checks.csv", case_result)]:
        frame.to_csv(args.output_dir / name, index=False)
    result = {"status": "bounded_direct_swpc_positive_evidence_audit_complete", "daily_files": len(sources),
        "download_bytes": sum(x["bytes"] for x in sources), "source_receipt_sha256": digest(receipt_path),
        "reviewed_science_events": len(selected), "events_with_direct_same_region_mx": int(event_result.status.eq("direct_xra_positive_evidence").sum()),
        "flagged_cases": len(case_result), "cases_with_direct_swpc_mx_in_window": int(case_result.direct_swpc_mx_window_evidence.sum()),
        "matched_cases_with_direct_swpc_mx_in_window": int((case_result.direct_swpc_mx_window_evidence & case_result.matched_inputs).sum()),
        "by_cohort": case_result.groupby("input_cohort").agg(review_cases=("forecast_case_id", "size"), direct_positive=("direct_swpc_mx_window_evidence", "sum")).reset_index().to_dict(orient="records"),
        "output_sha256": {p.name: digest(p) for p in sorted(args.output_dir.iterdir())},
        "limitations": ["Only the flagged positive-evidence queue was audited; absence of a matched row does not establish a negative.",
            "SWPC reports and HER may share underlying observations; direct reports improve provenance, not independent-observation count.",
            "Historical products can contain corrections and uncertainty. No frozen upstream labels or scores were overwritten."]}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
