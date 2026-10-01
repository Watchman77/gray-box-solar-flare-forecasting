"""Replay positive-evidence checks against raw daily report lines.

One separately documented source case links an unassociated XRA entry to its
same-day SWPC event-bin optical report. It does not rewrite source XRA fields.
"""

import argparse
import json
from pathlib import Path
import re

import pandas as pd

try:
    from scripts.prepare_dataset_sources import digest
except ModuleNotFoundError:
    from prepare_dataset_sources import digest

XRA = re.compile(r"^\s*(\d+)\s+\+?\s*(\d{4})\s+(\d{4})\s+\S+\s+\S+\s+\S+\s+XRA\s+\S+\s+([MX]\d+(?:\.\d+)?)\s+\S+(?:\s+(\d{4}))?\s*$")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Verification output already exists")
    receipt = json.loads((args.source_dir / "receipt.json").read_text())
    summary = json.loads((args.audit_dir / "summary.json").read_text())
    if digest(args.source_dir / "receipt.json") != summary["source_receipt_sha256"]:
        raise ValueError("Source receipt mismatch")
    raw = {}
    for item in receipt["sources"]:
        path = args.source_dir / item["file"]
        if digest(path) != item["sha256"]:
            raise ValueError("Raw SWPC source changed")
        raw[path.name] = path.read_text().splitlines()
    for name, expected in summary["output_sha256"].items():
        if digest(args.audit_dir / name) != expected:
            raise ValueError("SWPC audit artifact changed")
    cases = pd.read_csv(args.audit_dir / "case_checks.csv")
    reviewed_rows = set()
    corrections = []
    for case in cases.loc[cases.direct_swpc_mx_window_evidence].itertuples():
        supported = False
        for reference in case.supporting_swpc_rows.split(";"):
            name, line = reference.split(":")
            match = XRA.fullmatch(raw[name][int(line) - 1])
            if not match:
                raise ValueError(f"Raw XRA line does not match the independent parser: {reference}")
            event_bin, begin, peak, cls, region = match.groups()
            if region is None or int(region) + 10000 != case.NOAA_AR_clean:
                raise ValueError("Raw source region does not support the forecast target")
            start = pd.Timestamp(name[:8], tz="UTC") + pd.Timedelta(hours=int(begin[:2]), minutes=int(begin[2:]))
            issue = pd.Timestamp(case.issue_utc)
            if not issue < start <= issue + pd.Timedelta(hours=48):
                raise ValueError("Raw SWPC event falls outside the future window")
            supported = True
            reviewed_rows.add(reference)
        if not supported or case.upstream_label_48h != 0:
            raise ValueError("Unsupported correction proposal")
        corrections.append({"forecast_case_id": case.forecast_case_id, "input_cohort": case.input_cohort,
            "matched_inputs": case.matched_inputs, "original_label_48h": 0, "source_supported_label_48h": 1,
            "evidence_basis": "same_region_xra_report_in_future_window", "supporting_swpc_rows": case.supporting_swpc_rows})
    unresolved = cases.loc[~cases.direct_swpc_mx_window_evidence]
    expected_case = "gb72-native-v1:20240814_1412_HARP11689_NOAA13784"
    if len(unresolved) != 1 or unresolved.iloc[0].forecast_case_id != expected_case:
        raise ValueError("The remaining queue differs from the documented event-bin review")
    case = unresolved.iloc[0]
    name = "20240814events.txt"
    xra_line, optical_line = raw[name][101], raw[name][100]
    xra = XRA.fullmatch(xra_line)
    if not xra or xra.groups() != ("3510", "1539", "1549", "M5.3", None):
        raise ValueError("The reviewed unassociated XRA row changed")
    optical = optical_line.split()
    if optical[0] != "3510" or "FLA" not in optical or optical[-1] != "3784":
        raise ValueError("The reviewed optical region/event-bin link changed")
    # Do not infer association by proximity alone. Inspect all same-bin optical
    # rows and reject any conflicting region in that daily event group.
    optical_regions = {line.split()[-1] for line in raw[name] if re.match(r"^3510\s", line)
                       and " FLA " in line and line.split()[-1].isdigit()}
    if optical_regions != {"3784"}:
        raise ValueError("Conflicting optical region within the SWPC event group")
    start = pd.Timestamp("2024-08-14T15:39:00Z")
    if not pd.Timestamp(case.issue_utc) < start <= pd.Timestamp(case.issue_utc) + pd.Timedelta(hours=48):
        raise ValueError("Event-bin review is outside the forecast horizon")
    corrections.append({"forecast_case_id": expected_case, "input_cohort": case.input_cohort,
        "matched_inputs": bool(case.matched_inputs), "original_label_48h": 0, "source_supported_label_48h": 1,
        "evidence_basis": "xra_and_optical_region_in_same_swpc_day_event_bin", "supporting_swpc_rows": f"{name}:101;{name}:102"})
    result = pd.DataFrame(corrections)
    if len(result) != len(cases) or result.forecast_case_id.duplicated().any():
        raise ValueError("Correction proposal population mismatch")
    result["applied_to_upstream_labels"] = False
    args.output_dir.mkdir(parents=True)
    result.to_csv(args.output_dir / "source_supported_corrections_48h.csv", index=False)
    report = {"status": "all_flagged_cases_supported_by_archived_swpc_positive_evidence",
        "cases_checked": len(result), "matched_input_cases": int(result.matched_inputs.sum()),
        "direct_xra_cases": int(result.evidence_basis.eq("same_region_xra_report_in_future_window").sum()),
        "event_bin_linked_cases": 1, "distinct_direct_xra_rows": len(reviewed_rows),
        "raw_daily_files_hash_verified": len(raw),
        "event_bin_review": {"day": "2024-08-14", "event_bin": "3510", "xra_row": f"{name}:102", "optical_row": f"{name}:101",
            "original_xra_region_missing": True, "optical_noaa_region": 13784,
            "rule_source": "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_event_reports/docs/SolarEventReport.pdf"},
        "audit_summary_sha256": digest(args.audit_dir / "summary.json"),
        "correction_proposals_sha256": digest(args.output_dir / "source_supported_corrections_48h.csv"),
        "limitations": ["These are source-supported corrections for this bounded queue, not a complete repaired dataset.",
            "The event-bin link is separately recorded; the source XRA region remains missing.",
            "No existing research label, prediction, training job or reported model metric was changed."]}
    (args.output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
