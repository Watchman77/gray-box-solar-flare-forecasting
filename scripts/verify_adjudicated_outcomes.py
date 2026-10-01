"""Verify all accepted associations and all candidate labels through separate code.

Uses token-based raw-report parsing and event-to-window assignment rather than
the fixed-column parser and case-to-event search used by the builders. This is
a computational cross-check, not independent physical-observation validation.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, description):
    if not condition:
        raise AssertionError(description)


def raw_report(path):
    day = datetime.strptime(path.name[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
    rows = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not re.match(r"^\d{4}\s", line):
            continue
        tokens = list(re.finditer(r"\S+", line))
        words = [m.group() for m in tokens]
        offset = 2 if words[1] == "+" else 1
        start, peak, end, obs, quality, kind = words[offset:offset + 6]
        last = tokens[-1]
        region = int(last.group()) if last.start() >= 76 and re.fullmatch(r"\d{4,5}", last.group()) else None
        if region == 0:
            region = None
        if region is not None and region < 10000:
            region += 10000
        clocks, bad = [], False
        for value in (start, peak, end):
            if re.fullmatch(r"\d{4}", value):
                try:
                    clocks.append(day.replace(hour=int(value[:2]), minute=int(value[2:])))
                except ValueError:
                    clocks.append(None)
                    bad = True
            else:
                clocks.append(None)
        begin, maximum, finish = clocks
        if begin is None:
            maximum = finish = None
        else:
            maximum = maximum + timedelta(days=1) if maximum and maximum < begin else maximum
            finish = finish + timedelta(days=1) if finish and finish < begin else finish
        if maximum and finish and finish < maximum:
            bad = True
            finish = None
        cls = words[offset + 7] if kind == "XRA" else ""
        if kind == "XRA":
            check(bool(re.fullmatch(r"[ABCMX]\d+(?:\.\d+)?", cls)), f"Raw class: {path.name}:{n}")
        rows.append({"id": f"{path.name}:{n}", "group": f"{path.name[:8]}:{words[0]}",
            "type": kind, "quality": quality, "start": begin, "peak": maximum, "end": finish,
            "region": region, "class": cls, "time_problem": bad,
            "usable_xra": kind == "XRA" and quality == "5" and begin is not None and maximum is not None and not bad})
    return rows


def known_utc(tai):
    offsets = np.full(len(tai), 34)
    offsets[tai >= pd.Timestamp("2012-07-01T00:00:35")] = 35
    offsets[tai >= pd.Timestamp("2015-07-01T00:00:36")] = 36
    offsets[tai >= pd.Timestamp("2017-01-01T00:00:37")] = 37
    return (tai - pd.to_timedelta(offsets, unit="s")).dt.tz_localize("UTC")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source-dir", "adjudication-dir", "outcome-dir", "original-outcome-dir", "inventory-dir", "output"]:
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Verification receipt exists")
    receipt = json.loads((args.source_dir / "receipt.json").read_text())
    adj = json.loads((args.adjudication_dir / "summary.json").read_text())
    summary = json.loads((args.outcome_dir / "build_summary.json").read_text())
    original_summary = json.loads((args.original_outcome_dir / "build_summary.json").read_text())
    check(sha(args.source_dir / "receipt.json") == adj["source_receipt_sha256"], "Daily receipt lineage")
    check(sha(args.adjudication_dir / "summary.json") == summary["adjudication_summary_sha256"], "Adjudication lineage")
    check(sha(args.original_outcome_dir / "build_summary.json") == summary["original_summary_sha256"], "Original lineage")
    for root, manifest in [(args.adjudication_dir, adj), (args.outcome_dir, summary), (args.original_outcome_dir, original_summary)]:
        for name, expected in manifest["output_sha256"].items():
            check(sha(root / name) == expected, f"Output checksum {name}")
    raw = []
    for item in receipt["sources"]:
        path = args.source_dir / item["file"]
        check(sha(path) == item["sha256"], f"Raw source checksum {path.name}")
        raw.extend(raw_report(path))
    rows = pd.DataFrame(raw).set_index("id", drop=False)
    for col in ["start", "peak", "end"]:
        rows[col] = pd.to_datetime(rows[col], utc=True)
    parsed = pd.read_csv(args.adjudication_dir / "parsed_daily_rows.csv.gz").set_index("row_id")
    check(set(parsed.index) == set(rows.index), "All daily rows retained")
    rows = rows.loc[parsed.index]
    check(parsed.region.fillna(-1).eq(rows.region.fillna(-1)).all(), "All raw region fields")
    for a, b in [("start_utc", "start"), ("peak_utc", "peak"), ("end_utc", "end")]:
        actual = pd.to_datetime(parsed[a], utc=True)
        check((actual.eq(rows[b]) | (actual.isna() & rows[b].isna())).all(), f"All raw {a}")
    check(parsed.source_time_problem.eq(rows.time_problem).all(), "Raw timing flags")
    science = pd.read_csv(args.original_outcome_dir / "normalized_events.csv.gz", low_memory=False).set_index("event_id", drop=False)
    events = pd.read_csv(args.outcome_dir / "adjudicated_events.csv.gz", low_memory=False).set_index("event_id", drop=False)
    decisions = pd.read_csv(args.adjudication_dir / "event_decisions.csv").set_index("event_id", drop=False)
    check(set(science.index) == set(events.index), "Science catalogue population")
    events = events.loc[science.index]
    for col in science.columns:
        if col != "noaa_full_id_candidate":
            # Datetime formatting changes are semantically immaterial.
            before, after = science[col], events[col]
            if col in ("time", "start_time", "end_time"):
                before, after = pd.to_datetime(before, utc=True), pd.to_datetime(after, utc=True)
            check((before.eq(after) | (before.isna() & after.isna())).all(), f"Preserved science field {col}")
    check((events.original_noaa_full_id.fillna(-1) == science.noaa_full_id_candidate.fillna(-1)).all(), "Original region preserved")
    peak_counts = pd.to_datetime(science.time, utc=True).value_counts()
    accepted_count = 0
    for decision in decisions.itertuples(index=False):
        event = events.loc[decision.event_id]
        if pd.isna(decision.candidate_region):
            check(pd.isna(event.noaa_full_id_candidate), f"Unresolved event quarantine {decision.event_id}")
            continue
        accepted_count += 1
        check(event.noaa_full_id_candidate == decision.candidate_region, "Applied region")
        peak = pd.Timestamp(decision.science_peak_utc)
        check(peak_counts[peak] == 1, "Science peak uniqueness")
        matches = rows.loc[rows.usable_xra & rows.peak.eq(peak)]
        check(matches.group.nunique() == 1, "Unique exact-peak daily group")
        members = rows.loc[rows.group.eq(matches.group.iloc[0])]
        regions = set(members.region.dropna().astype(int))
        check(regions == {int(decision.candidate_region)}, "Daily group region unanimity")
        supported = matches.loc[matches.region.eq(decision.candidate_region)]
        if len(supported):
            check(decision.evidence_tier == "direct_xra", "Direct support tier")
            evidence_ids = set(supported.id)
        else:
            check(decision.evidence_tier == "same_day_event_bin_optical", "Optical support tier")
            optical = members.loc[members.type.eq("FLA") & members.quality.isin(["3", "4", "5"])
                & members.region.eq(decision.candidate_region) & ~members.time_problem]
            evidence_ids = {r.id for r in optical.itertuples() if pd.notna(r.start) and pd.notna(r.end)
                and ((matches.start <= r.end) & (matches.end >= r.start)).any()}
            check(bool(evidence_ids), "Qualified optical overlap")
        check(set(str(decision.supporting_region_rows).split(";")) == evidence_ids, "Supporting raw row IDs")
        if pd.isna(decision.original_region):
            reference = {int(x) for x in str(decision.reference_regions).split(";") if x.isdigit()}
            check(not reference or reference == regions, "No contradictory reference in a missing-region fill")
        else:
            check(decision.original_region == decision.candidate_region, "No existing region overwritten")
        check(decision.daily_source_search_complete, "Accepted search source completeness")
        dates = {pd.Timestamp(decision.science_start_utc).strftime("%Y%m%d"), peak.strftime("%Y%m%d"),
                 (peak.normalize() - pd.Timedelta(days=1)).strftime("%Y%m%d")}
        check(dates.issubset({x["file"][:8] for x in receipt["sources"]}), "Accepted source files present")
    untouched = ~events.index.isin(decisions.index)
    check(events.loc[untouched, "noaa_full_id_candidate"].fillna(-1).eq(science.loc[untouched, "noaa_full_id_candidate"].fillna(-1)).all(), "Other associations unchanged")
    inventory = args.inventory_dir / "forecast_case_inventory.csv.gz"
    check(sha(inventory) == summary["inventory_sha256"], "Inventory checksum")
    cases = pd.read_csv(inventory, low_memory=False).set_index("forecast_case_id", drop=False)
    labels = pd.read_csv(args.outcome_dir / "candidate_outcomes.csv.gz", low_memory=False)
    check(not labels.duplicated(["forecast_case_id", "horizon_hours"]).any(), "Unique case/horizon")
    check(not labels.training_ready.any() and labels.new_experiment_role.eq("unassigned").all(), "Readiness/split guards")
    check(set(labels.horizon_hours) == {48, 72}, "Horizon set")
    mx = events.loc[events.is_mx].copy()
    mx["start"] = pd.to_datetime(mx.start_time, utc=True)
    # Reverse assignment: visit every event and assign it to all matching windows.
    for horizon, part in labels.groupby("horizon_hours"):
        check(set(part.forecast_case_id) == set(cases.index), "Both horizons preserve every case")
        part = part.set_index("forecast_case_id").loc[cases.index]
        begin = pd.to_datetime(part.issue_utc, utc=True)
        end = pd.to_datetime(part.outcome_end_utc, utc=True)
        native = pd.to_datetime(cases.issue_tai, format="%Y.%m.%d_%H:%M:%S_TAI")
        check(known_utc(native).equals(begin), "All issue times")
        check(known_utc(native + pd.Timedelta(hours=int(horizon))).equals(end), "All elapsed-hour endpoints")
        check(part.upstream_label_48h.eq(cases.upstream_label_48h).all(), "Upstream labels preserved")
        expected = {scope: [set() for _ in range(len(cases))] for scope in ["primary", "patch", "unknown"]}
        patch = [{int(x) for x in str(value).split(",")} for value in cases.NOAA_ARS]
        by_patch = {}
        for index, regions in enumerate(patch):
            for region in regions:
                by_patch.setdefault(region, []).append(index)
        primary = cases.NOAA_AR_clean.to_numpy()
        begin_ns, end_ns = begin.astype("int64").to_numpy(), end.astype("int64").to_numpy()
        for event in mx.itertuples(index=False):
            inside = (begin_ns < event.start.value) & (end_ns >= event.start.value)
            if pd.isna(event.noaa_full_id_candidate):
                for i in np.flatnonzero(inside):
                    expected["unknown"][i].add(event.event_id)
            else:
                for i in np.flatnonzero(inside & (primary == event.noaa_full_id_candidate)):
                    expected["primary"][i].add(event.event_id)
                indices = np.array(by_patch.get(int(event.noaa_full_id_candidate), []), dtype=int)
                for i in indices[inside[indices]]:
                    expected["patch"][i].add(event.event_id)
        for scope, id_col, count_col in [("primary", "primary_event_ids", "primary_event_count"),
                ("patch", "patch_event_ids", "patch_event_count"),
                ("unknown", "unassigned_region_mx_event_ids", "unassigned_region_mx_count")]:
            saved = [set(str(x).split(";")) if pd.notna(x) else set() for x in part[id_col]]
            check(saved == expected[scope], f"All event IDs {horizon}/{scope}")
            check(np.array_equal(part[count_col], [len(x) for x in saved]), f"All event counts {scope}")
        within = (begin >= pd.Timestamp(original_summary["nominal_start_utc"])) & (end <= pd.Timestamp(original_summary["conservative_nominal_end_utc"]))
        check(np.array_equal(within, part.nominal_span_contains_window), "Nominal span flags")
        for scope in ["primary", "patch"]:
            calculated = [(-1 if not covered else 1 if hits else -1 if unknown else 0)
                for covered, hits, unknown in zip(within, expected[scope], expected["unknown"])]
            check(np.array_equal(part[f"candidate_{scope}_label"].fillna(-1).to_numpy(), calculated), f"All labels {horizon}/{scope}")
    result = {"status": "full_computational_cross_check_passed_provisional_only", "daily_reports": len(receipt["sources"]),
        "raw_report_rows_checked": len(rows), "accepted_associations_checked": accepted_count,
        "event_decisions_checked": len(decisions), "candidate_case_horizon_rows_checked": len(labels),
        "scopes_checked": ["primary", "patch"], "all_event_links_checked": True,
        "outcome_sha256": sha(args.outcome_dir / "candidate_outcomes.csv.gz"),
        "limitations": ["Computational agreement is not independent physical-observation validation.",
            "Continuous event detection, class-convention choice, historical availability and model performance remain unverified."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
