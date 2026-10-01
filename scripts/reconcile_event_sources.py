"""Reconcile event-source evidence without replacing any upstream research labels.

Only exact-start/exact-peak, unique-science-peak, unanimous known-region matches
can propose filling a missing science-region field. Peak-only and conflicting
matches remain review candidates. Shared HER lineage is not independent evidence.
"""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

try:
    from scripts.build_dataset_inventory import require_unique, tai_to_utc, write_csv
    from scripts.build_candidate_outcomes import label_cases
    from scripts.prepare_dataset_sources import digest
except ModuleNotFoundError:
    from build_dataset_inventory import require_unique, tai_to_utc, write_csv
    from build_candidate_outcomes import label_cases
    from prepare_dataset_sources import digest

ANNUAL_ROW = re.compile(r"^\s*(\d{1,2}-[A-Za-z]{3}-\d{4})\s+(\d{2}:\d{2})\s+(\d{2}:\d{2})\s+(\d{2}:\d{2})\s+([ABCMX]\d+(?:\.\d+)?)\s*(.*)$")
SEP_SHA = "b8ff6e145c1c45648560e229ff796d525992b7a0af9248d0fdae514baddf34a9"
EXT_SHA = "a39f94935e9d11b727b689ab06049aadb68527d5797ed558c3da53d16aaa1472"


def parse_annual(path):
    lines = path.read_text().splitlines()
    declared = re.search(r"(\d+) events were retrieved", "\n".join(lines[:8]))
    if not declared:
        raise ValueError("Annual source has no declared row count")
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not re.match(r"^\s*\d{1,2}-[A-Za-z]{3}-\d{4}", line):
            continue
        match = ANNUAL_ROW.fullmatch(line)
        if not match:
            raise ValueError(f"Unparsed event row: {path.name}:{line_number}")
        date, start, peak, end, cls, tail = match.groups()
        day = pd.Timestamp(datetime.strptime(date, "%d-%b-%Y"), tz="UTC")
        clocks = []
        for clock in (start, peak, end):
            hour, minute = map(int, clock.split(":"))
            if not (0 <= hour < 24 and 0 <= minute < 60):
                raise ValueError("Invalid annual clock")
            clocks.append(day + pd.Timedelta(hours=hour, minutes=minute))
        start_time, peak_time, end_time = clocks
        peak_rollover = peak_time < start_time
        if peak_rollover:
            peak_time += pd.Timedelta(days=1)
        end_rollover = end_time < peak_time
        if end_rollover:
            end_time += pd.Timedelta(days=1)
        time_problem = end_time < peak_time or end_time - start_time >= pd.Timedelta(days=1)
        if time_problem:
            end_time = pd.NaT
        tokens = tail.split()
        region = int(tokens[-1]) if tokens and tokens[-1].isdigit() else None
        if region == 0:
            region = None
        rows.append({"reference_id": f"HER-annual:{path.name}:{line_number}",
            "source_family": "HER_annual", "source_file": path.name, "source_row": line_number,
            "start_utc": start_time, "peak_utc": peak_time, "end_utc": end_time,
            "flare_class": cls, "noaa_region": region,
            "clock_rollover_assumed": bool(peak_rollover or end_rollover), "source_time_problem": bool(time_problem)})
    if len(rows) != int(declared[1]):
        raise ValueError("Annual source row-count mismatch")
    return pd.DataFrame(rows)


def parse_hek(path, family):
    raw = pd.read_csv(path)
    return pd.DataFrame({"reference_id": [f"{family}:{i + 2}" for i in range(len(raw))],
        "source_family": family, "source_file": path.name, "source_row": np.arange(len(raw)) + 2,
        "start_utc": pd.to_datetime(raw.event_starttime, utc=True),
        "peak_utc": pd.to_datetime(raw.event_peaktime, utc=True),
        "end_utc": pd.to_datetime(raw.event_endtime, utc=True),
        "flare_class": raw.fl_goescls, "noaa_region": raw.NOAA_AR_clean,
        "clock_rollover_assumed": False, "source_time_problem": False})


def validate_references(refs):
    require_unique(refs, ["reference_id"], "reference ID")
    for col in ("start_utc", "peak_utc", "end_utc"):
        refs[col] = pd.to_datetime(refs[col], utc=True)
    if refs[["start_utc", "peak_utc"]].isna().any().any():
        raise ValueError("Missing reference start/peak")
    if (refs.start_utc < pd.Timestamp("2010-01-01", tz="UTC")).any() or (refs.peak_utc >= pd.Timestamp("2027-01-01", tz="UTC")).any():
        raise ValueError("Reference timestamps outside the checked NOAA normalization epoch")
    refs["source_time_problem"] |= (refs.start_utc > refs.peak_utc) | (refs.end_utc.notna() & (refs.end_utc < refs.peak_utc))
    regions = refs.noaa_region.dropna()
    if not ((regions >= 1) & (regions < 20000) & (regions % 1 == 0)).all():
        raise ValueError("Reference NOAA region outside checked epoch")
    refs["noaa_region_recorded"] = refs.noaa_region.astype("Int64")
    refs["source_region_was_four_digit_suffix"] = refs.noaa_region.between(1, 9999)
    refs.loc[refs.source_region_was_four_digit_suffix, "noaa_region"] += 10000
    refs["noaa_region"] = refs.noaa_region.astype("Int64")
    if not refs.flare_class.astype(str).str.fullmatch(r"[ABCMX]\d+(?:\.\d+)?").all():
        raise ValueError("Invalid reference class")
    refs["is_mx"] = refs.flare_class.str.startswith(("M", "X"))
    return refs


def reconcile(science, refs):
    """Preserve contradictions; never infer a missing region from nearest time alone."""
    frame = science.copy(deep=True)
    for col in ("time", "start_time", "end_time"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    multiplicity = frame.groupby("time").size()
    admissible = refs.loc[~refs.source_time_problem & ~refs.clock_rollover_assumed]
    groups = {time: part for time, part in admissible.groupby("peak_utc")}
    rows = []
    for event in frame.loc[frame.is_mx].itertuples():
        exact = groups.get(event.time, refs.iloc[0:0])
        known = exact.loc[exact.noaa_region.notna()]
        regions = sorted(int(x) for x in known.noaa_region.unique())
        same_start = known.loc[(known.start_utc == event.start_time) & ~known.clock_rollover_assumed]
        unique_peak = multiplicity[event.time] == 1
        unanimous = len(regions) == 1
        candidate = regions[0] if unique_peak and unanimous else None
        strict = candidate is not None and len(same_start) > 0
        original = None if pd.isna(event.noaa_full_id_candidate) else int(event.noaa_full_id_candidate)
        if original is None:
            status = ("missing_region_exact_start_peak_candidate" if strict else
                "missing_region_peak_only_candidate" if candidate is not None else
                "missing_region_conflicting_references" if len(regions) > 1 else
                "missing_region_nonunique_science_peak" if not unique_peak else "missing_region_no_exact_reference")
        else:
            status = ("existing_region_reference_conflict" if regions and any(x != original for x in regions) else
                "existing_region_exact_peak_agreement" if regions else "existing_region_no_known_exact_reference")
        mx_same_region = known.loc[known.is_mx & known.noaa_region.eq(original)] if original is not None else known.iloc[0:0]
        annual_mx = mx_same_region.loc[mx_same_region.source_family == "HER_annual"]
        rows.append({"event_id": event.event_id, "start_utc": event.start_time, "peak_utc": event.time,
            "science_class": event.flare_class, "original_noaa_region": original,
            "science_peak_multiplicity": int(multiplicity[event.time]), "reference_status": status,
            "exact_peak_reference_ids": ";".join(exact.reference_id),
            "exact_start_peak_reference_ids": ";".join(same_start.reference_id),
            "exact_peak_known_regions": ";".join(map(str, regions)),
            "exact_peak_classes": ";".join(sorted(set(exact.flare_class))),
            "candidate_region_from_exact_peak": candidate,
            "proposed_region_fill": candidate if original is None and strict else None,
            "same_region_mx_exact_peak_reference_ids": ";".join(mx_same_region.reference_id),
            "same_region_mx_annual_reference_ids": ";".join(annual_mx.reference_id),
            "same_region_mx_base_hek_reference": bool((mx_same_region.source_family == "HEK_base").any()),
            "same_region_mx_sep_hek_reference": bool((mx_same_region.source_family == "HEK_sep").any()),
            "reference_families": ";".join(sorted(set(exact.source_family))),
            "label_clearance": "provisional_source_reconciliation_only"})
    return pd.DataFrame(rows)


def presence_by_primary(cases, references, starts, ends):
    """Recorded positive evidence only; zero is not a certified negative."""
    presence = np.zeros(len(cases), dtype=bool)
    by_region = {int(region): np.sort(part.start_utc.astype("int64").unique())
        for region, part in references.loc[references.is_mx & references.noaa_region.notna() & ~references.source_time_problem].groupby("noaa_region")}
    start_values = pd.to_datetime(starts, utc=True).astype("int64").to_numpy()
    end_values = pd.to_datetime(ends, utc=True).astype("int64").to_numpy()
    for region, indices in cases.groupby("NOAA_AR_clean").indices.items():
        times = by_region.get(int(region), np.array([], dtype="int64"))
        presence[indices] = np.searchsorted(times, end_values[indices], side="right") > np.searchsorted(times, start_values[indices], side="right")
    return presence.astype("int8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aia-root", required=True, type=Path)
    parser.add_argument("--legacy-hek", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--inventory-dir", required=True, type=Path)
    parser.add_argument("--previous-outcomes-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Output version already exists")
    annual_root = args.aia_root / "private_research_sources/goes_source_audit_20261001"
    annual_audit = json.loads((annual_root / "all_annual_event_source_audit_20261001.json").read_text())
    annual_expected = {x["file"]: x["sha256"] for x in annual_audit["files"]}
    if base64.b64encode(hashlib.md5(args.legacy_hek.read_bytes()).digest()).decode() != "TuwuXKTp1ZY4An6+6Z5W/w==":
        raise ValueError("Legacy HEK object differs from locked generation 1778961084955762")
    sep = args.source_dir / "hek_sep_1790075866552766.csv"
    if digest(sep) != SEP_SHA:
        raise ValueError("September HEK generation mismatch")
    sources = [(args.legacy_hek, "hek_base_1778961084955762.csv", digest(args.legacy_hek), "HEK_base"),
        (sep, sep.name, SEP_SHA, "HEK_sep"),
        (args.aia_root / "archives/2026_readiness_20260930/hek_swpc_mx_extension.csv", "hek_swpc_mx_extension.csv", EXT_SHA, "HEK_extension")]
    sources += [(annual_root / f"goes_events_{year}.txt", f"goes_events_{year}.txt", annual_expected[f"goes_events_{year}.txt"], "HER_annual") for year in range(2010, 2026)]
    staged, parts = [], []
    for source, name, expected, family in sources:
        if digest(source) != expected:
            raise ValueError(f"Reference hash changed: {name}")
        destination = args.source_dir / name
        if destination.exists() and digest(destination) != expected:
            raise ValueError(f"Existing staged reference changed: {name}")
        if not destination.exists():
            shutil.copyfile(source, destination)
        staged.append({"filename": name, "family": family, "bytes": destination.stat().st_size, "sha256": expected})
        parts.append(parse_annual(destination) if family == "HER_annual" else parse_hek(destination, family))
    refs = validate_references(pd.concat(parts, ignore_index=True))
    inv_summary = json.loads((args.inventory_dir / "build_summary.json").read_text())
    inv_path = args.inventory_dir / "forecast_case_inventory.csv.gz"
    previous_summary = json.loads((args.previous_outcomes_dir / "build_summary.json").read_text())
    if digest(inv_path) != inv_summary["manifest_sha256"] or digest(inv_path) != previous_summary["inventory_sha256"]:
        raise ValueError("Inventory lineage mismatch")
    for name in ["normalized_events.csv.gz", "candidate_outcomes.csv.gz"]:
        if digest(args.previous_outcomes_dir / name) != previous_summary["output_sha256"][name]:
            raise ValueError("Previous candidate output changed")
    cases = pd.read_csv(inv_path, low_memory=False)
    science = pd.read_csv(args.previous_outcomes_dir / "normalized_events.csv.gz", low_memory=False)
    audit = reconcile(science, refs)
    enriched = science.copy(deep=True)
    enriched["original_noaa_full_id"] = enriched.noaa_full_id_candidate
    fills = audit.dropna(subset=["proposed_region_fill"]).set_index("event_id").proposed_region_fill
    proposed = enriched.event_id.map(fills)
    if (proposed.notna() & enriched.noaa_full_id_candidate.notna()).any():
        raise ValueError("A proposed fill would overwrite an existing science association")
    enriched["noaa_full_id_candidate"] = enriched.noaa_full_id_candidate.fillna(proposed).astype("Int64")
    enriched["region_provenance"] = np.where(proposed.notna(), "proposed_exact_start_peak_reference_fill", "original_science_catalogue")
    for col in ("time", "start_time", "end_time"):
        enriched[col] = pd.to_datetime(enriched[col], utc=True)
    # Reproduce the old catalogue under both clock interpretations. These are
    # diagnostic positive-presence fields, not automatic corrected negatives.
    native = pd.to_datetime(cases.issue_tai, format="%Y.%m.%d_%H:%M:%S_TAI")
    # The leap table is pinned alongside the original source stage, recorded in the inventory receipt.
    leap_file = args.inventory_dir.parent.parent / "raw/gray_box_inventory_v1/Leap_Second.dat"
    if digest(leap_file) != inv_summary["leap_table_sha256"]:
        raise ValueError("Leap table mismatch")
    utc48end = tai_to_utc((native + pd.Timedelta(hours=48)).dt.strftime("%Y.%m.%d_%H:%M:%S_TAI"), leap_file)
    diagnostics = cases[["forecast_case_id", "source_sample_id", "issue_utc", "HARPNUM", "NOAA_AR_clean", "input_cohort", "upstream_matched_three_slot_inputs", "upstream_label_48h"]].copy()
    diagnostics["input_cohort"] = diagnostics.input_cohort.fillna("not_in_matched_inputs")
    for family in ["HEK_base", "HEK_sep", "HER_annual"]:
        part = refs.loc[refs.source_family == family].copy()
        part["noaa_region"] = part.noaa_region_recorded
        diagnostics[f"{family}_recorded_presence_48h_utc"] = presence_by_primary(cases, part, cases.issue_utc, utc48end)
    old = refs.loc[refs.source_family.isin(["HEK_base", "HEK_extension"])].copy()
    old["noaa_region"] = old.noaa_region_recorded
    diagnostics["legacy_combined_presence_48h_utc"] = presence_by_primary(cases, old, cases.issue_utc, utc48end)
    diagnostics["legacy_combined_presence_48h_native_clock"] = presence_by_primary(cases, old, native.dt.tz_localize("UTC"), (native + pd.Timedelta(hours=48)).dt.tz_localize("UTC"))
    old48 = pd.read_csv(args.previous_outcomes_dir / "candidate_outcomes.csv.gz", low_memory=False)
    old48 = old48.loc[old48.horizon_hours == 48, ["forecast_case_id", "candidate_primary_label", "primary_event_ids"]]
    diagnostics = diagnostics.merge(old48, on="forecast_case_id", validate="one_to_one")
    corroborated = set(audit.loc[(audit.original_noaa_region.notna()) & (audit.same_region_mx_annual_reference_ids != "") & ~audit.same_region_mx_base_hek_reference, "event_id"])
    diagnostics["science_positive_corroborated_by_annual_not_base"] = diagnostics.primary_event_ids.fillna("").map(lambda value: bool(set(value.split(";")) & corroborated))
    diagnostics["old_negative_with_corroborated_science_positive"] = diagnostics.upstream_label_48h.eq(0) & diagnostics.candidate_primary_label.eq(1) & diagnostics.science_positive_corroborated_by_annual_not_base
    diagnostics["legacy_replay_disagrees_utc"] = diagnostics.legacy_combined_presence_48h_utc.ne(diagnostics.upstream_label_48h)
    diagnostics["legacy_replay_disagrees_native_clock"] = diagnostics.legacy_combined_presence_48h_native_clock.ne(diagnostics.upstream_label_48h)
    diagnostics["year"] = pd.to_datetime(diagnostics.issue_utc, utc=True).dt.year
    replay_summary = diagnostics.groupby(["year", "input_cohort"]).agg(cases=("forecast_case_id", "size"),
        upstream_positive=("upstream_label_48h", "sum"), legacy_utc_disagreements=("legacy_replay_disagrees_utc", "sum"),
        legacy_native_clock_disagreements=("legacy_replay_disagrees_native_clock", "sum"),
        old_negatives_with_corroborated_science_positive=("old_negative_with_corroborated_science_positive", "sum")).reset_index()
    # Keep source-enriched candidate labels separate from v1 and every upstream experiment.
    label_parts, changes = [], []
    for horizon in [48, 72]:
        ends = tai_to_utc((native + pd.Timedelta(hours=horizon)).dt.strftime("%Y.%m.%d_%H:%M:%S_TAI"), leap_file)
        labels = label_cases(cases, enriched, ends, pd.Timestamp(previous_summary["nominal_start_utc"]), pd.Timestamp(previous_summary["conservative_nominal_end_utc"]))
        labels["horizon_hours"] = horizon
        labels["issue_utc"] = cases.issue_utc
        labels["outcome_end_utc"] = ends
        labels["label_status"] = "provisional_with_proposed_reference_region_fills"
        labels["training_ready"] = False
        labels["new_experiment_role"] = "unassigned"
        label_parts.append(labels)
    labels = pd.concat(label_parts, ignore_index=True)
    old_labels = pd.read_csv(args.previous_outcomes_dir / "candidate_outcomes.csv.gz", low_memory=False)
    compared = labels.merge(old_labels[["forecast_case_id", "horizon_hours", "candidate_primary_label", "candidate_patch_label"]], on=["forecast_case_id", "horizon_hours"], validate="one_to_one", suffixes=("_enriched", "_v1"))
    for horizon, part in compared.groupby("horizon_hours"):
        for scope in ["primary", "patch"]:
            a, b = part[f"candidate_{scope}_label_v1"], part[f"candidate_{scope}_label_enriched"]
            if (a.notna() & (b.isna() | a.ne(b))).any():
                raise ValueError("Region-fill sensitivity changed an already-resolved label")
            changes.append({"horizon_hours": int(horizon), "scope": scope, "previous_unresolved": int(a.isna().sum()),
                "remaining_unresolved": int(b.isna().sum()), "new_candidate_positives": int((a.isna() & b.eq(1)).sum()),
                "new_candidate_negatives": int((a.isna() & b.eq(0)).sum())})
    args.output_dir.mkdir(parents=True)
    outputs = {"reference_events.csv.gz": refs, "event_reconciliation.csv.gz": audit, "source_enriched_events.csv.gz": enriched,
        "label_lineage_diagnostics.csv.gz": diagnostics, "reference_enriched_candidate_outcomes.csv.gz": labels}
    for name, frame in outputs.items():
        write_csv(frame, args.output_dir / name)
    replay_summary.to_csv(args.output_dir / "label_lineage_by_year_cohort.csv", index=False)
    yearly = audit.assign(year=pd.to_datetime(audit.peak_utc, utc=True).dt.year).groupby(["year", "reference_status"]).size().rename("science_mx_events").reset_index()
    yearly.to_csv(args.output_dir / "event_reconciliation_by_year.csv", index=False)
    source_profile = refs.assign(year=refs.start_utc.dt.year).groupby(["source_family", "year"]).agg(rows=("reference_id", "size"), mx=("is_mx", "sum"), region_present=("noaa_region", "count"), rollover_assumed=("clock_rollover_assumed", "sum"), time_problem=("source_time_problem", "sum")).reset_index()
    source_profile.to_csv(args.output_dir / "reference_source_profile.csv", index=False)
    report = {"built_utc": datetime.now(timezone.utc).isoformat(), "status": "event_source_reconciliation_executed_not_final_truth",
        "inventory_sha256": digest(inv_path), "previous_outcome_summary_sha256": digest(args.previous_outcomes_dir / "build_summary.json"),
        "reference_sources": staged, "reference_rows": len(refs), "science_mx_events": len(audit),
        "reference_rows_with_time_problems": int(refs.source_time_problem.sum()),
        "reference_rows_with_assumed_rollovers": int(refs.clock_rollover_assumed.sum()),
        "reference_rows_with_four_digit_region_suffix": int(refs.source_region_was_four_digit_suffix.sum()),
        "reference_status_counts": {str(k): int(v) for k, v in audit.reference_status.value_counts().items()},
        "proposed_region_fills": len(fills), "label_sensitivity": changes,
        "upstream_label_replay_disagreements_utc": int(diagnostics.legacy_replay_disagrees_utc.sum()),
        "upstream_label_replay_disagreements_native_clock": int(diagnostics.legacy_replay_disagrees_native_clock.sum()),
        "old_negative_cases_with_corroborated_science_positive": int(diagnostics.old_negative_with_corroborated_science_positive.sum()),
        "matched_old_negative_cases_with_corroborated_science_positive": int((diagnostics.old_negative_with_corroborated_science_positive & diagnostics.upstream_matched_three_slot_inputs).sum()),
        "output_sha256": {p.name: digest(p) for p in sorted(args.output_dir.iterdir())},
        "limitations": ["HER/HEK products share lineage; agreement is not an independent physical observation.",
            "Exact timestamp matches provide source-backed candidates, not adjudicated physical-event truth.",
            "Science classes/start times remain unchanged; no operational-to-science class rescaling is inferred.",
            "Annual clock rollovers are minimal-forward assumptions and are excluded as strict fill evidence.",
            "Annual files omit observed December-31 entries and the 2025 file ends in April; none certifies continuous coverage.",
            "All negatives remain provisional; observing coverage, region/crop scope and final experiment splits are unresolved.",
            "The pre-existing AIA and ASR datasets, labels, jobs and results are not changed."]}
    (args.output_dir / "reconciliation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("reference_sources", "output_sha256", "limitations")}, indent=2))


if __name__ == "__main__":
    main()
