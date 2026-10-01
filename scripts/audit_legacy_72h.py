"""Replay historical cohort construction, never train or repair the source data.

This is a lineage audit, not operational data clearance. Stored labels and naive
timestamps are preserved; neither event provenance nor UTC is established here.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "MEANGBZ", "MEANGAM", "MEANGBT", "MEANGBH", "MEANJZD", "TOTUSJZ",
    "MEANALP", "MEANJZH", "ABSNJZH", "SAVNCPP", "MEANSHR", "SHRGT45",
    "R_VALUE", "USFLUX", "TOTPOT", "TOTUSJH",
]
TIME, REGION, LABEL = "T_REC_dt", "NOAA_AR", "label_MX_3d"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_cases(raw, lookback=7):
    """Independently reconstruct the notebook's per-region row-window support."""
    required = [TIME, REGION, LABEL] + FEATURES
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if lookback < 2:
        raise ValueError("lookback must be at least two")
    frame = raw.copy()
    frame["source_row_zero_based"] = np.arange(len(frame))
    frame[TIME] = pd.to_datetime(frame[TIME], errors="coerce")
    for feature in FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    if not frame[LABEL].dropna().isin([0, 1]).all():
        raise ValueError("Non-binary labels; refusing the legacy integer coercion")
    clean = frame.dropna(subset=required).sort_values([REGION, TIME]).reset_index(drop=True)
    rows = []
    for region, group in clean.groupby(REGION):
        group = group.sort_values(TIME).reset_index(drop=True)
        for end in range(lookback - 1, len(group)):
            history = group.iloc[end - lookback + 1:end + 1]
            times = history[TIME]
            rows.append({
                "source_row_zero_based": int(group.iloc[end]["source_row_zero_based"]),
                "region_id_as_recorded": int(region),
                "issue_time_as_recorded": times.iloc[-1],
                "history_start_as_recorded": times.iloc[0],
                "label_as_recorded": int(group.iloc[end][LABEL]),
                "history_span_hours": float((times.iloc[-1] - times.iloc[0]).total_seconds() / 3600),
                "largest_history_gap_hours": float(times.diff().dt.total_seconds().max() / 3600),
            })
    if not rows:
        raise ValueError("No complete row windows")
    profile = {
        "raw_rows": len(raw), "raw_columns": len(raw.columns), "clean_rows": len(clean),
        "rows_removed_by_legacy_complete_case_rule": len(raw) - len(clean),
        "clean_duplicate_region_time_rows": int(clean.duplicated([REGION, TIME], keep=False).sum()),
        "clean_nonpositive_region_rows": int((clean[REGION] <= 0).sum()),
        "clean_nonfinite_feature_values": int((~np.isfinite(clean[FEATURES].to_numpy())).sum()),
        "invalid_or_missing_required_counts": {c: int(frame[c].isna().sum()) for c in required},
    }
    return pd.DataFrame(rows), profile


def split_roles(n, rounding):
    if rounding not in {"floor", "round"}:
        raise ValueError("Unknown rounding rule")
    convert = int if rounding == "floor" else lambda x: int(round(x))
    a, b = convert(0.70 * n), convert(0.15 * n)
    return np.array(["train"] * a + ["validation"] * b + ["test"] * (n - a - b))


def summarize(cases, rounding):
    cases = cases.copy()
    cases["role"] = split_roles(len(cases), rounding)
    support, boundaries = {}, {}
    for role, part in cases.groupby("role", sort=False):
        support[role] = {
            "rows": len(part), "positives": int(part.label_as_recorded.sum()),
            "regions_as_recorded": int(part.region_id_as_recorded.nunique()),
            "first_issue_as_recorded": str(part.issue_time_as_recorded.min()),
            "last_issue_as_recorded": str(part.issue_time_as_recorded.max()),
        }
    for before, after in [("train", "validation"), ("validation", "test")]:
        left, right = cases[cases.role == before], cases[cases.role == after]
        boundary = right.issue_time_as_recorded.min()
        boundaries[f"{before}_to_{after}"] = {
            "shared_region_ids": sorted(set(left.region_id_as_recorded) & set(right.region_id_as_recorded)),
            "issue_time_tie_across_boundary": bool(left.issue_time_as_recorded.max() == boundary),
            "earlier_rows_with_72h_outcome_end_after_next_block_start": int(
                ((left.issue_time_as_recorded + pd.Timedelta(hours=72)) > boundary).sum()),
        }
    return {"support": support, "boundaries": boundaries}, cases


def audit(raw):
    cases, profile = build_cases(raw)
    # Preserve the upstream default sort in the two historical candidates. A
    # separate deterministic tie sort diagnoses sensitivity, not a new baseline.
    once = cases.iloc[np.argsort(cases.issue_time_as_recorded.to_numpy())].reset_index(drop=True)
    twice = once.iloc[np.argsort(once.issue_time_as_recorded.to_numpy())].reset_index(drop=True)
    stable = cases.sort_values([
        "issue_time_as_recorded", "region_id_as_recorded", "source_row_zero_based"
    ]).reset_index(drop=True)
    variants, manifests = {}, []
    for name, frame, rounding in [
        ("headline_floor_single_sort", once, "floor"),
        ("supplement_round_second_sort", twice, "round"),
        ("sensitivity_floor_deterministic_ties", stable, "floor"),
    ]:
        variants[name], manifest = summarize(frame, rounding)
        manifest.insert(0, "variant", name)
        manifests.append(manifest)
    profile.update({
        "sequences": len(cases), "positive_sequences": int(cases.label_as_recorded.sum()),
        "sequence_prevalence": float(cases.label_as_recorded.mean()),
        "sequences_with_gap_over_24h": int((cases.largest_history_gap_hours > 24).sum()),
        "sequences_with_gap_over_48h": int((cases.largest_history_gap_hours > 48).sum()),
        "maximum_history_span_hours": float(cases.history_span_hours.max()),
        "nonpositive_region_sequences": int((cases.region_id_as_recorded <= 0).sum()),
    })
    return {"profile": profile, "variants": variants}, pd.concat(manifests, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--compare-copy", type=Path)
    parser.add_argument("--notebook", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report, cases = audit(pd.read_csv(args.source))
    report.update({
        "status": "cohort_lineage_audit_only_no_model_run",
        "source": {"filename": args.source.name, "sha256": sha256(args.source)},
        "upstream_notebook": {"filename": args.notebook.name, "sha256": sha256(args.notebook)},
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "limitations": [
            "Stored timestamps have no established time-scale/availability provenance.",
            "Stored labels are not independently validated against an event catalogue.",
            "The historical default argsort does not explicitly resolve equal-time ties.",
            "Summary agreement does not prove identity of historical case membership or predictions.",
            "Region IDs are used as stored, not reconciled physical region components.",
            "Outcome-boundary diagnostics assume a 72-hour window with zero reporting delay.",
        ],
    })
    if args.compare_copy:
        report["comparison_copy"] = {"sha256": sha256(args.compare_copy), "identical_bytes": sha256(args.source) == sha256(args.compare_copy)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases.to_csv(args.output_dir / "candidate_case_roles.csv", index=False)
    report["candidate_case_roles_sha256"] = sha256(args.output_dir / "candidate_case_roles.csv")
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
