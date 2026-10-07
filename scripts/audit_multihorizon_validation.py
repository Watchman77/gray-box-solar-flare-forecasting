"""Audit short-horizon SHARP results without fitting or retuning.

This audit joins an immutable training prediction file to the aligned case
metadata and the horizon-specific provisional labels.  It verifies the
chronological role reconstruction, active-region separation and event
separation, then reports role-level metrics and paired AR-block bootstrap
intervals.  It deliberately does not fit a model, recalibrate probabilities,
select a threshold, or modify a policy.

Example (run from the repository root)::

    python scripts/audit_multihorizon_validation.py --horizon 24 \
      --bootstrap-reps 2000

The output directory is immutable by default; use a new version if inputs or
settings change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROLE_BLOCKS = [
    ("train", "2010-01-01", "2014-01-01"),
    ("model_validation", "2014-01-01", "2014-07-01"),
    ("probability_calibration", "2014-07-01", "2015-01-01"),
    ("conformal_calibration", "2015-01-01", "2015-07-01"),
    ("policy_validation", "2015-07-01", "2020-01-01"),
    ("retrospective_cycle25", "2021-01-01", "2026-01-01"),
    ("supplementary_2026", "2026-01-01", "2027-01-01"),
]
BOOTSTRAP_ROLES = ("policy_validation", "retrospective_cycle25", "supplementary_2026")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def assign_roles(issue_utc: pd.Series, labels: pd.Series, tensor_row: pd.Series) -> pd.Series:
    roles = pd.Series("outside_blocks", index=issue_utc.index, dtype="object")
    for role, start, end in ROLE_BLOCKS:
        mask = (issue_utc >= pd.Timestamp(start, tz="UTC")) & (
            issue_utc < pd.Timestamp(end, tz="UTC")
        )
        roles.loc[mask] = role
    roles.loc[labels.isna()] = "unknown_label_excluded"
    roles.loc[tensor_row < 0] = "missing_input_excluded"
    return roles


def split_event_ids(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return set()
    # Current receipts use a single ID, while future versions may preserve a
    # semicolon/comma-delimited list.  Treat both forms conservatively.
    out = {part.strip() for part in text.replace(",", ";").split(";")}
    return {item for item in out if item and item.lower() not in {"nan", "none"}}


def safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute threshold-free and thresholded metrics for one sample."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    positive = y == 1
    negative = y == 0
    tp = int(np.sum(pred[positive] == 1))
    fn = int(np.sum(pred[positive] == 0))
    fp = int(np.sum(pred[negative] == 1))
    tn = int(np.sum(pred[negative] == 0))
    return {
        "cases": int(len(y)),
        "positive": int(np.sum(positive)),
        "prevalence": float(np.mean(y)) if len(y) else float("nan"),
        "brier": float(np.mean((p - y) ** 2)) if len(y) else float("nan"),
        "average_precision": float(average_precision_score(y, p))
        if len(np.unique(y)) == 2
        else float("nan"),
        "roc_auc": float(roc_auc_score(y, p))
        if len(np.unique(y)) == 2
        else float("nan"),
        "tss": safe_rate(tp, tp + fn) - safe_rate(fp, fp + tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def role_metrics(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for role, group in frame.groupby("role", sort=False):
        usable = group[group["label"].isin([0, 1]) & group["probability"].notna()]
        if usable.empty:
            continue
        row = {"role": role, **binary_metrics(usable["label"].to_numpy(), usable["probability"].to_numpy(), threshold)}
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_metrics(
    frame: pd.DataFrame,
    threshold: float,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    """Paired active-region block bootstrap for the three evaluation roles."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for role in BOOTSTRAP_ROLES:
        group = frame[
            (frame["role"] == role)
            & frame["label"].isin([0, 1])
            & frame["probability"].notna()
            & frame["region_component_id"].notna()
        ].copy()
        region_codes, regions = pd.factorize(group["region_component_id"].astype(str), sort=False)
        require(len(regions) > 1, f"Too few region components for bootstrap: {role}")
        y = group["label"].to_numpy(dtype=int)
        p = group["probability"].to_numpy(dtype=float)
        for rep in range(reps):
            sampled_codes = rng.integers(0, len(regions), size=len(regions))
            multiplicity = np.bincount(sampled_codes, minlength=len(regions))
            row_index = np.repeat(np.arange(len(group)), multiplicity[region_codes])
            metrics = binary_metrics(y[row_index], p[row_index], threshold)
            rows.append({"role": role, "replicate": rep, **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, choices=(3, 24), required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261007)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    aligned = root / "data/processed/gray_box_aligned_v1"
    predictions = args.predictions or root / f"results/{args.horizon}h_graybox/sharp_training_v1/predictions.csv.gz"
    labels = args.labels or root / f"data/processed/gray_box_multihorizon_labels_{args.horizon}h_v1/candidate_labels_{args.horizon}h.csv.gz"
    threshold_file = root / f"results/{args.horizon}h_graybox/sharp_threshold_audit_v1/summary.json"
    output = args.output_dir or root / f"results/{args.horizon}h_graybox/validation_audit_v1"
    require(args.bootstrap_reps > 0, "--bootstrap-reps must be positive")
    require(not output.exists(), f"Refusing to overwrite existing output: {output}")
    for path in (predictions, labels, threshold_file, aligned / "master_cases.csv.gz"):
        require(path.is_file(), f"Missing input: {path}")

    threshold_record = json.loads(threshold_file.read_text())
    threshold = float(threshold_record["selected_threshold"])

    cases = pd.read_csv(aligned / "master_cases.csv.gz", low_memory=False)
    label = pd.read_csv(labels, low_memory=False)
    pred = pd.read_csv(predictions, low_memory=False)
    for name, table in (("cases", cases), ("labels", label), ("predictions", pred)):
        require("forecast_case_id" in table.columns, f"{name} lacks forecast_case_id")
        table["forecast_case_id"] = table["forecast_case_id"].astype(str)
        require(table["forecast_case_id"].is_unique, f"Duplicate forecast_case_id in {name}")

    required_cases = ["forecast_case_id", "issue_utc", "region_component_id", "tensor_row"]
    required_labels = ["forecast_case_id", "candidate_primary_label", "primary_event_ids", "patch_event_ids", "unknown_mx_event_ids"]
    for column in required_cases:
        require(column in cases.columns, f"Aligned cases lack {column}")
    for column in required_labels:
        require(column in label.columns, f"Labels lack {column}")
    for column in ("role", "label", "probability_gru_mean"):
        require(column in pred.columns, f"Predictions lack {column}")

    frame = cases[required_cases].merge(
        label[required_labels], on="forecast_case_id", how="inner", validate="one_to_one"
    ).merge(
        pred[["forecast_case_id", "role", "label", "probability_gru_mean"]],
        on="forecast_case_id", how="inner", validate="one_to_one", suffixes=("", "_prediction")
    )
    require(len(frame) == len(pred), "Prediction/case alignment loss")
    frame["issue_utc"] = pd.to_datetime(frame["issue_utc"], utc=True, errors="coerce")
    require(frame["issue_utc"].notna().all(), "Unparseable issue times")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    expected_roles = assign_roles(
        frame["issue_utc"],
        frame["candidate_primary_label"],
        pd.to_numeric(frame["tensor_row"], errors="raise"),
    )
    role_mismatch = int(np.sum(expected_roles.to_numpy() != frame["role"].astype(str).to_numpy()))
    require(role_mismatch == 0, f"Prediction roles disagree with protocol for {role_mismatch} cases")
    label_mismatch = int(np.sum(frame["label"].to_numpy() != frame["candidate_primary_label"].fillna(-1).astype(int).to_numpy()))
    require(label_mismatch == 0, f"Prediction labels disagree with labels for {label_mismatch} cases")
    frame["probability"] = pd.to_numeric(frame["probability_gru_mean"], errors="coerce")

    output.mkdir(parents=True)
    frame.to_csv(output / "audit_cases.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})

    role_rows: list[dict[str, object]] = []
    for role, group in frame.groupby("role", sort=False):
        usable = group[group["label"].isin([0, 1])]
        role_rows.append(
            {
                "role": role,
                "cases": int(len(group)),
                "known_label_cases": int(len(usable)),
                "positive": int(usable["label"].sum()),
                "prevalence": float(usable["label"].mean()) if len(usable) else float("nan"),
                "region_components": int(group["region_component_id"].nunique(dropna=True)),
            }
        )
    pd.DataFrame(role_rows).to_csv(output / "role_summary.csv", index=False)

    train_regions = set(frame.loc[frame["role"] == "train", "region_component_id"].dropna().astype(str))
    ar_rows: list[dict[str, object]] = []
    for role, group in frame.groupby("role", sort=False):
        role_regions = set(group["region_component_id"].dropna().astype(str))
        overlap = role_regions & train_regions
        ar_rows.append(
            {
                "role": role,
                "role_region_components": int(len(role_regions)),
                "train_region_components": int(len(train_regions)),
                "overlap_with_train": int(len(overlap)),
            }
        )
    pd.DataFrame(ar_rows).to_csv(output / "region_component_overlap.csv", index=False)

    event_rows: list[dict[str, object]] = []
    for event_column in ("primary_event_ids", "patch_event_ids", "unknown_mx_event_ids"):
        role_events: dict[str, set[str]] = {}
        for role, group in frame.groupby("role", sort=False):
            values: set[str] = set()
            for value in group[event_column]:
                values.update(split_event_ids(value))
            role_events[role] = values
        train_events = role_events.get("train", set())
        for role, values in role_events.items():
            event_rows.append(
                {
                    "event_field": event_column,
                    "role": role,
                    "unique_events": int(len(values)),
                    "overlap_with_train": int(len(values & train_events)),
                }
            )
    pd.DataFrame(event_rows).to_csv(output / "event_overlap.csv", index=False)

    role_metrics(frame, threshold).to_csv(output / "role_metrics.csv", index=False)
    bootstrap = bootstrap_metrics(frame, threshold, args.bootstrap_reps, args.bootstrap_seed)
    bootstrap.to_csv(output / "ar_block_bootstrap_replicates.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    interval_rows: list[dict[str, object]] = []
    point = role_metrics(frame, threshold).set_index("role")
    for role, group in bootstrap.groupby("role", sort=False):
        for metric in ("tss", "average_precision", "brier", "roc_auc"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            interval_rows.append(
                {
                    "role": role,
                    "metric": metric,
                    "point_estimate": float(point.loc[role, metric]),
                    "bootstrap_replicates": int(len(values)),
                    "ci_2_5": float(np.quantile(values, 0.025)) if len(values) else float("nan"),
                    "ci_97_5": float(np.quantile(values, 0.975)) if len(values) else float("nan"),
                }
            )
    pd.DataFrame(interval_rows).to_csv(output / "ar_block_bootstrap_intervals.csv", index=False)

    receipt = {
        "status": "completed_validation_audit",
        "horizon_hours": args.horizon,
        "predictions": str(predictions),
        "predictions_sha256": sha256(predictions),
        "labels": str(labels),
        "labels_sha256": sha256(labels),
        "selected_threshold": threshold,
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "role_mismatch_cases": role_mismatch,
        "label_mismatch_cases": label_mismatch,
        "limitations": [
            "Labels remain provisional candidate labels, not confirmatory truth.",
            "This audit validates AR/event separation and uncertainty intervals; it does not add AIA, GOES or fusion predictors.",
            "Event IDs are reported separately for primary, patch and unresolved/unknown fields.",
        ],
    }
    (output / "audit_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
