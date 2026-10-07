"""Quantify the effect of the two 2014 train/model-validation AR boundary cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    pred = p >= threshold
    pos = y == 1
    neg = y == 0
    tp = int(np.sum(pred & pos))
    fn = int(np.sum(~pred & pos))
    fp = int(np.sum(pred & neg))
    tn = int(np.sum(~pred & neg))
    return {
        "cases": int(len(y)),
        "positive": int(y.sum()),
        "tss": float(tp / (tp + fn) - fp / (fp + tn)),
        "brier": float(np.mean((p - y) ** 2)),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--horizon", type=int, choices=(3, 24), required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    base = root / f"results/{args.horizon}h_graybox"
    audit = base / "validation_audit_v1/audit_cases.csv.gz"
    search = base / "sharp_threshold_audit_v1/threshold_search_model_validation.csv"
    receipt = base / "sharp_threshold_audit_v1/summary.json"
    output = base / "validation_audit_v1/boundary_sensitivity.csv"
    for path in (audit, search, receipt):
        if not path.is_file():
            raise FileNotFoundError(path)

    frame = pd.read_csv(audit, low_memory=False)
    search_grid = pd.read_csv(search, low_memory=False)
    original_threshold = float(json.loads(receipt.read_text())["selected_threshold"])
    train_regions = set(frame.loc[frame.role == "train", "region_component_id"].dropna().astype(str))
    model_regions = set(frame.loc[frame.role == "model_validation", "region_component_id"].dropna().astype(str))
    overlap = train_regions & model_regions
    model = frame[frame.role == "model_validation"].copy()
    model["overlap_region"] = model["region_component_id"].astype(str).isin(overlap)
    model_clean = model[~model.overlap_region].copy()
    thresholds = sorted(pd.to_numeric(search_grid.threshold, errors="raise").unique())

    rows: list[dict[str, object]] = []
    for subset_name, subset in (("all_model_validation", model), ("exclude_train_overlap", model_clean)):
        y = subset.label.to_numpy(dtype=int)
        p = subset.probability_gru_mean.to_numpy(dtype=float)
        grid = [(float(t), binary_metrics(y, p, float(t))["tss"]) for t in thresholds]
        selected_threshold, selected_tss = max(grid, key=lambda item: (item[1], item[0]))
        for threshold_name, threshold in (("original", original_threshold), ("reselected_after_exclusion", selected_threshold)):
            m = binary_metrics(y, p, threshold)
            rows.append({
                "horizon_hours": args.horizon,
                "subset": subset_name,
                "threshold_source": threshold_name,
                "threshold": threshold,
                "selected_tss_after_exclusion": selected_tss,
                "overlap_regions_excluded": ";".join(sorted(overlap)) if subset_name != "all_model_validation" else "",
                **m,
            })
        for role in ("policy_validation", "retrospective_cycle25", "supplementary_2026"):
            eval_frame = frame[frame.role == role]
            y_eval = eval_frame.label.to_numpy(dtype=int)
            p_eval = eval_frame.probability_gru_mean.to_numpy(dtype=float)
            for threshold_name, threshold in (("original", original_threshold), ("reselected_after_exclusion", selected_threshold)):
                rows.append({
                    "horizon_hours": args.horizon,
                    "subset": role,
                    "threshold_source": threshold_name,
                    "threshold": threshold,
                    "selected_tss_after_exclusion": selected_tss,
                    "overlap_regions_excluded": ";".join(sorted(overlap)) if subset_name != "all_model_validation" else "",
                    **binary_metrics(y_eval, p_eval, threshold),
                })
    pd.DataFrame(rows).to_csv(output, index=False)
    print(json.dumps({"horizon_hours": args.horizon, "overlap_regions": sorted(overlap), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
