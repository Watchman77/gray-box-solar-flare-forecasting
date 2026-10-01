"""Reconcile recovered arrays with the saved headline table; no model fitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def discrimination(y, score):
    """Non-interpolated AP and pairwise-equivalent ROC AUC, grouping score ties."""
    y, score = np.asarray(y), np.asarray(score)
    if y.ndim != 1 or score.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected aligned one-dimensional binary outcomes and scores")
    if not np.isfinite(score).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite input")
    positives, negatives = int(y.sum()), int((1 - y).sum())
    if not positives or not negatives:
        raise ValueError("Both classes required for this reconciliation")
    order = np.argsort(-score, kind="stable")
    sy, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    tp = np.cumsum(sy)[ends]
    fp = ends + 1 - tp
    recall, precision = tp / positives, tp / (ends + 1)
    ap = np.sum(np.diff(np.r_[0, recall]) * precision)
    auc = np.trapezoid(np.r_[0, recall], np.r_[0, fp / negatives])
    return {"average_precision": float(ap), "roc_auc": float(auc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--case-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.artifact_dir
    read = lambda name: np.load(root / name, allow_pickle=False)
    y, score, stored = read("y_test_72h.npy"), read("test_prob_simple.npy"), read("y_pred_simple.npy")
    table = pd.read_csv(root / "FINAL_CLEAN_72H_MAIN_RESULTS.csv")
    matches = table[table.Model == "Simple/Avg Ensemble"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one headline row")
    headline = matches.iloc[0]
    if score.shape != y.shape or stored.shape != y.shape or not np.isin(stored, [0, 1]).all():
        raise ValueError("Malformed or unaligned arrays")
    if ((score < 0) | (score > 1)).any():
        raise ValueError("Probability out of range")
    metrics = discrimination(y, score)
    counts = {
        "TN": int(((y == 0) & (stored == 0)).sum()),
        "FP": int(((y == 0) & (stored == 1)).sum()),
        "FN": int(((y == 1) & (stored == 0)).sum()),
        "TP": int(((y == 1) & (stored == 1)).sum()),
    }
    metrics["TSS"] = counts["TP"] / (counts["TP"] + counts["FN"]) - counts["FP"] / (counts["FP"] + counts["TN"])
    checks = {"threshold_reproduces_stored_decisions": bool(np.array_equal(stored, score >= float(headline.Threshold)))}
    for metric, table_name in [("average_precision", "PR_AUC"), ("roc_auc", "ROC_AUC"), ("TSS", "Test_TSS")]:
        checks[f"{metric}_matches_saved_table"] = bool(np.isclose(metrics[metric], headline[table_name], rtol=0, atol=1e-8))
    checks["confusion_counts_match_table"] = all(v == int(headline[k]) for k, v in counts.items())
    cases = pd.read_csv(args.case_manifest)
    cases = cases[cases.variant == "headline_floor_single_sort"]
    for role, filename in [("validation", "y_val_72h.npy"), ("test", "y_test_72h.npy")]:
        checks[f"reconstructed_{role}_label_order_matches"] = bool(np.array_equal(
            cases[cases.role == role].label_as_recorded.to_numpy(), read(filename)))
    report = {
        "status": "historical_array_reconciliation_not_model_reproduction",
        "model": "Simple/Avg Ensemble", "horizon_hours": 72,
        "rows": len(y), "positive_rows": int(y.sum()), "prevalence": float(y.mean()),
        "threshold": float(headline.Threshold), "metrics": metrics, "confusion_counts": counts,
        "checks": checks,
        "may_vs_august_array_comparison": {
            "test_labels_identical": bool(np.array_equal(y, read("pi_y_test_72h.npy"))),
            "simple_ensemble_probabilities_identical": bool(np.array_equal(score, read("pi_test_prob_simple.npy"))),
        },
        "limitation": "Identical labels do not uniquely identify case IDs. Source chronology, event labels and saved model weights still require verification.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise SystemExit("Reconciliation has unresolved checks; see saved report")


if __name__ == "__main__":
    main()
