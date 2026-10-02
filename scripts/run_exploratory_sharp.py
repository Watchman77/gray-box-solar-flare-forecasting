"""CPU logistic baseline for the aligned package; no GPU or new data download.

Default is a runnable preparation check. --fit executes a retrospective,
provisional-label experiment, not a confirmatory Gray-Box evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.dataset_io import load_dataset
except ModuleNotFoundError:
    from dataset_io import load_dataset


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


def fit_logistic(x, y, penalty=0.01, max_iter=40):
    """Newton steps with backtracking for mean log loss + L2/2 (no intercept penalty)."""
    if set(np.unique(y)) != {0, 1} or not np.isfinite(x).all() or penalty <= 0:
        raise ValueError("Need finite features, both binary classes and positive L2")
    design = np.column_stack([np.ones(len(x)), x])
    ridge = np.full(design.shape[1], penalty)
    ridge[0] = 0
    beta = np.zeros(design.shape[1])
    beta[0] = np.log(y.mean() / (1 - y.mean()))

    def loss(value):
        score = design @ value
        return np.mean(np.logaddexp(0, score) - y * score) + np.sum(ridge * value**2) / 2

    converged = False
    for iteration in range(max_iter):
        p = sigmoid(design @ beta)
        gradient = design.T @ (p - y) / len(y) + ridge * beta
        if np.max(np.abs(gradient)) < 1e-7:
            converged = True
            break
        hessian = (design.T * (p * (1 - p))) @ design / len(y) + np.diag(ridge)
        step = np.linalg.solve(hessian, gradient)
        previous = loss(beta)
        scale = 1.0
        while scale >= 2**-20 and loss(beta - scale * step) > previous - 1e-4 * scale * (gradient @ step):
            scale *= 0.5
        if scale < 2**-20:
            raise ValueError("Optimizer line search failed")
        beta -= scale * step
    if not converged:
        p = sigmoid(design @ beta)
        converged = np.max(np.abs(design.T @ (p - y) / len(y) + ridge * beta)) < 1e-7
    return beta, {"iterations": iteration + 1, "converged": bool(converged), "penalized_train_loss": float(loss(beta))}


def preprocess(raw, train):
    flat = np.asarray(raw).reshape(len(raw), -1).copy()
    missing = ~np.isfinite(flat)
    flat[missing] = np.nan
    flat = np.sign(flat) * np.log1p(np.abs(flat))
    medians = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in flat[train].T])
    flat = np.where(missing, medians, flat)
    mean, std = flat[train].mean(axis=0), flat[train].std(axis=0)
    std[std == 0] = 1
    # Only include masks with variation in training; never use test values to select columns.
    mask_columns = np.flatnonzero(missing[train].any(axis=0) & ~missing[train].all(axis=0))
    x = np.column_stack([(flat - mean) / std, missing[:, mask_columns].astype(float)])
    return x, {"median": medians, "mean": mean, "std": std, "mask_columns": mask_columns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=[48, 72], default=72)
    parser.add_argument("--scope", choices=["primary", "patch"], default="primary")
    parser.add_argument("--fit", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Experiment output already exists")
    data = load_dataset(args.dataset, args.horizon, args.scope, verify=True)
    index, y, known = data["index"], data["labels"], data["known"]
    master = pd.read_csv(args.dataset / "master_cases.csv.gz", low_memory=False).set_index("forecast_case_id").loc[index.forecast_case_id]
    issue = pd.to_datetime(index.issue_utc, utc=True)
    end = pd.to_datetime(master[f"outcome_end_utc_{args.horizon}h"], utc=True).to_numpy()
    cutoff = pd.Timestamp("2020-01-01", tz="UTC")
    roles = np.full(len(index), "unused", dtype=object)
    roles[(issue < cutoff).to_numpy() & (end <= cutoff) & known] = "train_cycle24"
    roles[(issue >= pd.Timestamp("2021-01-01", tz="UTC")).to_numpy() & (issue < pd.Timestamp("2026-01-01", tz="UTC")).to_numpy() & known] = "retrospective_cycle25"
    roles[(issue >= pd.Timestamp("2026-01-01", tz="UTC")).to_numpy() & known] = "supplementary_2026"
    roles[~known] = "unknown_label_excluded"
    train = roles == "train_cycle24"
    if train.sum() == 0 or set(np.unique(y[train])) != {0, 1}:
        raise ValueError("Insufficient development support")
    plan = index[["forecast_case_id", "tensor_row", "issue_utc", "region_component_id"]].copy()
    plan["exploratory_role"] = roles
    plan["label"] = pd.Series(y).mask(~known).astype("Int64")
    summary = {"status": "preparation_complete" if not args.fit else "exploratory_fit_complete",
        "dataset_version": data["manifest"]["dataset_version"], "target_version": data["manifest"]["target_version"],
        "horizon_hours": args.horizon, "scope": args.scope, "model": "L2_logistic_flat_three_slot_SHARP",
        "training_rule": "Known candidate outcomes matured by 2020-01-01 UTC (Cycle-24 2010-2019 input cohort); fixed penalty 0.01; no class rebalancing.",
        "role_counts": pd.Series(roles).value_counts().to_dict(),
        "limitations": ["Exploratory candidate labels; provisional zeros and unknown-label exclusions can bias evaluation.",
            "Previously inspected Cycle-25 years are retrospective evaluation, not untouched or prospective confirmation.",
            "No AIA pixel model, continuous GOES predictor, calibration, conformal uncertainty or safety policy is fitted.",
            "Calendar roles do not assert AR-disjoint or event-disjoint validation; component overlaps are reported."]}
    train_regions = set(index.loc[train, "region_component_id"])
    summary["shared_components_with_training"] = {role: len(train_regions & set(index.loc[roles == role, "region_component_id"]))
        for role in ["retrospective_cycle25", "supplementary_2026"]}
    args.output_dir.mkdir(parents=True)
    if args.fit:
        x, transformation = preprocess(data["sharp"], train)
        beta, optimizer = fit_logistic(x[train], y[train].astype(float))
        if not optimizer["converged"]:
            raise ValueError("Optimizer did not converge; do not publish metrics")
        probability = sigmoid(beta[0] + x @ beta[1:])
        plan["probability"] = probability
        np.savez(args.output_dir / "model.npz", beta=beta, **transformation)
        summary["optimizer"] = optimizer
        summary["metrics"] = []
        for role in ["train_cycle24", "retrospective_cycle25", "supplementary_2026"]:
            selected = roles == role
            if not selected.any():
                continue
            target, p = y[selected], probability[selected]
            p_clip = np.clip(p, 1e-15, 1 - 1e-15)
            summary["metrics"].append({"role": role, "cases": int(selected.sum()), "positive": int(target.sum()),
                "brier": float(np.mean((target-p)**2)),
                "log_loss": float(-np.mean(target*np.log(p_clip)+(1-target)*np.log1p(-p_clip)))})
    plan.to_csv(args.output_dir / "cases.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
