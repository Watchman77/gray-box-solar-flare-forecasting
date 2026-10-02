"""CPU logistic baseline for the aligned package; no GPU or new data download.

Default is a runnable preparation check. --fit executes a retrospective,
provisional-label experiment, not a confirmatory Gray-Box evaluation.
"""

import argparse
import json
from pathlib import Path
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from scripts.dataset_io import preflight_dataset, file_sha256
except ModuleNotFoundError:
    from dataset_io import preflight_dataset, file_sha256


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


def fit_logistic(x, y, penalty=0.01, max_iter=40):
    """Newton steps with backtracking for mean log loss + L2/2 (no intercept penalty)."""
    if x.ndim != 2 or y.shape != (len(x),) or max_iter < 1 or set(np.unique(y)) != {0, 1} or not np.isfinite(x).all() or penalty <= 0:
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
    parameters = {"median": medians, "mean": mean, "std": std, "mask_columns": mask_columns}
    return transform(raw, parameters), parameters


def transform(raw, parameters):
    """Apply a saved training transform without refitting on the forecast batch."""
    flat = np.asarray(raw).reshape(len(raw), -1).copy()
    if flat.shape[1] != len(parameters["mean"]):
        raise ValueError("Forecast feature shape differs from the fitted transform")
    missing = ~np.isfinite(flat)
    flat[missing] = np.nan
    flat = np.sign(flat) * np.log1p(np.abs(flat))
    flat = np.where(missing, parameters["median"], flat)
    result = np.column_stack([(flat - parameters["mean"]) / parameters["std"],
                             missing[:, parameters["mask_columns"]].astype(float)])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite transformed model inputs")
    return result


def assign_roles(issue, end, known, supported):
    """Outcome maturity applies to training; unknown labels never become zeros."""
    cutoff = pd.Timestamp("2020-01-01", tz="UTC")
    roles = np.full(len(issue), "unused", dtype=object)
    roles[(issue < cutoff).to_numpy() & (end <= cutoff) & known] = "train_cycle24"
    roles[(issue >= pd.Timestamp("2021-01-01", tz="UTC")).to_numpy() & (issue < pd.Timestamp("2026-01-01", tz="UTC")).to_numpy() & known] = "retrospective_cycle25"
    roles[(issue >= pd.Timestamp("2026-01-01", tz="UTC")).to_numpy() & known] = "supplementary_2026"
    roles[~known] = "unknown_label_excluded"
    roles[~supported] = "missing_feature_excluded"
    return roles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=[48, 72], default=72)
    parser.add_argument("--scope", choices=["primary", "patch"], default="primary")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse a complete, matching, checksum-verified run")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.resume:
        raise ValueError("Experiment output already exists")
    data = preflight_dataset(args.dataset, args.horizon, args.scope)
    contract = {"manifest_sha256": data["preflight"]["manifest_sha256"],
        "runner_sha256": file_sha256(Path(__file__)),
        "loader_sha256": file_sha256(Path(__file__).with_name("dataset_io.py")),
        "horizon": args.horizon, "scope": args.scope, "fit": args.fit,
        "numpy": np.__version__, "pandas": pd.__version__}
    if args.output_dir.exists():
        saved = json.loads((args.output_dir / "summary.json").read_text())
        if saved.get("run_contract") != contract:
            raise ValueError("Existing run has a different data/code/target contract; choose a new output")
        for name, expected in saved["output_sha256"].items():
            if file_sha256(args.output_dir / name) != expected:
                raise ValueError(f"Existing result changed: {name}; do not reuse it")
        print(json.dumps({"status": "verified_existing_run_reused", "output_dir": str(args.output_dir)}, indent=2))
        return
    index, y, known = data["index"], data["labels"], data["known"]
    master = data["master"].set_index("forecast_case_id").loc[index.forecast_case_id]
    issue = pd.to_datetime(index.issue_utc, utc=True)
    end = pd.to_datetime(master[f"outcome_end_utc_{args.horizon}h"], utc=True).to_numpy()
    supported = ~data["missing"].reshape(len(index), -1).any(axis=1)
    roles = assign_roles(issue, end, known, supported)
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
        "run_contract": contract, "preflight": data["preflight"],
        "limitations": ["Exploratory candidate labels; provisional zeros and unknown-label exclusions can bias evaluation.",
            "Previously inspected Cycle-25 years are retrospective evaluation, not untouched or prospective confirmation.",
            "No AIA pixel model, continuous GOES predictor, calibration, conformal uncertainty or safety policy is fitted.",
            "Calendar roles do not assert AR-disjoint or event-disjoint validation; component overlaps are reported."]}
    train_regions = set(index.loc[train, "region_component_id"])
    summary["shared_components_with_training"] = {role: len(train_regions & set(index.loc[roles == role, "region_component_id"]))
        for role in ["retrospective_cycle25", "supplementary_2026"]}
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=args.output_dir.name + ".incomplete-", dir=args.output_dir.parent))
    try:
        if args.fit:
            x, transformation = preprocess(data["sharp"], train)
            beta, optimizer = fit_logistic(x[train], y[train].astype(float))
            if not optimizer["converged"]:
                raise ValueError("Optimizer did not converge; do not publish metrics")
            probability = sigmoid(beta[0] + x @ beta[1:])
            probability[~supported] = np.nan
            plan["probability"] = probability
            np.savez(staging / "model.npz", beta=beta, **transformation)
            # Exercise actual saved-model inference, including preprocessing.
            with np.load(staging / "model.npz", allow_pickle=False) as saved:
                replay = sigmoid(saved["beta"][0] + transform(data["sharp"], saved) @ saved["beta"][1:])
            if not np.array_equal(replay[supported], probability[supported]):
                raise ValueError("Saved model does not reproduce the fitted predictions")
            summary["saved_model_replay"] = "all_supported_predictions_identical"
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
            population = data["master"][["forecast_case_id", "issue_utc", "inputs_available",
                f"candidate_{args.scope}_label_{args.horizon}h"]].merge(
                    plan[["forecast_case_id", "probability", "exploratory_role"]], on="forecast_case_id", how="left", validate="one_to_one")
            population["forecast_status"] = np.where(population.probability.notna(), "scored_exploratory", "no_supported_sharp_input")
            population.to_csv(staging / "population_predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
            summary["population_rows"] = len(population)
            summary["unscored_missing_input_rows"] = int(population.probability.isna().sum())
        plan.to_csv(staging / "cases.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        summary["output_sha256"] = {p.name: file_sha256(p) for p in staging.iterdir()}
        summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if args.output_dir.exists():
            raise ValueError("Output appeared during execution; refusing to replace it")
        staging.rename(args.output_dir)
    except Exception as exc:
        (staging / "failure.json").write_text(json.dumps({"error": str(exc), "run_contract": contract,
            "recovery": "Original output path is not finalized. Rerun it; this incomplete attempt is retained."}, indent=2) + "\n")
        raise
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
