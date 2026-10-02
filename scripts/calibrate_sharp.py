"""Chronological calibration of frozen SHARP predictions; no backbone refitting."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_probabilities(p, y=None):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Need a nonempty vector of finite probabilities in [0,1]")
    if y is not None and (np.shape(y) != p.shape or not np.isin(y, [0, 1]).all()):
        raise ValueError("Unknown outcomes must not enter calibration or metrics")
    return p


def apply_calibrator(p, parameters):
    p = check_probabilities(p)
    method = parameters["method"]
    if method == "raw":
        return p.copy()
    if method == "platt":
        epsilon = parameters["epsilon"]
        return expit(parameters["intercept"] + parameters["slope"] * logit(np.clip(p, epsilon, 1-epsilon)))
    if method == "isotonic":
        return np.interp(p, parameters["x"], parameters["y"])
    raise ValueError("Unknown calibration method")


def fit_calibrator(p, y, method, epsilon=1e-6):
    p = check_probabilities(p, y)
    y = np.asarray(y, dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both outcome classes are required for fitting")
    if method == "raw":
        return {"method": "raw"}
    if method == "platt":
        x = logit(np.clip(p, epsilon, 1-epsilon)).reshape(-1, 1)
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, tol=1e-10)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x, y)
        parameters = {"method": "platt", "intercept": float(model.intercept_[0]),
                      "slope": float(model.coef_[0, 0]), "epsilon": epsilon}
        expected = model.predict_proba(x)[:, 1]
    elif method == "isotonic":
        model = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(p, y)
        parameters = {"method": "isotonic", "x": model.X_thresholds_.tolist(), "y": model.y_thresholds_.tolist()}
        expected = model.predict(p)
    else:
        raise ValueError("Unknown calibration method")
    restored = json.loads(json.dumps(parameters))
    np.testing.assert_allclose(apply_calibrator(p, restored), expected, rtol=1e-12, atol=1e-12)
    return parameters


def reliability_table(y, p, bins=10):
    p = check_probabilities(p, y)
    y = np.asarray(y, dtype=int)
    edges = np.linspace(0, 1, bins+1)
    membership = np.minimum(np.searchsorted(edges, p, side="right")-1, bins-1)
    rows = []
    for i in range(bins):
        selected = membership == i
        n = int(selected.sum())
        rows.append({"bin": i, "left": edges[i], "right": edges[i+1], "count": n,
                     "positive": int(y[selected].sum()),
                     "mean_probability": float(p[selected].mean()) if n else None,
                     "observed_fraction": float(y[selected].mean()) if n else None})
    return pd.DataFrame(rows)


def score_probabilities(y, p, climatology, config, diagnostics=True):
    p = check_probabilities(p, y)
    y = np.asarray(y, dtype=int)
    epsilon = config["log_loss_clip_epsilon"]
    clipped = np.clip(p, epsilon, 1-epsilon)
    bins = reliability_table(y, p, config["reliability_bins"])
    brier = float(np.mean((p-y)**2))
    reference = float(np.mean((climatology-y)**2))
    result = {"cases": len(y), "positive": int(y.sum()), "prevalence": float(y.mean()),
              "brier": brier, "brier_skill_train_climatology": 1-brier/reference if reference else None,
              "log_loss": float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
              "ece_equal_width_10": float((bins["count"] * (bins.mean_probability-bins.observed_fraction).abs()).sum()/len(y))}
    if diagnostics:
        try:
            diagnostic = fit_calibrator(p, y, "platt", config["logit_clip_epsilon"])
            result.update(calibration_intercept=diagnostic["intercept"], calibration_slope=diagnostic["slope"],
                          calibration_diagnostic_status="joint_unpenalized_logistic_fit_descriptive_only")
        except (ValueError, ConvergenceWarning):
            result.update(calibration_intercept=None, calibration_slope=None, calibration_diagnostic_status="undefined_or_nonconverged")
    return result


def calibration_masks(frame, config):
    issue = pd.to_datetime(frame.issue_utc, utc=True, format="ISO8601")
    mature = pd.to_datetime(frame.outcome_end_utc_72h, utc=True, format="ISO8601") + pd.Timedelta(hours=config["reporting_delay_hours"])
    start, cut, stop = (pd.Timestamp(config[key], tz="UTC") for key in ("calibration_start", "inner_selection_start", "calibration_end"))
    eligible = frame.role.eq("probability_calibration") & frame.label.isin([0, 1]) & frame.label_known
    if (eligible & ~((issue >= start) & (issue < stop) & (mature <= stop))).any():
        raise ValueError("Calibration role violates its frozen information boundary")
    fit = eligible & (issue < cut) & (mature <= cut)
    selection = eligible & (issue >= cut)
    if not start < cut < stop or (fit & selection).any():
        raise ValueError("Invalid internal calibration boundaries")
    for selected in (fit, selection):
        if set(frame.loc[selected, "label"].unique()) != {0, 1}:
            raise ValueError("Both classes required in inner fitting and method-selection blocks")
    return eligible.to_numpy(), fit.to_numpy(), selection.to_numpy()


def select_and_refit(frame, config, climatology):
    eligible, fit, selection = calibration_masks(frame, config)
    choices, models, scores = {}, {}, []
    for model_name, column in (("gru", "probability_gru_mean"), ("logistic", "probability_logistic")):
        candidates = []
        for rank, method in enumerate(config["candidate_methods"]):
            parameters = fit_calibrator(frame.loc[fit, column].to_numpy(), frame.loc[fit, "label"].to_numpy(), method, config["logit_clip_epsilon"])
            p = apply_calibrator(frame.loc[selection, column].to_numpy(), parameters)
            score = score_probabilities(frame.loc[selection, "label"].to_numpy(), p, climatology, config, diagnostics=False)
            record = {"model": model_name, "method": method, **score}
            scores.append(record)
            candidates.append((score["brier"], score["log_loss"], rank, method))
        choices[model_name] = min(candidates)[-1]
        models[model_name] = {method: fit_calibrator(frame.loc[eligible, column].to_numpy(), frame.loc[eligible, "label"].to_numpy(), method, config["logit_clip_epsilon"])
                              for method in config["candidate_methods"]}
    support = {name: {"cases": int(mask.sum()), "positive": int(frame.loc[mask, "label"].sum())}
               for name, mask in (("inner_fit", fit), ("inner_selection", selection), ("full_final_fit", eligible),
                                  ("inner_boundary_purged", eligible & ~fit & ~selection))}
    return {"selected_methods": choices, "calibrators": models, "inner_scores": scores, "support": support}


def paired_brier_bootstrap(y, candidate, raw, groups, repetitions=1000, seed=2718):
    check_probabilities(candidate, y)
    check_probabilities(raw, y)
    if pd.isna(groups).any():
        raise ValueError("Missing bootstrap group")
    delta = (np.asarray(candidate)-y)**2 - (np.asarray(raw)-y)**2
    grouped = pd.DataFrame({"group": groups, "delta": delta}).groupby("group", sort=True).delta.agg(["sum", "count"])
    count = len(grouped)
    if count < 2:
        return {"difference": float(delta.mean()), "low": None, "high": None, "groups": count}
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(count, np.full(count, 1/count), size=repetitions)
    draws = (weights @ grouped["sum"].to_numpy()) / (weights @ grouped["count"].to_numpy())
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"difference": float(delta.mean()), "low": float(low), "high": float(high), "groups": count}


def load_sources(training_dir, dataset, config):
    if sha256(training_dir / "summary.json") != config["training_summary_sha256"] or sha256(training_dir / "predictions.csv.gz") != config["prediction_sha256"]:
        raise ValueError("Frozen training result changed")
    summary = json.loads((training_dir / "summary.json").read_text())
    if summary["horizon"] != 72 or summary["scope"] != "primary" or config["horizon"] != 72 or config["scope"] != "primary":
        raise ValueError("This experiment requires the pinned 72h primary target")
    for name, expected in summary["output_sha256"].items():
        if sha256(training_dir / name) != expected:
            raise ValueError(f"Frozen training artifact changed: {name}")
    if sha256(dataset / "manifest.json") != config["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    manifest = json.loads((dataset / "manifest.json").read_text())
    if sha256(dataset / "master_cases.csv.gz") != manifest["outputs"]["master_cases.csv.gz"]["sha256"]:
        raise ValueError("Master-case table changed")
    master = pd.read_csv(dataset / "master_cases.csv.gz", low_memory=False)
    predictions = pd.read_csv(training_dir / "predictions.csv.gz")
    frame = predictions.merge(master[["forecast_case_id", "outcome_end_utc_72h", "candidate_primary_label_72h"]],
                              on="forecast_case_id", how="left", validate="one_to_one")
    if frame.forecast_case_id.duplicated().any() or frame.outcome_end_utc_72h.isna().any():
        raise ValueError("Case identities are incomplete")
    if not np.array_equal(frame.label, frame.candidate_primary_label_72h.fillna(-1)) or not np.array_equal(frame.label_known, frame.label.ne(-1)):
        raise ValueError("Frozen predictions and source labels disagree")
    return frame, master, summary


def run_calibration(training_dir, dataset, config, output, source_code):
    if output.exists():
        raise ValueError("Choose a new result directory; frozen outputs are never overwritten")
    frame, master, parent = load_sources(training_dir, dataset, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".incomplete-", dir=output.parent))
    try:
        selection = select_and_refit(frame, config, parent["train_climatology"])
        # Freeze the chosen method and final earlier-data fits before any evaluation.
        save_json(staging / "selection.json", selection)
        restored = json.loads((staging / "selection.json").read_text())
        evaluation, bins, intervals = [], [], []
        for model_name, column in (("gru", "probability_gru_mean"), ("logistic", "probability_logistic")):
            available = frame[column].notna()
            for method in config["candidate_methods"]:
                out_col = f"{model_name}_{method}"
                frame[out_col] = np.nan
                frame.loc[available, out_col] = apply_calibrator(frame.loc[available, column].to_numpy(), restored["calibrators"][model_name][method])
                for role in config["evaluation_roles"]:
                    selected = frame.role.eq(role) & frame.label_known & available
                    y = frame.loc[selected, "label"].to_numpy()
                    p = frame.loc[selected, out_col].to_numpy()
                    evaluation.append({"role": role, "model": model_name, "method": method,
                                       "selected_on_earlier_data": method == restored["selected_methods"][model_name],
                                       **score_probabilities(y, p, parent["train_climatology"], config)})
                    binned = reliability_table(y, p, config["reliability_bins"])
                    binned["role"], binned["model"], binned["method"] = role, model_name, method
                    bins.append(binned)
                    if method != "raw":
                        times = pd.to_datetime(frame.loc[selected, "issue_utc"], utc=True, format="ISO8601")
                        for scheme in config["bootstrap_groups"]:
                            groups = frame.loc[selected, "region_component_id"].to_numpy() if scheme == "region_component" else (times.astype("int64") // (7*86400*10**9)).to_numpy()
                            intervals.append({"role": role, "model": model_name, "method": method, "grouping": scheme,
                                **paired_brier_bootstrap(y, p, frame.loc[selected, column].to_numpy(), groups,
                                                        config["bootstrap_repetitions"], config["bootstrap_seed"])})
            frame[f"{model_name}_selected"] = frame[f"{model_name}_{restored['selected_methods'][model_name]}"]
        probabilities = [f"{model}_{method}" for model in ("gru", "logistic") for method in [*config["candidate_methods"], "selected"]]
        frame[["forecast_case_id", "issue_utc", "region_component_id", "role", "label", "label_known", *probabilities]].to_csv(
            staging / "calibrated_predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        population = master[["forecast_case_id", "issue_utc", "inputs_available", "candidate_primary_label_72h", "label_known_primary_72h"]].merge(
            frame[["forecast_case_id", *probabilities]], on="forecast_case_id", how="left", validate="one_to_one")
        population["forecast_status"] = np.where(population.gru_selected.notna(), "scored_exploratory", "no_supported_sharp_input")
        population.to_csv(staging / "population_predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        pd.DataFrame(evaluation).to_csv(staging / "metrics.csv", index=False)
        pd.concat(bins, ignore_index=True).to_csv(staging / "reliability.csv", index=False)
        pd.DataFrame(intervals).to_csv(staging / "paired_brier_intervals.csv", index=False)
        save_json(staging / "run_contract.json", {"config": config, "source_code_sha256": hashlib.sha256(source_code.encode()).hexdigest(),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}})
        (staging / "executed_calibration_code.py").write_text(source_code)
        summary = {"status": "completed_exploratory_calibration", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "selected_methods": restored["selected_methods"], "support": restored["support"], "metrics": evaluation,
            "population_rows": len(population), "unscored_rows": int(population.gru_selected.isna().sum()),
            "backbone_refitted": False, "conformal_or_policy_fitted": False,
            "limitations": ["Candidate source labels and unknown-outcome exclusions remain provisional.",
                "Selection is based on one earlier calendar block; it is not guaranteed to transfer across cycles.",
                "Lower Brier loss is not, on its own, proof of better calibration.",
                "Calibration intercept/slope fitted on evaluation labels are descriptive diagnostics only; they never modify forecasts.",
                "Bootstrap intervals are paired group-resampling sensitivities conditional on these frozen models and labels, not independent validation or simultaneous intervals.",
                "Previously inspected Cycle-25/2026 data remain retrospective/supplementary, not prospective confirmation.",
                "Conformal and policy blocks remain reserved; no operational trust-state policy has been validated."],
            "output_sha256": {p.name: sha256(p) for p in staging.iterdir()}}
        save_json(staging / "summary.json", summary)
        if output.exists():
            raise ValueError("Output appeared during execution")
        staging.rename(output)
        return summary
    except Exception as exc:
        save_json(staging / "failure.json", {"error": str(exc), "incomplete_run": True})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_calibration(args.training_dir, args.dataset, json.loads(args.config.read_text()), args.output_dir, Path(__file__).read_text())
    print(json.dumps({"status": result["status"], "selected_methods": result["selected_methods"]}, indent=2))
