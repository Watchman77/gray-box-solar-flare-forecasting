"""Build calibration figures and a scope-aware 3/24/72-hour comparison table."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/graybox-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/multi_horizon_comparison_v1"


def fit_selected_calibrator(frame: pd.DataFrame, method: str):
    calibration = frame[frame.role == "probability_calibration"].sort_values("issue_utc")
    n_fit = max(1, int(np.floor(0.70 * len(calibration))))
    fit = calibration.iloc[:n_fit]
    p = fit.probability.to_numpy(float)
    y = fit.label.to_numpy(int)
    if method == "raw":
        return lambda values: np.asarray(values, dtype=float)
    if method == "platt":
        eps = 1e-6
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, tol=1e-10)
        x = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
        model.fit(x, y)
        return lambda values: model.predict_proba(
            np.log(np.clip(np.asarray(values, dtype=float), eps, 1 - eps)
                   / np.clip(1 - np.asarray(values, dtype=float), eps, 1 - eps)).reshape(-1, 1)
        )[:, 1]
    if method == "isotonic":
        model = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(p, y)
        return lambda values: model.predict(np.asarray(values, dtype=float))
    raise ValueError(method)


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    membership = np.minimum(np.searchsorted(edges, p, side="right") - 1, bins - 1)
    rows = []
    for index in range(bins):
        selected = membership == index
        rows.append(
            {
                "bin": index,
                "left": edges[index],
                "right": edges[index + 1],
                "count": int(selected.sum()),
                "mean_probability": float(p[selected].mean()) if selected.any() else np.nan,
                "observed_fraction": float(y[selected].mean()) if selected.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_short_horizon_evidence(horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = ROOT / f"results/{horizon}h_graybox"
    frame = pd.read_csv(base / "validation_audit_v1/audit_cases.csv.gz", low_memory=False)
    frame["issue_utc"] = pd.to_datetime(frame["issue_utc"], utc=True)
    frame["probability"] = pd.to_numeric(frame["probability_gru_mean"], errors="coerce")
    selected_method = json.loads((base / "sharp_calibration_v1/summary.json").read_text())["selected_method"]
    calibrator = fit_selected_calibrator(frame, selected_method)
    frame["selected_probability"] = np.nan
    available = frame.probability.notna() & frame.label.isin([0, 1])
    frame.loc[available, "selected_probability"] = calibrator(frame.loc[available, "probability"])

    reliability = []
    for role in ("policy_validation", "retrospective_cycle25", "supplementary_2026"):
        role_frame = frame[(frame.role == role) & frame.label.isin([0, 1]) & frame.probability.notna()]
        for method, column in (("raw", "probability"), (selected_method, "selected_probability")):
            bins = reliability_table(role_frame.label.to_numpy(int), role_frame[column].to_numpy(float))
            bins.insert(0, "horizon_hours", horizon)
            bins.insert(1, "role", role)
            bins.insert(2, "method", method)
            bins["ece_equal_width_10"] = (
                (bins["count"] * (bins["mean_probability"] - bins["observed_fraction"]).abs()).sum()
                / len(role_frame)
            )
            reliability.append(bins)
    reliability_frame = pd.concat(reliability, ignore_index=True)

    role_metrics = pd.read_csv(base / "validation_audit_v1/role_metrics.csv")
    conformal = pd.read_csv(base / "sharp_conformal_v1/conformal_metrics.csv")
    policy = pd.read_csv(base / "sharp_policy_diagnostic_v1/policy_metrics.csv")
    threshold = json.loads((base / "sharp_threshold_audit_v1/summary.json").read_text())["selected_threshold"]
    rows = []
    for role in ("policy_validation", "retrospective_cycle25", "supplementary_2026"):
        metric = role_metrics[role_metrics.role == role].iloc[0]
        conf = conformal[conformal.role == role].iloc[0]
        pol = policy[(policy.role == role) & (policy.policy_state == "NORMAL")].iloc[0]
        rows.append(
            {
                "horizon_hours": horizon,
                "scope": "SHARP-only",
                "role": role,
                "cases": int(metric.cases),
                "positives": int(metric.positive),
                "threshold": float(threshold),
                "tss": float(metric.tss),
                "average_precision": float(metric.average_precision),
                "brier": float(metric.brier),
                "roc_auc": float(metric.roc_auc),
                "conformal_overall_coverage": float(conf.overall_coverage),
                "conformal_flare_coverage": float(conf.flare_coverage),
                "singleton_rate": float(conf.singleton_rate),
                "normal_share": float(pol.share_of_known_cases),
                "normal_flare_coverage": float(pol.flare_coverage),
                "note": "Provisional labels; SHARP-only short-horizon result.",
            }
        )
    return reliability_frame, pd.DataFrame(rows)


def build_72h_rows() -> pd.DataFrame:
    policy = pd.read_csv(ROOT / "results/72h_graybox/20261005/policy_validation_v1_results.csv")
    q90 = policy[policy.policy_id == "policy_q90"].iloc[0]
    replay = pd.read_csv(ROOT / "results/72h_graybox/20261006/rolling_operational_replay_v1/final_replay_summary.csv")
    conf = pd.read_csv(ROOT / "results/72h_graybox/20261005/results_synthesis_v2/conformal_shift_summary.csv")
    rows = []
    rows.append(
        {
            "horizon_hours": 72,
            "scope": "SHARP+AIA+fusion fallback-v3",
            "role": "policy_validation",
            "cases": int(q90.requested_cases),
            "positives": int(q90.positives),
            "threshold": np.nan,
            "tss": float(q90.issued_tss),
            "average_precision": np.nan,
            "brier": np.nan,
            "roc_auc": np.nan,
            "conformal_overall_coverage": np.nan,
            "conformal_flare_coverage": np.nan,
            "singleton_rate": np.nan,
            "normal_share": float(q90.normal_rate),
            "normal_flare_coverage": np.nan,
            "note": "Final 72-hour policy-validation summary; multimodal policy metrics are not directly equivalent to SHARP-only rows.",
        }
    )
    for role, replay_role, conf_period in (
        ("retrospective_cycle25", "retrospective_cycle25", "cycle25_mondrian_v2"),
        ("supplementary_2026", "supplementary_2026", "supplementary2026_mondrian_v2"),
    ):
        result = replay[(replay.scenario == "nominal_frozen_inputs") & (replay.role == replay_role)].iloc[0]
        coverage = conf[(conf.branch == "sharp_aia") & (conf.period == conf_period)].iloc[0]
        rows.append(
            {
                "horizon_hours": 72,
                "scope": "SHARP+AIA+fusion fallback-v3",
                "role": role,
                "cases": int(result.cases),
                "positives": int(result.tp + result.fn),
                "threshold": np.nan,
                "tss": float(result.tss),
                "average_precision": float(result.average_precision),
                "brier": float(result.brier),
                "roc_auc": float(result.roc_auc),
                "conformal_overall_coverage": float(coverage.overall_coverage),
                "conformal_flare_coverage": float(coverage.flare_coverage),
                "singleton_rate": np.nan,
                "normal_share": float(result.normal_rate),
                "normal_flare_coverage": np.nan,
                "note": "Frozen 72-hour multimodal fallback-v3 replay; metrics are scope-aware and not a like-for-like SHARP-only comparison.",
            }
        )
    return pd.DataFrame(rows)


def plot_reliability(reliability: pd.DataFrame) -> None:
    roles = ["policy_validation", "retrospective_cycle25", "supplementary_2026"]
    labels = {
        "policy_validation": "Policy validation",
        "retrospective_cycle25": "Cycle 25",
        "supplementary_2026": "2026",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True, constrained_layout=True)
    for row_index, horizon in enumerate((3, 24)):
        for col_index, role in enumerate(roles):
            ax = axes[row_index, col_index]
            subset = reliability[(reliability.horizon_hours == horizon) & (reliability.role == role)]
            for method, style in (("raw", "--"), ("platt", "-"), ("isotonic", "-")):
                data = subset[subset.method == method].dropna(subset=["mean_probability", "observed_fraction"])
                if data.empty:
                    continue
                ax.plot(data.mean_probability, data.observed_fraction, marker="o", linestyle=style, label=method)
            ax.plot([0, 1], [0, 1], color="0.6", linewidth=1)
            ax.set_title(f"{horizon} h · {labels[role]}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            if row_index == 1:
                ax.set_xlabel("Mean predicted probability")
            if col_index == 0:
                ax.set_ylabel("Observed event fraction")
            if row_index == 0 and col_index == 2:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("SHARP probability reliability by forecast horizon and evaluation role")
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures/calibration_reliability.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reliability_frames = []
    metric_frames = []
    for horizon in (3, 24):
        reliability, metrics = build_short_horizon_evidence(horizon)
        reliability_frames.append(reliability)
        metric_frames.append(metrics)
    reliability = pd.concat(reliability_frames, ignore_index=True)
    comparison = pd.concat([*metric_frames, build_72h_rows()], ignore_index=True)
    reliability.to_csv(OUT / "calibration_reliability.csv", index=False)
    comparison.to_csv(OUT / "multihorizon_comparison.csv", index=False)
    plot_reliability(reliability)
    receipt = {
        "status": "completed_manuscript_evidence_build",
        "horizons": [3, 24, 72],
        "outputs": [
            "calibration_reliability.csv",
            "multihorizon_comparison.csv",
            "figures/calibration_reliability.png",
        ],
        "limitations": [
            "Short-horizon labels remain provisional candidate labels.",
            "Short-horizon rows are SHARP-only; 72-hour rows are multimodal fallback-v3 summaries and are not like-for-like model comparisons.",
            "The figures are descriptive reliability displays; they do not refit the frozen operational policy.",
        ],
    }
    (OUT / "manuscript_evidence_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
