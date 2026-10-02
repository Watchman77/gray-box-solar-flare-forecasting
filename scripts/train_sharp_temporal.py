"""Train a small 72h SHARP GRU on frozen chronological candidate-data roles."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    from scripts.dataset_io import file_sha256, preflight_dataset
    from scripts.run_exploratory_sharp import fit_logistic, sigmoid
except ModuleNotFoundError:
    from dataset_io import file_sha256, preflight_dataset
    from run_exploratory_sharp import fit_logistic, sigmoid


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_roles(index, master, known, supported, config):
    issue = pd.to_datetime(index.issue_utc, utc=True, format="ISO8601")
    end = pd.to_datetime(master[f"outcome_end_utc_{config['horizon']}h"], utc=True, format="ISO8601")
    mature = end + pd.Timedelta(hours=config["reporting_delay_hours"])
    roles = np.full(len(index), "outside_blocks", dtype=object)
    previous_end = None
    for block in config["blocks"]:
        start, stop = (pd.Timestamp(block[key], tz="UTC") for key in ("start", "end"))
        if start >= stop or (previous_end is not None and start < previous_end):
            raise ValueError("Overlapping or unordered time blocks")
        previous_end = stop
        selected = ((issue >= start) & (issue < stop)).to_numpy()
        # Purge incomplete outcome/reporting windows at every role boundary.
        roles[selected] = "purged_outcome_boundary"
        roles[selected & (mature <= stop).to_numpy()] = block["role"]
    roles[~known] = "unknown_label_excluded"
    roles[~supported] = "missing_input_excluded"
    return roles


def fit_transform(raw, train):
    if not train.any() or not np.isfinite(raw[train]).all():
        raise ValueError("Need finite training histories")
    logged = np.sign(raw[train]) * np.log1p(np.abs(raw[train]))
    mean = logged.mean(axis=(0, 1))
    std = logged.std(axis=(0, 1))
    std[std < 1e-12] = 1
    return {"mean": mean, "std": std}


def transform(raw, parameters):
    logged = np.sign(raw) * np.log1p(np.abs(raw))
    return np.asarray((logged - parameters["mean"]) / parameters["std"], dtype=np.float32)


class SharpGRU(nn.Module):
    def __init__(self, features, hidden, dropout):
        super().__init__()
        self.gru = nn.GRU(features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        _, state = self.gru(x)
        return self.head(state[-1]).squeeze(-1)


def predict(model, x, batch_size=4096):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i + batch_size]).sigmoid()
                          for i in range(0, len(x), batch_size)]).numpy()


def metrics(y, p, climatology, threshold=0.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if not len(y):
        return {"cases": 0}
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require known binary targets and finite predictions")
    positive, negative = int(y.sum()), int((1-y).sum())
    brier = float(np.mean((y-p)**2))
    reference = float(np.mean((y-climatology)**2))
    clipped = np.clip(p, 1e-12, 1-1e-12)
    # Aggregate score ties before computing threshold-based AP and ROC area.
    order = np.argsort(-p, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(y)-1]
    tp_curve = np.cumsum(y[order])[ends]
    fp_curve = (ends + 1) - tp_curve
    ap = float(np.sum(np.diff(np.r_[0, tp_curve]) / positive * tp_curve / (ends+1))) if positive else None
    auc = float(np.trapezoid(np.r_[0, tp_curve / positive], np.r_[0, fp_curve / negative])) if positive and negative else None
    alert = p >= threshold
    tp, fp = int((alert & (y == 1)).sum()), int((alert & (y == 0)).sum())
    fn, tn = positive-tp, negative-fp
    return {"cases": len(y), "positive": positive, "prevalence": positive/len(y),
            "brier": brier, "brier_skill_train_climatology": 1-brier/reference if reference else None,
            "log_loss": float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
            "average_precision": ap, "roc_auc": auc, "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "tss": tp/positive-fp/negative if positive and negative else None}


def train_seed(x, y, roles, config, seed, output):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = SharpGRU(x.shape[2], config["hidden_size"], config["head_dropout"])
    train, val = np.flatnonzero(roles == "train"), np.flatnonzero(roles == "model_validation")
    prior = float(y[train].mean())
    with torch.no_grad():
        model.head[-1].bias.fill_(np.log(prior/(1-prior)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    targets = torch.from_numpy(y.astype(np.float32))
    best, best_epoch, stale, history = float("inf"), 0, 0, []
    checkpoint = output / f"seed_{seed}.pt"
    for epoch in range(1, config["max_epochs"]+1):
        started = time.monotonic()
        model.train()
        order = train[torch.randperm(len(train)).numpy()]
        total = 0.0
        for start in range(0, len(order), config["batch_size"]):
            indices = order[start:start+config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[indices]), targets[indices])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            total += loss.item()*len(indices)
        p = predict(model, x[val])
        val_loss = metrics(y[val], p, prior)["log_loss"]
        history.append({"seed": seed, "epoch": epoch, "train_loss": total/len(train),
                        "validation_log_loss": val_loss, "seconds": time.monotonic()-started})
        if val_loss < best:
            best, best_epoch, stale = val_loss, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        pd.DataFrame(history).to_csv(output / f"seed_{seed}_history.csv", index=False)
        print(json.dumps({"seed": seed, "epoch": epoch, "validation_log_loss": round(val_loss, 6),
                          "best_epoch": best_epoch, "seconds": round(history[-1]["seconds"], 2)}), flush=True)
        if stale >= config["patience"]:
            break
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    return model, {"seed": seed, "epochs": epoch, "best_epoch": best_epoch, "best_validation_log_loss": best}


def run(dataset, config_path, output):
    if output.exists():
        raise ValueError("Output exists; choose a new run path. Existing experiments are immutable")
    config = json.loads(config_path.read_text())
    if config["device"] != "cpu":
        raise ValueError("This bounded local runner supports CPU only")
    if file_sha256(dataset / "manifest.json") != config["dataset_manifest_sha256"]:
        raise ValueError("Data version differs from the frozen experiment")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    data = preflight_dataset(dataset, config["horizon"], config["scope"])
    index, raw, y, known = data["index"], data["sharp"], data["labels"], data["known"]
    master = data["master"].set_index("forecast_case_id").loc[index.forecast_case_id].reset_index()
    supported = np.isfinite(raw).all(axis=(1, 2))
    roles = make_roles(index, master, known, supported, config)
    for role in ("train", "model_validation"):
        if set(np.unique(y[roles == role])) != {0, 1}:
            raise ValueError(f"Both classes required in {role}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".incomplete-", dir=output.parent))
    contract = {"config": config, "config_sha256": file_sha256(config_path),
                "code_sha256": {name: file_sha256(Path(__file__).with_name(name))
                                for name in (Path(__file__).name, "dataset_io.py", "run_exploratory_sharp.py")},
                "versions": {"numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__},
                "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(staging / "run_contract.json", contract)
    try:
        plan = index[["forecast_case_id", "issue_utc", "region_component_id", "tensor_row"]].copy()
        plan["role"], plan["label"], plan["label_known"] = roles, y, known
        plan.to_csv(staging / "split_manifest.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        role_support = []
        for role in sorted(set(roles)):
            mask = roles == role
            role_support.append({"role": role, "cases": int(mask.sum()), "positive": int((y[mask] == 1).sum()),
                                 "region_components": int(index.loc[mask, "region_component_id"].nunique())})
        print(json.dumps({"status": "training_started", "output_staging": str(staging), "roles": role_support}), flush=True)
        train = roles == "train"
        parameters = fit_transform(raw, train)
        np.savez(staging / "transform.npz", **parameters)
        # Unsupported rows are never fed into the model or scored.
        normalized = np.zeros(raw.shape, dtype=np.float32)
        normalized[supported] = transform(raw[supported], parameters)
        x = torch.from_numpy(normalized)
        prior = float(y[train].mean())
        seeds = []
        probabilities = []
        for seed in config["seeds"]:
            model, record = train_seed(x, y, roles, config, seed, staging)
            p = np.full(len(y), np.nan)
            p[supported] = predict(model, x[supported])
            replay = SharpGRU(raw.shape[2], config["hidden_size"], config["head_dropout"])
            replay.load_state_dict(torch.load(staging / f"seed_{seed}.pt", weights_only=True, map_location="cpu"))
            with np.load(staging / "transform.npz", allow_pickle=False) as saved:
                x_replay = torch.from_numpy(transform(raw[supported], saved))
            if not np.array_equal(p[supported], predict(replay, x_replay)):
                raise ValueError("Saved checkpoint/transform did not reproduce predictions")
            plan[f"probability_seed_{seed}"] = p
            probabilities.append(p)
            seeds.append(record)
        plan["probability_gru_mean"] = np.mean(probabilities, axis=0)
        # A fair interpretable reference uses exactly the same fitting population.
        flat = normalized.reshape(len(raw), -1).astype(float)
        beta, optimizer = fit_logistic(flat[train], y[train].astype(float))
        if not optimizer["converged"]:
            raise ValueError("Matched logistic reference failed to converge")
        np.savez(staging / "logistic_reference.npz", beta=beta, **parameters)
        plan["probability_logistic"] = sigmoid(beta[0] + flat @ beta[1:])
        plan.loc[~supported, "probability_logistic"] = np.nan
        rows = []
        # Calibration and policy labels remain reserved: no fitting or performance
        # inspection on those blocks during this backbone-training stage.
        for role in ("train", "model_validation", "retrospective_cycle25", "supplementary_2026"):
            selected = roles == role
            for column in ("probability_gru_mean", "probability_logistic"):
                rows.append({"role": role, "model": column, **metrics(y[selected], plan.loc[selected, column].to_numpy(), prior, config["evaluation_threshold"])})
        plan.to_csv(staging / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        population = data["master"][["forecast_case_id", "issue_utc", "inputs_available"]].merge(
            plan.drop(columns=["issue_utc", "tensor_row"]), on="forecast_case_id", how="left", validate="one_to_one")
        population["forecast_status"] = np.where(population.probability_gru_mean.notna(), "scored_exploratory", "no_supported_sharp_input")
        population.to_csv(staging / "population_predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        region_sets = {block["role"]: set(index.loc[roles == block["role"], "region_component_id"]) for block in config["blocks"]}
        overlap = {f"{a}|{b}": len(region_sets[a] & region_sets[b]) for i, a in enumerate(region_sets) for b in list(region_sets)[i+1:]}
        summary = {"status": "completed_exploratory_training", "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "experiment": config["experiment"], "horizon": config["horizon"], "scope": config["scope"],
                   "roles": role_support, "seeds": seeds, "saved_model_replay": "identical_all_supported_cases_each_seed",
                   "metrics": rows, "train_climatology": prior, "region_overlap_by_role": overlap,
                   "population_rows": len(population), "unscored_rows": int(population.probability_gru_mean.isna().sum()),
                   "limitations": ["Candidate labels and provisional zeros; not confirmatory results.",
                       "Chronological roles with outcome/reporting purge, not an AR-disjoint or event-disjoint evaluation.",
                       "Prior Cycle-25 inspection makes evaluation retrospective; 2026 is supplementary.",
                       "Three short histories are not the published seven-daily-state ASR input contract.",
                       "Probabilities are uncalibrated; calibration/conformal/policy blocks are reserved for later stages.",
                       "Ensemble spread is seed disagreement, not a calibrated confidence interval.",
                       "Matched cohort selection and unassembled histories limit SHARP-only population claims."],
                   "output_sha256": {p.name: file_sha256(p) for p in staging.iterdir()}}
        write_json(staging / "summary.json", summary)
        if output.exists():
            raise ValueError("Output appeared during training; refusing replacement")
        staging.rename(output)
        print(json.dumps({"status": summary["status"], "output": str(output), "seeds": seeds}), flush=True)
    except Exception as exc:
        write_json(staging / "failure.json", {"error": str(exc), "incomplete_run": True})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.config, args.output_dir)
