"""Export the frozen 72-hour experiment for AIA/GOES integration; no training."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.dataset_io import file_sha256, preflight_dataset
from scripts.train_sharp_temporal import make_roles


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def id_hash(values):
    """Order-independent identity of an exact set of unique cases."""
    values = list(values)
    if len(values) != len(set(values)) or not all(isinstance(x, str) and x for x in values):
        raise ValueError("Case IDs must be unique nonempty strings")
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def build_cases(data, temporal, parent_split):
    master = data["master"].copy()
    target = f"candidate_{temporal['scope']}_label_{temporal['horizon']}h"
    y = master[target].fillna(-1).to_numpy(dtype=int)
    known = y != -1
    rows = master.tensor_row.to_numpy(dtype=int)
    indexed = rows >= 0
    supported = np.zeros(len(master), dtype=bool)
    supported[indexed] = np.isfinite(data["sharp"][rows[indexed]]).all(axis=(1, 2))
    roles = make_roles(master, master, known, supported, temporal)
    temporal_roles = make_roles(master, master, np.ones(len(master), dtype=bool),
                                np.ones(len(master), dtype=bool), temporal)
    fields = ["forecast_case_id", "source_sample_id", "region_component_id", "HARPNUM",
              "NOAA_AR_clean", "source_cohort", "issue_tai", "issue_utc", "outcome_end_utc_72h", "tensor_row",
              "target_version", "continuous_outcome_coverage_verified", "historical_availability_status"]
    cases = master[fields].copy()
    cases["label"], cases["label_known"] = y, known
    cases["role"], cases["temporal_role"] = roles, temporal_roles
    cases["sharp_input_finite"] = supported
    references = np.ones(len(master), dtype=bool)
    past = np.ones(len(master), dtype=bool)
    history = []
    issue = pd.to_datetime(master.issue_utc, utc=True, format="ISO8601")
    for lag in data["manifest"]["history_lag_native_minutes"]:
        uri, time = f"history_uri_tminus{lag}", f"history_{lag}_UTC"
        cases[uri], cases[time] = master[uri], master[time]
        references &= master[uri].fillna("").str.startswith("gs://").to_numpy()
        stamp = pd.to_datetime(master[time], utc=True, format="ISO8601")
        past &= (stamp.notna() & stamp.lt(issue)).to_numpy()
        history.append(stamp)
    for earlier, later in zip(history[:-1], history[1:]):
        past &= earlier.lt(later).to_numpy()
    cases["aia_references_present"] = references
    cases["aia_history_past_and_ordered"] = past
    # Reference metadata does not establish remote bytes, quality or delivery.
    cases["aia_full_pixel_verification"] = "not_established"
    cases["goes_predictor_verification"] = "not_imported"
    cases["maximum_input_observation_utc"] = cases.issue_utc
    allowed = [b["role"] for b in temporal["blocks"]]
    cases["candidate_sharp_aia_comparison"] = known & supported & references & past & cases.role.isin(allowed)
    # Keep the SHARP case order, labels and roles exactly as executed in Notebook 01.
    matched = cases.loc[indexed].sort_values("tensor_row")
    if not np.array_equal(matched.forecast_case_id, parent_split.forecast_case_id):
        raise ValueError("Parent SHARP case identity/order differs")
    for field in ["role", "label", "label_known"]:
        if not np.array_equal(matched[field], parent_split[field]):
            raise ValueError(f"Parent SHARP {field} differs")
    return cases


def build_aia_inventory(cases, lags):
    selected = cases[cases.candidate_sharp_aia_comparison]
    parts = []
    for lag in lags:
        part = selected[["forecast_case_id", "role", f"history_uri_tminus{lag}", f"history_{lag}_UTC"]].copy()
        part.columns = ["forecast_case_id", "role", "uri", "observation_utc"]
        parts.append(part)
    uses = pd.concat(parts, ignore_index=True)
    if uses.groupby("uri").observation_utc.nunique().gt(1).any():
        raise ValueError("The same AIA URI has conflicting observation timestamps")
    inventory = uses.groupby("uri", sort=True).agg(
        observation_utc=("observation_utc", "first"),
        case_uses=("forecast_case_id", "size"),
        roles=("role", lambda x: "|".join(sorted(set(x)))),
    ).reset_index()
    inventory["object_generation"] = pd.NA
    inventory["sha256"] = pd.NA
    inventory["pixel_verification"] = "pending_exact_object_receipt"
    return inventory


def validate_prediction_import(frame, metadata, contract, cases):
    """Validate a candidate raw branch export; this does not prove its training history.

    Require every candidate case, with explicit failed-input rows. Check actual model
    provenance/replay separately before scientific acceptance or model comparison.
    """
    for key in ["horizon_hours", "target_scope", "target_version", "dataset_manifest_sha256",
                "split_manifest_sha256", "training_case_ids_sha256"]:
        if metadata.get(key) != contract[key]:
            raise ValueError(f"Prediction contract mismatch: {key}")
    if metadata.get("probability_kind") != "raw":
        raise ValueError("Import raw probabilities before separately reserved calibration")
    if metadata.get("fit_roles") != ["train"] or metadata.get("selection_roles") != ["model_validation"]:
        raise ValueError("Fitting/selection roles differ from the matched experiment")
    if metadata.get("branch") not in contract["branches"]:
        raise ValueError("Unknown branch")
    combiner_roles = metadata.get("combiner_fit_roles", [])
    if combiner_roles not in [[], ["model_validation"]] or (combiner_roles and "_" not in metadata["branch"]):
        raise ValueError("Combiner fitting must use the earlier model-validation block only")
    required = {"forecast_case_id", "probability", "input_status", "last_observation_utc"}
    if not required.issubset(frame):
        raise ValueError("Prediction columns missing")
    if frame.forecast_case_id.isna().any() or frame.forecast_case_id.duplicated().any():
        raise ValueError("Missing or duplicate prediction identity")
    expected = cases[cases.candidate_sharp_aia_comparison].set_index("forecast_case_id")
    if set(frame.forecast_case_id) != set(expected.index):
        raise ValueError("Prediction export must retain every requested case, including input failures")
    aligned = frame.set_index("forecast_case_id").loc[expected.index].copy()
    if not aligned.input_status.isin(["ok", "missing_input", "invalid_input"]).all():
        raise ValueError("Undeclared input status")
    ok = aligned.input_status.eq("ok")
    p = pd.to_numeric(aligned.probability, errors="raise")
    if not np.isfinite(p[ok]).all() or not p[ok].between(0, 1).all() or p[~ok].notna().any():
        raise ValueError("Probability/status mismatch")
    observed = pd.to_datetime(aligned.last_observation_utc, utc=True, format="ISO8601")
    issue = pd.to_datetime(expected.issue_utc, utc=True, format="ISO8601")
    if observed[ok].isna().any() or observed[ok].gt(issue[ok]).any():
        raise ValueError("Missing or future input observation")
    if metadata.get("historical_availability") not in ["unverified_retrospective", "verified_as_of"]:
        raise ValueError("Declare historical availability explicitly")
    if metadata["historical_availability"] == "verified_as_of":
        if "last_available_utc" not in aligned or not metadata.get("availability_evidence_sha256"):
            raise ValueError("As-of claim requires availability timestamps and evidence")
        available = pd.to_datetime(aligned.last_available_utc, utc=True, format="ISO8601")
        if available[ok].isna().any() or available[ok].gt(issue[ok]).any() or available[ok].lt(observed[ok]).any():
            raise ValueError("Invalid input availability timestamp")
    return {"status": "structural_checks_passed_not_scientific_acceptance", "cases": len(aligned),
            "issued": int(ok.sum()), "input_failures": int((~ok).sum()),
            "historical_availability": metadata["historical_availability"]}


def prepare(dataset, parent, temporal_path, output):
    dataset, parent, temporal_path, output = map(Path, [dataset, parent, temporal_path, output])
    if output.exists():
        raise ValueError("Choose a new output directory; existing results are preserved")
    temporal = json.loads(temporal_path.read_text())
    if temporal["horizon"] != 72 or temporal["scope"] != "primary":
        raise ValueError("This experiment requires the primary-region 72-hour target")
    if file_sha256(dataset / "manifest.json") != temporal["dataset_manifest_sha256"]:
        raise ValueError("Frozen data version differs")
    summary = json.loads((parent / "summary.json").read_text())
    parent_contract = json.loads((parent / "run_contract.json").read_text())
    if parent_contract["config"] != temporal:
        raise ValueError("Parent experiment configuration differs")
    for name in ["split_manifest.csv.gz", "predictions.csv.gz"]:
        if file_sha256(parent / name) != summary["output_sha256"][name]:
            raise ValueError(f"Parent artifact changed: {name}")
    data = preflight_dataset(dataset, 72, "primary")
    parent_split = pd.read_csv(parent / "split_manifest.csv.gz")
    cases = build_cases(data, temporal, parent_split)
    if cases.target_version.nunique(dropna=False) != 1 or cases.target_version.isna().any():
        raise ValueError("One explicit target version is required")
    aia = build_aia_inventory(cases, data["manifest"]["history_lag_native_minutes"])
    selected = cases[cases.candidate_sharp_aia_comparison]
    roles = []
    for block in temporal["blocks"]:
        all_role = cases[cases.temporal_role.eq(block["role"])]
        matched = selected[selected.role.eq(block["role"])]
        roles.append({"role": block["role"], "population": len(all_role),
                      "candidate_sharp_aia": len(matched), "positive_candidates": int(matched.label.eq(1).sum()),
                      "region_components": int(matched.region_component_id.nunique()),
                      "unknown_labels": int((~all_role.label_known).sum()),
                      "missing_sharp_histories": int((~all_role.sharp_input_finite).sum())})
    support = pd.DataFrame(roles)
    annual = cases.assign(year=pd.to_datetime(cases.issue_utc, utc=True, format="ISO8601").dt.year).groupby("year").agg(
        population=("forecast_case_id", "size"), candidate_sharp_aia=("candidate_sharp_aia_comparison", "sum"),
        sharp_histories=("sharp_input_finite", "sum"), known_labels=("label_known", "sum")).reset_index()
    output.mkdir(parents=True)
    compression = {"method": "gzip", "mtime": 0}
    cases.to_csv(output / "multimodal_case_manifest.csv.gz", index=False, compression=compression)
    selected.to_csv(output / "candidate_comparison_cases.csv.gz", index=False, compression=compression)
    aia.to_csv(output / "aia_object_requests.csv.gz", index=False, compression=compression)
    # A request table is deliberately not a matrix of invented GOES measurements.
    selected[["forecast_case_id", "source_sample_id", "role", "issue_utc", "maximum_input_observation_utc"]].to_csv(
        output / "goes_case_requests.csv.gz", index=False, compression=compression)
    support.to_csv(output / "role_support.csv", index=False)
    annual.to_csv(output / "annual_support.csv", index=False)
    contract = {
        "experiment": "multimodal72_matched_candidate_v1", "status": "prepared_not_trained",
        "horizon_hours": 72, "target_scope": "primary", "target_version": str(cases.target_version.iloc[0]),
        "dataset_manifest_sha256": temporal["dataset_manifest_sha256"],
        "split_manifest_sha256": file_sha256(output / "candidate_comparison_cases.csv.gz"),
        "training_case_ids_sha256": id_hash(selected.loc[selected.role.eq("train"), "forecast_case_id"]),
        "parent_training_summary_sha256": file_sha256(parent / "summary.json"),
        "parent_split_sha256": file_sha256(parent / "split_manifest.csv.gz"),
        "parent_predictions_sha256": file_sha256(parent / "predictions.csv.gz"),
        "branches": ["sharp", "aia", "goes", "sharp_goes", "sharp_aia", "sharp_aia_goes"],
        "blocks": temporal["blocks"], "reporting_delay_hours": temporal["reporting_delay_hours"],
        "time_scale": "Preserve native issue_tai and the pinned UTC endpoints. The 72h interval is added in native TAI; naive UTC subtraction can differ by one second across an inserted leap second.",
        "training_case_policy": "Exact matched training IDs required; input failures require a versioned protocol amendment and matched refits, not silent dropping.",
        "probability_kind": "raw", "fit_roles": ["train"], "selection_roles": ["model_validation"],
        "late_fusion": "Fixed equal probability average reference; any learned combiner uses model_validation only. Fit calibration afterward on probability_calibration.",
        "goes_source_role": "Past continuous XRS predictors are separate from future-event outcome labels.",
        "goes_observation_cutoff": "Observation and any aggregation interval end must be at or before issue time; operational claims additionally require availability evidence.",
        "goes_input_specification": "Pending upstream source-quality/availability acceptance and explicit history, detector, satellite and preprocessing specification.",
        "matched_evaluation": "Compare every pair on identical known-outcome cases with accepted inputs; also report whole-population availability. Repeat SHARP+AIA on triple-input support to isolate GOES addition.",
        "scientific_acceptance": "Structural import checks do not verify actual training provenance, checkpoint replay, all source pixels, GOES quality or historical delivery.",
        "validation_status": "Retrospective exploratory labels. AR/event-disjoint sensitivities and genuinely prospective evaluation remain separate work.",
        "reuse_scope": "Reuse upstream raw images, source inventories and compatible code; 48-hour scores and differently fitted checkpoints are not this 72-hour experiment.",
    }
    write_json(output / "experiment_contract.json", contract)
    completed = {"status": "completed_multimodal_preparation_not_training", "completed_utc": datetime.now(timezone.utc).isoformat(),
                 "population_cases": len(cases), "candidate_comparison_cases": len(selected),
                 "unique_requested_aia_objects": len(aia), "aia_history_references": int(aia.case_uses.sum()),
                 "parent_sharp_rows_reconciled": len(parent_split), "model_fits": 0, "downloads": 0,
                 "goes_predictor_matrix_created": False, "aia_full_pixel_verification": False,
                 "sources": {"temporal_config_sha256": file_sha256(temporal_path),
                             "prepare_code_sha256": file_sha256(Path(__file__))},
                 "output_sha256": {p.name: file_sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / "summary.json", completed)
    return completed, cases, support, annual, aia, contract
