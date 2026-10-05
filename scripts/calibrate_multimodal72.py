"""Earlier-data calibration of frozen SHARP, AIA and equal-probability fusion.

This CPU-only stage can consume a completely archived calibration role while
inference continues on other roles. It never reads later prediction archives.
"""

from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

from scripts.aia72_inference_archive_relay import verify_archive
from scripts.calibrate_sharp import (
    apply_calibrator, calibration_masks, check_probabilities, fit_calibrator,
    score_probabilities, sha256, save_json,
)

BRANCHES = ("sharp", "aia", "sharp_aia")


def require(value, message):
    if not value:
        raise ValueError(message)


def load_complete_role(repo, config):
    """Require every expected calibration block, including failed-input rows."""
    repo = Path(repo)
    package = repo / config["inference_package"]
    collection = package / "collected_blocks"
    contract_path = package / "bundle/configs/aia72_inference_v1.json"
    require(sha256(contract_path) == config["inference_contract_sha256"], "Inference contract changed")
    contract = json.loads(contract_path.read_text())
    require(contract["fitting_allowed"] is False and contract["seeds"] == [17, 29, 43], "Wrong inference experiment")
    sources = {"inference_contract_sha256": sha256(contract_path), "inputs": {}, "archives": {}}
    for name in ["cases.csv.gz", "objects.json", "blocks.json"]:
        digest = sha256(package / "inputs" / name)
        require(digest == contract["inputs"][name]["sha256"], "Inference input changed: " + name)
        sources["inputs"][name] = digest
    require(contract["inputs"]["cases.csv.gz"]["sha256"] == config["case_manifest_sha256"], "Case cohort changed")
    cases = pd.read_csv(package / "inputs/cases.csv.gz", float_precision="round_trip")
    require(not cases.forecast_case_id.duplicated().any(), "Duplicate manifest identities")
    role = "probability_calibration"
    expected = cases[cases.role.eq(role)].copy()
    blocks = [b for b in json.loads((package / "inputs/blocks.json").read_text()) if b["role"] == role]
    require(blocks and [v for b in blocks for v in b["case_ids"]] == expected.forecast_case_id.tolist(), "Block support/order differs")
    missing = [b["block_id"] for b in blocks if not (collection / (b["block_id"] + "_ack.json")).is_file()]
    require(not missing, "Calibration role incomplete: " + ", ".join(missing))
    pins = json.loads((package / "inputs/objects.json").read_text())
    # The archive verifier compares original CSV string values, including UTC text.
    import csv
    with gzip.open(package / "inputs/cases.csv.gz", "rt") as stream:
        case_rows = {r["forecast_case_id"]: r for r in csv.DictReader(stream)}
    rows = []
    for block in blocks:
        name = block["block_id"]
        ack_raw = (collection / (name + "_ack.json")).read_bytes()
        ack = json.loads(ack_raw)
        require(ack["status"] == "local_block_archive_verified", "Archive has no verified acknowledgement")
        raw = (collection / (name + ".tar.gz")).read_bytes()
        verified = verify_archive(raw, ack, block, pins, case_rows)
        require(all(ack[k] == v for k, v in verified.items()), "Acknowledgement does not match archive")
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            payload = archive.extractfile(name + "_predictions.csv.gz").read()
        rows.append(pd.read_csv(io.BytesIO(payload), compression="gzip", float_precision="round_trip"))
        sources["archives"][name] = {"archive_sha256": verified["archive_sha256"],
                                      "ack_sha256": hashlib.sha256(ack_raw).hexdigest(),
                                      "cases": verified["cases"]}
    aia = pd.concat(rows, ignore_index=True)
    require(aia.forecast_case_id.tolist() == expected.forecast_case_id.tolist(), "Incomplete or reordered predictions")
    require(np.array_equal(aia.label, expected.label), "AIA labels changed")
    parent = repo / config["sharp_training_dir"]
    for name, key in [("summary.json", "sharp_summary_sha256"), ("predictions.csv.gz", "sharp_predictions_sha256")]:
        digest = sha256(parent / name)
        require(digest == config[key], "Frozen SHARP input changed: " + name)
        sources[name] = digest
    summary = json.loads((parent / "summary.json").read_text())
    require(summary["horizon"] == 72 and summary["scope"] == "primary", "Wrong SHARP target")
    sharp = pd.read_csv(parent / "predictions.csv.gz", float_precision="round_trip")
    require(not sharp.forecast_case_id.duplicated().any(), "Duplicate SHARP identities")
    sharp = sharp.set_index("forecast_case_id").reindex(expected.forecast_case_id)
    for name in ["role", "label", "issue_utc", "region_component_id"]:
        require(np.array_equal(sharp[name].to_numpy(), expected[name].to_numpy()), "SHARP case metadata changed: " + name)
    frame = expected.reset_index(drop=True)
    frame["sharp_raw"] = check_probabilities(sharp.probability_gru_mean.to_numpy(), frame.label.to_numpy())
    frame["aia_raw"] = aia.probability.to_numpy()
    frame["aia_input_status"] = aia.input_status.to_numpy()
    frame["aia_input_failure_reason"] = aia.input_failure_reason.fillna("").to_numpy()
    frame["sharp_aia_raw"] = .5 * frame.sharp_raw + .5 * frame.aia_raw
    sources["roles_read_from_AIA_archives"] = [role]
    sources["full_inference_completion_required_for_this_earlier_fit"] = False
    return frame, float(summary["train_climatology"]), sources


def fit_matched_calibration(frame, config, climatology):
    """Select on earlier December only; preserve every requested row separately."""
    require(frame.role.eq("probability_calibration").all(), "Only the reserved calibration role may enter this stage")
    require(not frame.forecast_case_id.duplicated().any(), "Duplicate calibration cases")
    require(frame.label_known.all() and frame.label.isin([0, 1]).all(), "Unknown outcomes in calibration cohort")
    require(frame.aia_input_status.isin(["ok", "missing_input", "invalid_input"]).all(), "Undeclared AIA input status")
    check_probabilities(frame.sharp_raw.to_numpy(), frame.label.to_numpy())
    available = frame.aia_input_status.eq("ok")
    check_probabilities(frame.loc[available, "aia_raw"].to_numpy())
    require(frame.loc[~available, "aia_raw"].isna().all(), "Failed input has an AIA probability")
    require(frame.loc[~available, "aia_input_failure_reason"].str.len().gt(0).all(), "Failed input needs a reason")
    expected_fusion = .5 * frame.sharp_raw + .5 * frame.aia_raw
    require(np.array_equal(frame.sharp_aia_raw, expected_fusion, equal_nan=True), "Fusion must use the fixed equal-probability mean")
    # Validate the time boundary on all requested cases before excluding failures.
    calibration_masks(frame, config)
    matched = frame.loc[available].copy()
    eligible, fit, select = calibration_masks(matched, config)
    choices, calibrators, scores = {}, {}, []
    for branch in BRANCHES:
        column = branch + "_raw"
        check_probabilities(matched[column].to_numpy(), matched.label.to_numpy())
        candidates = []
        for rank, method in enumerate(config["candidate_methods"]):
            parameters = fit_calibrator(matched.loc[fit, column].to_numpy(), matched.loc[fit, "label"].to_numpy(), method, config["logit_clip_epsilon"])
            p = apply_calibrator(matched.loc[select, column].to_numpy(), parameters)
            score = score_probabilities(matched.loc[select, "label"].to_numpy(), p, climatology, config, diagnostics=False)
            scores.append({"branch": branch, "method": method, **score})
            candidates.append((score["brier"], score["log_loss"], rank, method))
        choices[branch] = min(candidates)[-1]
        calibrators[branch] = {
            method: fit_calibrator(matched.loc[eligible, column].to_numpy(), matched.loc[eligible, "label"].to_numpy(), method, config["logit_clip_epsilon"])
            for method in config["candidate_methods"]
        }
    support = {name: {"cases": int(mask.sum()), "positive": int(matched.loc[mask, "label"].sum())}
               for name, mask in [("inner_fit", fit), ("inner_selection", select), ("final_fit", eligible),
                                  ("purged_at_inner_boundary", eligible & ~fit & ~select)]}
    return {"selected_methods": choices, "calibrators": calibrators, "inner_scores": scores,
            "matched_support": support, "requested_cases": len(frame), "input_failure_cases": int((~available).sum()),
            "fit_case_ids_sha256": hashlib.sha256("\n".join(matched.loc[eligible, "forecast_case_id"]).encode()).hexdigest(),
            "fusion_weights": {"sharp": .5, "aia": .5}, "train_climatology": climatology,
            "evaluation_roles_read": [], "scientific_acceptance": False}


def run(repo, config, output):
    """Persist source receipts and all requested calibration rows; no later scores."""
    repo, output = Path(repo), Path(output)
    require(not output.exists(), "Use a new output directory; preserve existing fits")
    for name, expected in config["source_sha256"].items():
        require(sha256(repo / name) == expected, "Calibration dependency changed: " + name)
    frame, climatology, sources = load_complete_role(repo, config)
    selection = fit_matched_calibration(frame, config, climatology)
    output.mkdir(parents=True)
    save_json(output / "config.json", config)
    save_json(output / "source_receipt.json", sources)
    save_json(output / "selection.json", selection)
    # Row preservation makes exclusions visible, rather than hiding them in fitting.
    columns = ["forecast_case_id", "role", "issue_utc", "outcome_end_utc_72h", "region_component_id", "label", "label_known",
               "sharp_raw", "aia_raw", "sharp_aia_raw", "aia_input_status", "aia_input_failure_reason"]
    frame[columns].to_csv(output / "calibration_cases.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(selection["inner_scores"]).to_csv(output / "inner_selection_scores.csv", index=False)
    summary = {"status": "completed_earlier_matched_calibration_not_evaluation",
               "completed_utc": datetime.now(timezone.utc).isoformat(),
               "selected_methods": selection["selected_methods"], "requested_cases": len(frame),
               "input_failure_cases": selection["input_failure_cases"], "matched_support": selection["matched_support"],
               "model_refits": 0, "GPU_calls": 0, "evaluation_roles_read": [],
               "scientific_acceptance": False, "source_sha256": sha256(Path(__file__)),
               "limitations": ["Candidate labels and historical source availability remain provisional.",
                               "Method choice on December 2014 does not establish later performance.",
                               "Full-population availability, conformal and operational policy evaluation remain separate stages.",
                               "Later Cycle-25/2026 results are retrospective; no independent or prospective validation is claimed."],
               "output_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    save_json(output / "summary.json", summary)
    return summary
