"""Real-data AIA sequence, optimization and saved-checkpoint technical check."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.aia72_data import AIA72Dataset, LAGS, fit_normalization, transform_image, validate_frame_identity
from scripts.aia72_model import TemporalAIACNNGRU
from scripts.aia_io import CHANNELS, load_aia_frame
from scripts.dataset_io import file_sha256
from scripts.prepare_multimodal72 import write_json


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        raw = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(raw.dtype).encode()); digest.update(str(raw.shape).encode()); digest.update(raw.tobytes())
    return digest.hexdigest()


def forward(model, loader):
    model.eval()
    rows = []
    with torch.no_grad():
        for images, _, identities in loader:
            logits = model(images)
            if logits.shape != (len(identities),) or not torch.isfinite(logits).all():
                raise ValueError("Invalid model output")
            rows.extend({"forecast_case_id": identity, "technical_logit": float(logit), "technical_only": True}
                        for identity, logit in zip(identities, logits))
    return pd.DataFrame(rows)


def run(root, stage, output):
    root, stage, output = map(Path, [root, stage, output])
    if output.exists():
        raise ValueError("Output exists; preserve prior checks")
    source = json.loads((stage / "complete.json").read_text())
    plan = json.loads((stage / "plan.json").read_text())
    if source["plan_sha256"] != file_sha256(stage / "plan.json") or plan["cases_sha256"] != file_sha256(stage / "cases.csv.gz"):
        raise ValueError("Staging plan/cases changed")
    cases = pd.read_csv(stage / "cases.csv.gz", low_memory=False)
    records = {r["uri"]: r for r in source["objects"]}
    required = set(cases[[f"history_uri_tminus{x}" for x in LAGS]].to_numpy().ravel())
    if len(cases) != 14 or len(records) != len(source["objects"]) or set(records) != required:
        raise ValueError("Canary scope differs from its fixed selection")
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    np.random.seed(17); torch.manual_seed(17)
    started = time.monotonic()
    image_checks = []
    for uri, record in records.items():
        path = root / record["path_relative_to_repository"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("Pinned source content changed")
        validate_frame_identity(path, uri)
        raw, checked = load_aia_frame(path)
        image_checks.append({"uri": uri, "generation": record["generation"], "sha256": record["sha256"],
                             "metadata_schema": checked["metadata_schema"], "minimum": float(raw.min()),
                             "maximum": float(raw.max()), "shape": str(raw.shape), "finite": checked["finite"]})
    checks = pd.DataFrame(image_checks)
    checks.to_csv(output / "image_checks.csv", index=False)
    normalization = fit_normalization(cases, records, root)
    normalization["scope"] = "Technical canary training rows only; not full-experiment fitted statistics"
    write_json(output / "normalization.json", normalization)
    dataset = AIA72Dataset(cases, records, root, normalization)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    train_cases = cases[cases.role.eq("train")].copy()
    train_loader = DataLoader(AIA72Dataset(train_cases, records, root, normalization), batch_size=2, shuffle=False, num_workers=0)
    model = TemporalAIACNNGRU()
    initial_state = state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    training_rows = []
    for step in range(2):
        model.train()
        images, labels, identities = next(iter(train_loader))
        if images.shape[1:] != (3, 6, 256, 256):
            raise ValueError("Unexpected AIA sequence shape")
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite technical training loss")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum() > 0 for g in grads):
            raise ValueError("Missing, nonfinite or zero gradients")
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        training_rows.append({"technical_step": step + 1, "training_cases": len(labels), "bce_loss": float(loss.detach()),
                              "gradient_norm_before_clip": float(norm), "case_ids": "|".join(identities)})
    pd.DataFrame(training_rows).to_csv(output / "technical_training_steps.csv", index=False)
    fitted_state = state_digest(model)
    if fitted_state == initial_state:
        raise ValueError("Training did not change model state")
    logits = forward(model, loader)
    if state_digest(model) != fitted_state:
        raise ValueError("Evaluation changed model buffers or weights")
    checkpoint = output / "technical_canary_only.pt"
    torch.save(model.state_dict(), checkpoint)
    replay = TemporalAIACNNGRU()
    replay.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    saved_norm = json.loads((output / "normalization.json").read_text())
    replay_loader = DataLoader(AIA72Dataset(cases, records, root, saved_norm), batch_size=2, shuffle=False, num_workers=0)
    replay_logits = forward(replay, replay_loader)
    if not logits.equals(replay_logits) or state_digest(replay) != fitted_state:
        raise ValueError("Saved-model replay differs")
    # Channelwise independent expression checks the broadcast transform on all
    # checked objects. Resizing uses the same declared PyTorch interpolation API.
    maximum_difference = 0.
    for uri in records:
        raw, _ = load_aia_frame(root / records[uri]["path_relative_to_repository"])
        channels = []
        for channel in range(6):
            v = np.arcsinh(raw[:, :, channel] / np.float32(saved_norm["channel_scale"][channel]))
            channels.append((v - np.float32(saved_norm["channel_mean"][channel])) / np.float32(saved_norm["channel_std"][channel]))
        independent = torch.nn.functional.interpolate(torch.from_numpy(np.stack(channels))[None], size=(256, 256), mode="bilinear", align_corners=False)[0]
        actual = transform_image(raw, saved_norm)
        maximum_difference = max(maximum_difference, float((independent - actual).abs().max()))
        torch.testing.assert_close(independent, actual, atol=1e-6, rtol=1e-6)
    logits.to_csv(output / "technical_logits_not_forecasts.csv", index=False)
    result = {"status": "technical_aia72_canary_passed_not_scientific_training", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "cases": len(cases), "roles": cases.role.value_counts().to_dict(), "objects_verified": len(records),
              "metadata_schemas": checks.metadata_schema.value_counts().to_dict(), "sequence_shape": [3, 6, 256, 256],
              "model_parameters": sum(p.numel() for p in model.parameters()), "technical_optimization_steps": 2,
              "preprocessing_fit_case_ids": normalization["training_case_ids"], "preprocessing_fit_objects": normalization["stats_files"],
              "source_48h_labels_used": False, "saved_model_replay": "exact_all_14_cases_same_batching",
              "evaluation_changed_weights_or_buffers": False, "independent_transform_max_absolute_difference": maximum_difference,
              "initial_state_sha256": initial_state, "final_state_sha256": fitted_state,
              "runtime": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "device": "cpu", "threads": 2},
              "seconds": time.monotonic() - started, "full_72h_training_complete": False,
              "stage_receipt_sha256": file_sha256(stage / "complete.json"),
              "code_sha256": {name: file_sha256(root / "scripts" / name) for name in ["run_aia72_canary.py", "aia72_data.py", "aia72_model.py", "aia_io.py"]},
              "limitations": ["Fourteen boundary cases are a software check, not representative model evaluation",
                              "Two optimization steps use only two training-role cases; weights/statistics must not initialize the scientific run",
                              "Image identities/channel order/pixels checked; full archive quality and actual per-channel observation/delivery histories are not certified",
                              "GOES is not an input to this AIA-only canary"],
              "output_sha256": {p.name: file_sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / "summary.json", result)
    return result, cases, checks, normalization, training_rows
