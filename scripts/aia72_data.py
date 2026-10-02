"""Identity-checked AIA sequences and training-only preprocessing for 72h cases."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from scripts.aia_io import CHANNELS, load_aia_frame

LAGS = (288, 192, 96)


def validate_frame_identity(path, uri):
    """Check the embedded image identity, never read embedded outcome labels."""
    expected = Path(uri).stem
    with np.load(path, allow_pickle=False) as obj:
        if "sample_id" not in obj.files or obj["sample_id"].ndim != 0 or str(obj["sample_id"].item()) != expected:
            raise ValueError("AIA object sample identity mismatch")


def validate_case_times(frame):
    issue = pd.to_datetime(frame.issue_utc, utc=True, format="ISO8601")
    history = [pd.to_datetime(frame[f"history_{lag}_UTC"], utc=True, format="ISO8601") for lag in LAGS]
    if issue.isna().any() or any(x.isna().any() or not x.lt(issue).all() for x in history):
        raise ValueError("History timestamp missing or not before issue")
    if any(not earlier.lt(later).all() for earlier, later in zip(history[:-1], history[1:])):
        raise ValueError("AIA history order is not chronological")


def fit_normalization(frame, records, root, seed=17, max_files=1000, pixels_per_file=1024):
    """Use only training rows; sorted URIs make selection independent of row order."""
    training = frame[frame.role.eq("train")]
    if training.empty:
        raise ValueError("No training rows for preprocessing")
    uris = sorted(set(training[[f"history_uri_tminus{x}" for x in LAGS]].to_numpy().ravel()))
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(uris, size=min(max_files, len(uris)), replace=False).tolist())
    chunks = []
    for uri in chosen:
        record = records[uri]
        path = Path(root) / record["path_relative_to_repository"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Normalization source content checksum changed")
        validate_frame_identity(path, uri)
        raw, _ = load_aia_frame(path)
        flat = raw.reshape(-1, 6)
        sampled = rng.choice(len(flat), min(pixels_per_file, len(flat)), replace=False)
        chunks.append(flat[sampled].astype(np.float64))
    values = np.concatenate(chunks)
    scale = np.percentile(np.abs(values), 75, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.).astype(np.float32)
    transformed = np.arcsinh(values / scale)
    mean = transformed.mean(axis=0).astype(np.float32)
    deviation = transformed.std(axis=0)
    std = np.where(deviation > 1e-6, deviation, 1.).astype(np.float32)
    return {"transform": "asinh(x / training q75 absolute scale), then training z-score",
            "channel_order": CHANNELS, "channel_scale": scale.tolist(), "channel_mean": mean.tolist(),
            "channel_std": std.tolist(), "fit_role": "train", "seed": seed,
            "stats_uris": chosen, "stats_files": len(chosen), "pixels_per_file": pixels_per_file,
            "training_case_ids": sorted(training.forecast_case_id.tolist()),
            "stats_sha256": {uri: records[uri]["sha256"] for uri in chosen}}


def transform_image(raw, normalization, image_size=256):
    if normalization["channel_order"] != CHANNELS:
        raise ValueError("Normalization channel order mismatch")
    scale, mean, std = (np.asarray(normalization[k], np.float32) for k in ["channel_scale", "channel_mean", "channel_std"])
    if any(v.shape != (6,) or not np.isfinite(v).all() for v in [scale, mean, std]) or (scale <= 0).any() or (std <= 0).any():
        raise ValueError("Invalid normalization parameters")
    x = (np.arcsinh(raw.astype(np.float32) / scale.reshape(1, 1, 6)) - mean.reshape(1, 1, 6)) / std.reshape(1, 1, 6)
    tensor = torch.from_numpy(x).permute(2, 0, 1).contiguous()
    if image_size != 512:
        tensor = F.interpolate(tensor[None], size=(image_size, image_size), mode="bilinear", align_corners=False)[0]
    if not torch.isfinite(tensor).all():
        raise ValueError("Nonfinite transformed image")
    return tensor


class AIA72Dataset(Dataset):
    def __init__(self, frame, records, root, normalization, image_size=256):
        self.frame = frame.reset_index(drop=True).copy()
        validate_case_times(self.frame)
        if not self.frame.forecast_case_id.is_unique or not self.frame.label.isin([0, 1]).all():
            raise ValueError("Unique identities and known 72-hour outcomes required")
        self.records, self.root, self.normalization, self.image_size = records, Path(root), normalization, image_size
        self.cache = {}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        sequence = []
        for lag in LAGS:
            uri = row[f"history_uri_tminus{lag}"]
            if uri not in self.records:
                raise ValueError("Missing exact AIA object receipt")
            record = self.records[uri]
            path = self.root / record["path_relative_to_repository"]
            # Each process verifies source bytes before first use; do not silently
            # trust a basename or file size as content identity.
            if uri not in self.cache:
                if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError("AIA content checksum changed")
                validate_frame_identity(path, uri)
                self.cache[uri] = True
            raw, _ = load_aia_frame(path)
            sequence.append(transform_image(raw, self.normalization, self.image_size))
        return torch.stack(sequence), torch.tensor(float(row.label), dtype=torch.float32), row.forecast_case_id
