"""Reuse pinned VM images with bounded RAM caching and no duplicate image dataset."""
import base64
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from scripts.aia72_data import LAGS, fit_normalization, transform_image, validate_case_times, validate_frame_identity
from scripts.aia_io import load_aia_frame
from scripts.dataset_io import file_sha256


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_source(record):
    path = Path(record['path'])
    payload = path.read_bytes()
    if len(payload) != record['bytes'] or base64.b64encode(hashlib.md5(payload).digest()).decode() != record['md5_base64']:
        raise ValueError('Cached AIA object does not match pinned cloud bytes: ' + record['uri'])
    validate_frame_identity(io.BytesIO(payload), record['uri'])
    raw, _ = load_aia_frame(io.BytesIO(payload))
    return raw, hashlib.sha256(payload).hexdigest()


def bind_inputs(inputs, inventory):
    inputs = Path(inputs)
    receipt = json.loads((inputs / 'preparation.json').read_text())
    for name, key in [('fit_cases.csv.gz', 'fit_cases_sha256'), ('objects.json', 'objects_sha256'),
                      ('parent_contract.json', 'parent_contract_sha256')]:
        if file_sha256(inputs / name) != receipt[key]:
            raise ValueError('Pinned input changed: ' + name)
    frame = pd.read_csv(inputs / 'fit_cases.csv.gz', low_memory=False)
    if not frame.forecast_case_id.is_unique or not frame.label.isin([0, 1]).all():
        raise ValueError('Training requires unique cases and known manifest labels')
    validate_case_times(frame)
    if frame.role.value_counts().to_dict() != {'train': 25586, 'model_validation': 3905}:
        raise ValueError('Frozen training/selection roles changed')
    source = json.loads((inputs / 'objects.json').read_text())
    wanted = set(frame[[f'history_uri_tminus{x}' for x in LAGS]].to_numpy().ravel())
    if set(source) != wanted:
        raise ValueError('Object inventory does not exactly cover the fixed cases')
    cached = pd.read_csv(inventory, dtype={'generation': str}).set_index('object_uri')
    if not cached.index.is_unique:
        raise ValueError('Duplicate cache source identities')
    records = {}
    for uri in sorted(wanted):
        if uri not in cached.index:
            raise ValueError('Required image absent from cache inventory: ' + uri)
        old = cached.loc[uri]
        path = Path(old.local_path).resolve()
        if not path.is_relative_to('/mnt/disks/aia-cache/cycle24'):
            raise ValueError('Unexpected raw-cache path')
        record = source[uri]
        if str(old.generation) != record['generation'] or int(old.size_bytes) != record['bytes']:
            raise ValueError('Cache inventory and pinned cloud version disagree')
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError('Required cached file missing or truncated: ' + str(path))
        records[uri] = {**record, 'path': str(path)}
    return frame, records, receipt


def prepare_normalization(frame, records, output):
    """Fit only the frozen training cases; keep raw images in their existing cache."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'normalization.json'
    expected_ids = sorted(frame.loc[frame.role.eq('train'), 'forecast_case_id'].tolist())
    if path.exists():
        norm = json.loads(path.read_text())
        if norm['training_case_ids'] != expected_ids:
            raise ValueError('Preprocessing fit identities changed')
        return norm
    train = frame[frame.role.eq('train')]
    candidates = sorted(set(train[[f'history_uri_tminus{x}' for x in LAGS]].to_numpy().ravel()))
    selected = sorted(np.random.default_rng(17).choice(candidates, min(1000, len(candidates)), replace=False).tolist())
    stats_records = {}
    for i, uri in enumerate(selected):
        _, digest = read_source(records[uri])
        stats_records[uri] = {'path_relative_to_repository': records[uri]['path'], 'sha256': digest}
        if (i + 1) % 250 == 0:
            print(f'Checked {i + 1}/{len(selected)} training normalization images', flush=True)
    norm = fit_normalization(frame, stats_records, Path('/'), seed=17)
    norm['scope'] = 'Frozen 2010–2013 training cases; deterministic sample of 1000 distinct training images'
    write_json(path, norm)
    return norm


class ExistingAIA72Dataset(Dataset):
    def __init__(self, frame, records, normalization, max_cached_images=96):
        self.frame = frame.reset_index(drop=True).copy()
        validate_case_times(self.frame)
        if not self.frame.forecast_case_id.is_unique or not self.frame.label.isin([0, 1]).all():
            raise ValueError('Unique 72-hour cases and known manifest outcomes required')
        if max_cached_images < 0 or max_cached_images > 256:
            raise ValueError('RAM cache exceeds its declared bound')
        self.records, self.normalization = records, normalization
        self.max_cached_images = max_cached_images
        self.images = OrderedDict()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]; images = []
        for lag in LAGS:
            uri = row[f'history_uri_tminus{lag}']; record = self.records[uri]
            stat = Path(record['path']).stat()
            stamp = (stat.st_size, stat.st_mtime_ns)
            cached = self.images.get(uri)
            if cached is not None and cached[0] == stamp:
                image = cached[1]; self.images.move_to_end(uri)
            else:
                raw, _ = read_source(record)
                image = transform_image(raw, self.normalization)
                if self.max_cached_images:
                    self.images[uri] = (stamp, image)
                    while len(self.images) > self.max_cached_images:
                        self.images.popitem(last=False)
            images.append(image)
        return torch.stack(images), torch.tensor(float(row.label), dtype=torch.float32), row.forecast_case_id
