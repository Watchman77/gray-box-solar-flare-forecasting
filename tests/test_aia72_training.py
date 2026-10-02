"""Restart equivalence and cached-source integrity tests for the shared-VM run."""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from scripts.aia72_vm_data import read_source
from scripts.aia_io import CHANNELS
from scripts.train_aia72_gpu import train_seed


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(2, 4), nn.Dropout(.3), nn.Linear(4, 1))
    def forward(self, x):
        return self.head(x).squeeze(-1)


class TinyCases(Dataset):
    def __init__(self):
        self.frame = pd.DataFrame({'forecast_case_id': [f'case{i}' for i in range(8)], 'label': [0, 1] * 4})
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, i):
        return torch.tensor([i / 10., 1.]), torch.tensor(float(i % 2)), f'case{i}'


class AIA72TrainingTests(unittest.TestCase):
    def test_mid_epoch_resume_preserves_optimizer_rng_and_results(self):
        torch.set_num_threads(1)
        config = {'batch_size': 2, 'num_workers': 0, 'learning_rate': .001, 'weight_decay': .0001,
                  'max_epochs': 2, 'patience': 4, 'checkpoint_interval_seconds': 999, 'gradient_clip': 1.}
        with tempfile.TemporaryDirectory() as tmp:
            full, resumed = Path(tmp) / 'full', Path(tmp) / 'resumed'
            cases = TinyCases()
            args = (cases, cases, config, 17)
            train_seed(*args, full, 'fixed', 'cpu', time.monotonic() + 60, model_factory=TinyModel)
            interrupted = train_seed(*args, resumed, 'fixed', 'cpu', time.monotonic() + 60,
                                     model_factory=TinyModel, stop_after_steps=2)
            self.assertEqual(interrupted['status'], 'test_interruption')
            train_seed(*args, resumed, 'fixed', 'cpu', time.monotonic() + 60, model_factory=TinyModel)
            a = torch.load(full / 'seed_17_resume.pt', weights_only=True)
            b = torch.load(resumed / 'seed_17_resume.pt', weights_only=True)
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key], b['model'][key]), key)
            self.assertEqual(a['state']['history'], b['state']['history'])
            for key in a['optimizer']['state']:
                for name, value in a['optimizer']['state'][key].items():
                    self.assertTrue(torch.equal(value, b['optimizer']['state'][key][name]))
            with self.assertRaisesRegex(ValueError, 'contract'):
                train_seed(*args, resumed, 'different', 'cpu', time.monotonic() + 60, model_factory=TinyModel)

    def test_read_source_requires_pinned_bytes_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'image.npz'
            np.savez(path, x=np.zeros((512, 512, 6), np.float32), channels=CHANNELS, sample_id='image', y=0)
            payload = path.read_bytes()
            record = {'path': str(path), 'uri': 'gs://source/image.npz', 'bytes': len(payload),
                      'md5_base64': base64.b64encode(hashlib.md5(payload).digest()).decode()}
            raw, sha = read_source(record)
            self.assertEqual(raw.shape, (512, 512, 6)); self.assertEqual(sha, hashlib.sha256(payload).hexdigest())
            with self.assertRaisesRegex(ValueError, 'pinned'):
                read_source({**record, 'bytes': len(payload) + 1})
            with self.assertRaisesRegex(ValueError, 'identity'):
                read_source({**record, 'uri': 'gs://source/wrong.npz'})


if __name__ == '__main__':
    unittest.main()
