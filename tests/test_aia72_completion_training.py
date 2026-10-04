"""CPU checks limited to the new parent binding and mid-epoch boundary."""
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
PREP = Path(os.environ.get('AIA72_NEXT_PREPARATION', REPO/'outputs/aia72_completion_training_v5_20261004')).resolve()
PRIOR = Path(os.environ.get('AIA72_PREDECESSOR', REPO/'outputs/aia72_continuation_epoch6_20261004/completed_snapshot')).resolve()
sys.path.insert(0, str(PREP/'bundle'))

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA before starting this CPU-only test process'

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
import scripts.aia72_continue as continuation
import scripts.aia72_continue_contract as gates
import scripts.train_aia72_gpu as training
from scripts.aia72_replay_contract import check_guards, sha


class TinyCases(Dataset):
    def __init__(self):
        self.frame = pd.DataFrame({'forecast_case_id': [f'c{i}' for i in range(8)], 'label': [0, 1]*4})
        self.accesses = 0
        self.seen = []
    def __len__(self):
        return 8
    def __getitem__(self, i):
        self.accesses += 1
        self.seen.append(i)
        return torch.tensor([i/10., 1.]), torch.tensor(float(i % 2)), f'c{i}'


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(2, 4), nn.Dropout(.3), nn.Linear(4, 1))
    def forward(self, x):
        return self.head(x).squeeze(-1)


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


class NextCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
        torch.set_num_threads(1)
        assert Path(gates.__file__).resolve().is_relative_to(PREP/'bundle')
        cls.c = json.loads((PREP/'bundle/configs/aia72_continuation_v1.json').read_text())
        cls.config = {'batch_size': 2, 'num_workers': 0, 'learning_rate': .001,
                      'weight_decay': .0001, 'max_epochs': 8, 'patience': 4,
                      'checkpoint_interval_seconds': 999, 'gradient_clip': 1.}

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_next_contract_and_original_scientific_parent(self):
        gates.check_contract(self.c)
        old = json.loads((PRIOR/'bundle/configs/aia72_continuation_v1.json').read_text())
        with self.assertRaises(ValueError):
            gates.check_contract(old)
        for key in ['parent_contract_sha256', 'initial_seed29_sha256', 'prior_return_sha256', 'review_root']:
            bad = copy.deepcopy(self.c)
            bad[key] = 'wrong'
            with self.subTest(field=key), self.assertRaises(ValueError):
                gates.check_contract(bad)
        for name in ['aia72_continue.py', 'run_aia72_continue.py', 'train_aia72_gpu.py',
                     'aia72_model.py', 'aia72_vm_data.py', 'aia72_replay_contract.py', 'run_aia72_replay.py']:
            self.assertEqual(sha(PREP/'bundle/scripts'/name), sha(PRIOR/'bundle/scripts'/name))

    def test_real_checkpoint_metadata_and_rejection_of_changed_state(self):
        frame, _, _, config, state = continuation.inspect_inputs(self.c, PREP/'source_mirror', PREP/'initial_training')
        self.assertEqual((state['epoch'], state['cursor'], state['steps'], len(state['history'])), (7, 20416, 10876, 6))
        self.assertEqual(config['patience'], 4)
        self.assertEqual(len(frame), 29491)
        changed = copy.deepcopy(self.c)
        changed['initial_seed29_state']['cursor'] = 0
        with self.assertRaisesRegex(ValueError, 'cursor/state changed'):
            continuation.inspect_inputs(changed, PREP/'source_mirror', PREP/'initial_training')
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(PREP/'initial_training', Path(tmp)/'initial')
            checkpoint = Path(tmp)/'initial/seed_29_resume.pt'
            with checkpoint.open('ab') as stream:
                stream.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'Initial snapshot changed'):
                gates.verify_initial_training(self.c, Path(tmp)/'initial')

    def test_consumed_predecessor_allowance_and_handoff_rejected(self):
        a = json.loads((PRIOR/'aia_coordination_20261003/authorization.json').read_text())
        h = json.loads((PRIOR/'aia_coordination_20261003/handoff_to_graybox_continuation.json').read_text())
        now = datetime.now(timezone.utc)
        with self.assertRaisesRegex(ValueError, 'Work digest changed'):
            gates.check_allowance(a, h, self.c, 'next-contract', 'next-bundle', 'old-auth', now)
        # Synthetic documents are in-memory only; these do not grant real compute.
        for item in [a, h]:
            item.update(contract_sha256='next-contract', bundle_sha256='next-bundle',
                        review_root=gates.ROOT, initial_seed29_sha256=gates.RESUME)
        a.update(not_before_utc=(now-timedelta(seconds=2)).isoformat(),
                 expires_utc=(now+timedelta(seconds=3900)).isoformat())
        h.update(utc=now.isoformat(), authorization_sha256='synthetic', prior_return_sha256=gates.RETURN)
        gates.check_allowance(a, h, self.c, 'next-contract', 'next-bundle', 'synthetic', now)
        old = json.loads((PRIOR/'bundle/configs/aia72_continuation_v1.json').read_text())
        for key, value in [('initial_seed29_sha256', old['initial_seed29_sha256']),
                           ('prior_return_sha256', old['prior_return_sha256'])]:
            bad = {**h, key: value}
            with self.subTest(field=key), self.assertRaises(ValueError):
                gates.check_allowance(a, bad, self.c, 'next-contract', 'next-bundle', 'synthetic', now)

    def test_continuous_budget_requires_new_bound_and_charges_setup(self):
        a = json.loads((PRIOR/'aia_coordination_20261003/authorization.json').read_text())
        h = json.loads((PRIOR/'aia_coordination_20261003/handoff_to_graybox_continuation.json').read_text())
        now = datetime.now(timezone.utc)
        for item in [a, h]:
            item.update(contract_sha256='continuous-contract', bundle_sha256='continuous-bundle',
                        review_root=gates.ROOT, initial_seed29_sha256=gates.RESUME)
        a.update(max_slot_seconds=129600, max_fitting_seconds=126000,
                 not_before_utc=(now-timedelta(seconds=301)).isoformat(),
                 expires_utc=(now+timedelta(seconds=129300)).isoformat())
        h.update(utc=(now-timedelta(seconds=300)).isoformat(),
                 authorization_sha256='synthetic-only', prior_return_sha256=gates.RETURN)
        limits = gates.check_allowance(a, h, self.c, 'continuous-contract', 'continuous-bundle', 'synthetic-only', now)
        self.assertEqual(limits['charged_before_launch_seconds'], 300)
        self.assertEqual(limits['remaining_total_seconds'], 129300)
        self.assertEqual(limits['maximum_fitting_seconds'], 126000)
        for key, value in [('max_slot_seconds',129601),('max_fitting_seconds',126001),
                           ('max_slot_seconds',True),('max_fitting_seconds',0)]:
            with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                gates.check_allowance({**a,key:value},h,self.c,'continuous-contract','continuous-bundle','synthetic-only',now)
        for key, value in [('maximum_slot_seconds',3900),('maximum_fitting_seconds',3600)]:
            with self.subTest(contract_field=key), self.assertRaises(ValueError):
                gates.check_contract({**self.c,key:value})
        shortened = {**a,'expires_utc':(now+timedelta(seconds=100)).isoformat()}
        self.assertEqual(gates.check_allowance(shortened,h,self.c,'continuous-contract','continuous-bundle','synthetic-only',now)['maximum_fitting_seconds'],20)

    def test_changed_current_owner_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner = Path(tmp)/'owner.json'
            owner.write_bytes((PRIOR/'handoff_return_to_aia.json').read_bytes())
            self.assertEqual(sha(owner), gates.RETURN)
            owner.write_text('owner changed after review')
            c = {**self.c, 'owner_pointer': str(owner)}
            with self.assertRaisesRegex(ValueError, 'Current GPU owner differs'):
                check_guards(c, tmp, gates.RETURN, time.monotonic()+30)

    def test_epoch7_midpoint_resumes_remaining_cases_and_matches_uninterrupted_run(self):
        train, selection = TinyCases(), TinyCases()
        with tempfile.TemporaryDirectory() as tmp:
            full, resumed = Path(tmp)/'full', Path(tmp)/'resumed'
            training.train_seed(train, selection, self.config, 29, full, 'fixed', 'cpu', time.monotonic()+30, model_factory=TinyModel)
            training.train_seed(train, selection, self.config, 29, resumed, 'fixed', 'cpu', time.monotonic()+30,
                                model_factory=TinyModel, stop_after_steps=26)
            before = torch.load(resumed/'seed_29_resume.pt', weights_only=True)
            self.assertEqual((before['state']['epoch'], before['state']['cursor'], len(before['state']['history'])), (7, 4, 6))
            train.accesses = 0
            train.seen = []
            expected_order = torch.randperm(8, generator=torch.Generator().manual_seed(29+100000+7)).tolist()[4:]
            validation_epochs = []
            original_validation = training.validation_predictions
            def validate(*args, **kwargs):
                checkpoint = torch.load(resumed/'seed_29_resume.pt', weights_only=True)
                validation_epochs.append(checkpoint['state']['epoch'])
                if len(validation_epochs) == 1:
                    self.assertEqual(train.seen, expected_order)
                    self.assertEqual(train.accesses, 4)
                    self.assertEqual(checkpoint['state']['steps'], 28)
                return original_validation(*args, **kwargs)
            with mock.patch.object(training, 'validation_predictions', side_effect=validate):
                training.train_seed(train, selection, self.config, 29, resumed, 'fixed', 'cpu', time.monotonic()+30, model_factory=TinyModel)
            self.assertEqual(validation_epochs, [7, 8])
            self.assertEqual(train.accesses, 12)
            a = torch.load(full/'seed_29_resume.pt', weights_only=True)
            b = torch.load(resumed/'seed_29_resume.pt', weights_only=True)
            for key in ['model', 'optimizer', 'rng']:
                self.assertTrue(exact(a[key], b[key]), key)
            for key in ['epoch', 'cursor', 'steps', 'loss_sum', 'stale', 'history', 'best_loss', 'best_epoch']:
                self.assertEqual(a['state'][key], b['state'][key])

    def test_expired_work_deadline_preserves_midpoint_and_does_not_start_seed43(self):
        train, selection = TinyCases(), TinyCases()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            training.train_seed(train, selection, self.config, 29, out, 'fixed', 'cpu', time.monotonic()+30,
                                model_factory=TinyModel, stop_after_steps=26)
            before = torch.load(out/'seed_29_resume.pt', weights_only=True)
            called = []
            def trainer(t, v, c, seed, path, parent, device, deadline):
                called.append(seed)
                return training.train_seed(t, v, c, seed, path, parent, 'cpu', time.monotonic()-1, model_factory=TinyModel)
            result = continuation.run_selected(train, selection, self.config, out, 'fixed',
                                                time.monotonic()+30, lambda: None, trainer=trainer)
            self.assertEqual(called, [29])
            self.assertEqual(result['status'], 'checkpointed_incomplete')
            self.assertFalse((out/'seed_43_resume.pt').exists())
            self.assertFalse((out/'seed_29_complete.json').exists())
            after = torch.load(out/'seed_29_resume.pt', weights_only=True)
            for key in ['model', 'optimizer', 'rng']:
                self.assertTrue(exact(before[key], after[key]), key)
            for key in ['epoch', 'cursor', 'steps', 'loss_sum', 'stale', 'history']:
                self.assertEqual(before['state'][key], after['state'][key])


if __name__ == '__main__':
    unittest.main(verbosity=2)
