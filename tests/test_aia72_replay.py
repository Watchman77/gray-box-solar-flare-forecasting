"""CPU checks for scientific rejection rules and actual bounded child cleanup."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.aia72_replay import ReplayDeadline, compare_predictions, infer_rows
from scripts.aia72_replay_contract import (LOCK, LOCK_INODE, REVIEW_ROOT, check_authorization, check_guards, check_fixed_review_guards, sha, verify_bundle, verify_inputs, write_json)
from scripts.run_aia72_replay import check_launch_location, live_group, publish_receipt, require_clean_worker, supervise


class ToyModel(nn.Module):
    def __init__(self, mutate=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.))
        self.register_buffer('count', torch.tensor(0))
        self.mutate = mutate
    def forward(self, images):
        if self.mutate:
            self.count += 1
        return images[:, 0] * self.weight


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.expected = pd.DataFrame({'forecast_case_id': ['a', 'b', 'c'], 'label': [0, 1, 0], 'logit': [0., 2., -2.]})
        self.batches = [(torch.tensor([[0.], [1.]]), torch.tensor([0., 1.]), ['a', 'b']),
                        (torch.tensor([[-1.]]), torch.tensor([0.]), ['c'])]

    def test_success_uses_eval_and_does_not_change_weights(self):
        model = ToyModel()
        rows = infer_rows(model, self.batches, self.expected, 'cpu', time.monotonic() + 10)
        comparison, result = compare_predictions(rows, self.expected)
        self.assertFalse(model.training)
        self.assertTrue(result['all_logits_within_tolerance'])
        self.assertEqual(result['max_absolute_logit_difference'], 0)
        self.assertEqual(model.weight.item(), 2.)

    def test_rejects_reordered_or_missing_cases(self):
        with self.assertRaisesRegex(ValueError, 'reordered'):
            infer_rows(ToyModel(), list(reversed(self.batches)), self.expected, 'cpu', time.monotonic() + 10)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            infer_rows(ToyModel(), self.batches[:1], self.expected, 'cpu', time.monotonic() + 10)

    def test_rejects_changed_labels(self):
        changed = self.expected.copy(); changed.loc[0, 'label'] = 1
        with self.assertRaisesRegex(ValueError, 'outcomes'):
            infer_rows(ToyModel(), self.batches, changed, 'cpu', time.monotonic() + 10)

    def test_rejects_model_buffer_mutation(self):
        with self.assertRaisesRegex(ValueError, 'mutated'):
            infer_rows(ToyModel(mutate=True), self.batches, self.expected, 'cpu', time.monotonic() + 10)

    def test_expired_replay_cannot_succeed(self):
        with self.assertRaises(ReplayDeadline):
            infer_rows(ToyModel(), self.batches, self.expected, 'cpu', time.monotonic() - 1)

    def test_comparison_does_not_relax_tolerance(self):
        changed = self.expected.copy(); changed.loc[1, 'logit'] += .01
        comparison, result = compare_predictions(changed, self.expected)
        self.assertFalse(result['all_logits_within_tolerance'])
        self.assertEqual(result['mismatched_cases'], 1)

    def test_comparison_rejects_nonfinite_and_duplicate_rows(self):
        changed = self.expected.copy(); changed.loc[1, 'logit'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            compare_predictions(changed, self.expected)
        changed = self.expected.copy(); changed.loc[1, 'forecast_case_id'] = 'a'
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            compare_predictions(changed, self.expected)

    def test_input_and_source_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = root / 'weights'; p.write_bytes(b'original')
            contract = {'source_root': str(root), 'inputs': {'checkpoint': {'path': 'weights', 'sha256': sha(p)}}}
            verify_inputs(contract)
            manifest = root / 'replay_bundle_manifest.json'
            manifest.write_text(json.dumps({'files': {'weights': sha(p)}})); expected = sha(manifest)
            verify_bundle(root, expected)
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Input hash'):
                verify_inputs(contract)
            with self.assertRaisesRegex(ValueError, 'source changed'):
                verify_bundle(root, expected)

    def authorization_fixture(self):
        now = datetime.now(timezone.utc)
        authorization = {'status': 'AUTHORIZED_GPU_REVIEW', 'kind': 'aia72_seed17_saved_checkpoint_replay',
                         'contract_sha256': 'contract', 'bundle_sha256': 'bundle', 'max_slot_seconds': 1800,
                         'fitting_allowed': False, 'user_approval_reference': 'synthetic test only',
                         'review_root': REVIEW_ROOT,
                         'not_before_utc': (now - timedelta(minutes=1)).isoformat(),
                         'expires_utc': (now + timedelta(minutes=1)).isoformat()}
        handoff = {'status': 'GPU_RELEASED_TO_GRAYBOX_REPLAY', 'authorization_sha256': 'authorization',
                   'contract_sha256': 'contract', 'bundle_sha256': 'bundle', 'common_lock': LOCK,
                   'common_lock_inode': LOCK_INODE, 'utc': now.isoformat(), 'fitting_allowed': False}
        handoff['review_root'] = REVIEW_ROOT
        prior = {'utc': (now - timedelta(hours=1)).isoformat()}
        return authorization, handoff, prior, now

    def test_fresh_separate_allowance_and_handoff_required(self):
        a, h, prior, now = self.authorization_fixture()
        check_authorization(a, h, 'contract', 'bundle', 'authorization', prior, now)
        for field, value in [('status', 'PROPOSAL_ONLY'), ('fitting_allowed', True), ('max_slot_seconds', 3600),
                             ('contract_sha256', 'different'), ('user_approval_reference', '')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_authorization({**a, field: value}, h, 'contract', 'bundle', 'authorization', prior, now)
        with self.assertRaisesRegex(ValueError, 'fresh'):
            check_authorization(a, {**h, 'status': 'GPU_RELEASED_TO_GRAYBOX'}, 'contract', 'bundle', 'authorization', prior, now)
        with self.assertRaisesRegex(ValueError, 'Old release'):
            check_authorization(a, {**h, 'utc': prior['utc']}, 'contract', 'bundle', 'authorization', prior, now)
        with self.assertRaisesRegex(ValueError, 'too old'):
            check_authorization(a, {**h, 'utc': (now-timedelta(minutes=20)).isoformat()}, 'contract', 'bundle', 'authorization', prior, now)

    def test_supervisor_terminates_unresponsive_child_and_grandchild(self):
        program = ('import subprocess,sys,signal,time\n'
                   'signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
                   'subprocess.Popen([sys.executable,"-c","import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"])\n'
                   'time.sleep(30)\n')
        with tempfile.TemporaryFile(mode='w+') as log:
            start = time.monotonic()
            result = supervise([sys.executable, '-c', program], os.environ.copy(), Path.cwd(), log,
                               start + .5, start + 8)
        self.assertTrue(result['work_deadline_reached'])
        self.assertTrue(result['process_group_empty'])
        self.assertFalse(live_group(result['worker_pid']))
        self.assertLess(time.monotonic() - start, 8)

    def test_supervisor_keeps_normal_exit_successful(self):
        with tempfile.TemporaryFile(mode='w+') as log:
            start = time.monotonic()
            result = supervise([sys.executable, '-c', 'print("synthetic only")'], os.environ.copy(), Path.cwd(), log,
                               start + 3, start + 5)
        self.assertEqual(result['returncode'], 0)
        self.assertFalse(result['work_deadline_reached'])
        self.assertTrue(result['process_group_empty'])

    def test_stop_owner_disk_and_late_output_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); owner = root / 'owner.json'; owner.write_text('{}')
            output = root / 'review' / 'execution'; output.mkdir(parents=True)
            contract = {'source_root': str(root), 'review_root': str(root/'review'),
                        'owner_pointer': str(owner), 'minimum_free_bytes': 100}
            with mock.patch('scripts.aia72_replay_contract.shutil.disk_usage', return_value=mock.Mock(free=200)):
                check_guards(contract, output, sha(owner), time.monotonic()+10)
                (root/'STOP_REQUESTED').touch()
                with self.assertRaisesRegex(ValueError, 'STOP'):
                    check_guards(contract, output, sha(owner), time.monotonic()+10)
                (root/'STOP_REQUESTED').unlink()
                with self.assertRaisesRegex(ValueError, 'owner'):
                    check_guards(contract, output, 'old-owner', time.monotonic()+10)
                with self.assertRaisesRegex(ValueError, 'deadline'):
                    check_guards(contract, output, sha(owner), time.monotonic()-1)
                (output/'execution_receipt.json').write_text('{"status":"review_incomplete_or_failed"}')
                with self.assertRaisesRegex(ValueError, 'Recorded failure'):
                    check_guards(contract, output, sha(owner), time.monotonic()+10)
                (output/'execution_receipt.json').unlink()
            with mock.patch('scripts.aia72_replay_contract.shutil.disk_usage', return_value=mock.Mock(free=50)):
                with self.assertRaisesRegex(ValueError, 'reserve'):
                    check_guards(contract, output, sha(owner), time.monotonic()+10)

    def test_copied_bundle_cannot_claim_a_second_root(self):
        check_launch_location(Path(REVIEW_ROOT)/'bundle')
        with self.assertRaisesRegex(ValueError, 'canonical'):
            check_launch_location('/tmp/copied-valid-bundle')

    def test_gpu_or_descendants_block_acceptance(self):
        good = {'process_group_empty': True, 'work_deadline_reached': False,
                'cancellation_signals': [], 'returncode': 0, 'guard_error': None}
        with mock.patch('scripts.run_aia72_replay.gpu_pids', return_value=[]):
            require_clean_worker(good)
            for key,value in [('process_group_empty',False),('work_deadline_reached',True),
                              ('guard_error','STOP'),('cancellation_signals',[signal.SIGTERM])]:
                with self.subTest(key=key),self.assertRaises(ValueError):
                    require_clean_worker({**good,key:value})
        with mock.patch('scripts.run_aia72_replay.gpu_pids', return_value=[123]):
            with self.assertRaisesRegex(ValueError, 'Cleanup incomplete'):
                require_clean_worker(good)

    def test_alarm_during_cleanup_does_not_leave_descendants(self):
        # The grandchild sends ALRM while stop_group is waiting after TERM; both ignore TERM.
        child = ('import os,signal,time\n'
                 'signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
                 'time.sleep(.7)\n'
                 'os.kill(int(os.environ["REPLAY_TEST_CONTROLLER"]),signal.SIGALRM)\n'
                 'time.sleep(30)\n')
        program = ('import subprocess,sys,signal,time\n'
                   'signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
                   f'subprocess.Popen([sys.executable,"-c",{child!r}])\n'
                   'time.sleep(30)\n')
        env={**os.environ,'REPLAY_TEST_CONTROLLER':str(os.getpid())}
        with tempfile.TemporaryFile(mode='w+') as log:
            start=time.monotonic()
            result=supervise([sys.executable,'-c',program],env,Path.cwd(),log,start+.3,start+8)
        self.assertIn(signal.SIGALRM,result['cancellation_signals'])
        self.assertTrue(result['process_group_empty'])
        self.assertFalse(live_group(result['worker_pid']))

    def test_mid_execution_guard_failure_terminates_worker(self):
        count=[0]
        def guard():
            count[0]+=1
            if count[0]>=3:
                raise ValueError('synthetic STOP marker')
        with tempfile.TemporaryFile(mode='w+') as log:
            start=time.monotonic()
            result=supervise([sys.executable,'-c','import time;time.sleep(30)'],os.environ.copy(),Path.cwd(),log,
                             start+4,start+8,guard=guard)
        self.assertIn('STOP',result['guard_error'])
        self.assertTrue(result['process_group_empty'])

    def test_simultaneous_handoff_and_owner_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);owner=root/'owner.json';handoff=root/'handoff.json';auth=root/'allowance.json'
            owner.write_text('{"owner":"this-job"}');handoff.write_bytes(owner.read_bytes());auth.write_text('{}')
            fixed=sha(handoff);auth_sha=sha(auth)
            contract={'source_root':str(root),'review_root':str(root/'review'),'owner_pointer':str(owner),'minimum_free_bytes':100}
            with mock.patch('scripts.aia72_replay_contract.shutil.disk_usage',return_value=mock.Mock(free=200)):
                check_fixed_review_guards(contract,root/'output',auth,auth_sha,handoff,fixed,time.monotonic()+10)
                owner.write_text('{"owner":"another-job"}');handoff.write_bytes(owner.read_bytes())
                with self.assertRaisesRegex(ValueError,'Handoff bytes changed'):
                    check_fixed_review_guards(contract,root/'output',auth,auth_sha,handoff,fixed,time.monotonic()+10)

    def test_changes_during_final_success_write_invalidate_receipt(self):
        for mode in ['STOP','owner','deadline','disk']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);owner=root/'owner.json';owner.write_text('{}');fixed=sha(owner)
                path=root/'execution_receipt.json';end=[time.monotonic()+10];free=[200]
                contract={'source_root':str(root),'review_root':str(root/'review'),'owner_pointer':str(owner),'minimum_free_bytes':100}
                def guard():
                    check_guards(contract,root,fixed,end[0])
                def changed_write(p,value):
                    write_json(p,value)
                    if value['status']=='technical_checkpoint_replay_verified':
                        if mode=='STOP':(root/'STOP_REQUESTED').touch()
                        elif mode=='owner':owner.write_text('{"changed":true}')
                        elif mode=='deadline':end[0]=time.monotonic()-1
                        else:free[0]=0
                receipt={'status':'technical_checkpoint_replay_verified'}
                with mock.patch('scripts.aia72_replay_contract.shutil.disk_usage',side_effect=lambda _:mock.Mock(free=free[0])),mock.patch('scripts.run_aia72_replay.write_json',side_effect=changed_write):
                    publish_receipt(path,receipt,guard)
                self.assertEqual(json.loads(path.read_text())['status'],'review_incomplete_or_failed')
                self.assertTrue(receipt['final_publication_gate_failed'])


if __name__ == '__main__':
    unittest.main()
