"""CPU-only phase-boundary, allowance, stop and explicit-return checks."""
import copy
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Hide CUDA before CPU checks'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scripts.aia72_completion_replay_contract as gates
import scripts.run_aia72_completion_replay as controller
from scripts.aia72_replay_contract import sha


def contract():
    return {'kind':gates.KIND,'review_root':gates.ROOT,'source_root':gates.ROOT+'/inputs',
            'consumed_root':gates.CONSUMED,'training_root':gates.TRAINING,'parent_contract_sha256':gates.PARENT,
            'owner_pointer':gates.ORIGINAL+'/resource_reservation_status.json',
            'common_lock':gates.LOCK,'common_lock_inode':gates.LOCK_INODE,'seeds':[29,43],
            'ensemble_seeds':[17,29,43],'fitting_allowed':False,'automatic_retry':False,'invocation_count':1,
            'maximum_slot_seconds':1800,'cleanup_seconds':60,'minimum_free_bytes':20*1024**3,
            'inputs':{'prior_return':{'path':'prior_return.json','sha256':'a'*64}}}


def allowance(c,now):
    base={'kind':gates.KIND,'review_root':gates.ROOT,'contract_sha256':'contract','bundle_sha256':'bundle',
          'seeds':[29,43],'fitting_allowed':False}
    a={**base,'status':'AUTHORIZED_GPU_REVIEW','automatic_retry':False,'invocation_count':1,
       'max_slot_seconds':1800,'cleanup_seconds':60,'charge_from':'handoff_utc','user_approval_reference':'synthetic_test_only',
       'not_before_utc':(now-timedelta(seconds=101)).isoformat(),'expires_utc':(now+timedelta(seconds=1700)).isoformat()}
    h={**base,'status':'GPU_RELEASED_TO_GRAYBOX_REPLAY','utc':(now-timedelta(seconds=100)).isoformat(),
       'authorization_sha256':'auth','prior_return_sha256':c['inputs']['prior_return']['sha256'],
       'common_lock':gates.LOCK,'common_lock_inode':gates.LOCK_INODE}
    return a,h


class ReplayControlTests(unittest.TestCase):
    def test_control_imports_do_not_load_torch(self):
        command=[sys.executable,'-c',"import scripts.run_aia72_completion_replay, sys; assert 'torch' not in sys.modules"]
        result=subprocess.run(command,cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_contract_requires_new_phase_and_forbids_fitting(self):
        c=contract();gates.check_contract(c)
        for field,value in [('fitting_allowed',True),('seeds',[17,29,43]),('maximum_slot_seconds',3600),
                            ('review_root',gates.TRAINING),('common_lock_inode',0),('automatic_retry',True)]:
            with self.subTest(field=field),self.assertRaises(ValueError):
                gates.check_contract({**c,field:value})

    def test_handoff_idle_and_expiry_reduce_budget(self):
        c=contract();now=datetime.now(timezone.utc);a,h=allowance(c,now)
        limits=gates.check_allowance(a,h,c,'contract','bundle','auth',now)
        self.assertEqual(limits,{'charged_before_launch_seconds':100,'remaining_total_seconds':1700,'remaining_work_seconds':1640})
        short={**a,'expires_utc':(now+timedelta(seconds=100)).isoformat()}
        self.assertEqual(gates.check_allowance(short,h,c,'contract','bundle','auth',now)['remaining_work_seconds'],40)
        short['expires_utc']=(now+timedelta(seconds=90)).isoformat()
        with self.assertRaisesRegex(ValueError,'exhausted'):
            gates.check_allowance(short,h,c,'contract','bundle','auth',now)

    def test_wrong_old_handoff_and_oversized_or_repeated_allowance_rejected(self):
        c=contract();now=datetime.now(timezone.utc);a,h=allowance(c,now)
        for field,value in [('max_slot_seconds',1801),('max_slot_seconds',True),('fitting_allowed',True),
                            ('invocation_count',2),('automatic_retry',True),('kind','training_continuation')]:
            with self.subTest(field=field),self.assertRaises(ValueError):
                gates.check_allowance({**a,field:value},h,c,'contract','bundle','auth',now)
        for field,value in [('prior_return_sha256','old'),('authorization_sha256','old'),('bundle_sha256','old'),
                            ('common_lock_inode',1)]:
            with self.subTest(field=field),self.assertRaises(ValueError):
                gates.check_allowance(a,{**h,field:value},c,'contract','bundle','auth',now)

    def test_incomplete_training_cannot_cross_replay_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            paths={name:root/(name+'.json') for name in ['training_result','training_execution','prior_return']}
            result={'status':'remaining_seeds_fit_pending_replay','remaining_seeds_fit_complete':True,
                    'checkpoints':[{'seed':s,'fit_complete':True} for s in [29,43]]}
            def write_chain(result, extra_receipt=None):
                paths['training_result'].write_text(json.dumps(result))
                receipt={'status':'bounded_training_invocation_verified','returncode':0,'process_group_empty':True,
                         'work_deadline_reached':False,'result_sha256':sha(paths['training_result']),**(extra_receipt or {})}
                paths['training_execution'].write_text(json.dumps(receipt))
                paths['prior_return'].write_text(json.dumps({'status':'GPU_RELEASED_TO_AIA','review_root':gates.TRAINING,
                    'execution_receipt_sha256':sha(paths['training_execution'])}))
            write_chain(result);controller.check_finished_training(paths)
            for changed in [{**result,'remaining_seeds_fit_complete':False},
                            {**result,'checkpoints':[{'seed':29,'fit_complete':True}]}]:
                write_chain(changed)
                with self.assertRaises(ValueError):controller.check_finished_training(paths)
            write_chain(result,{'cancellation_signals':[15]})
            with self.assertRaisesRegex(ValueError,'cancellation'):controller.check_finished_training(paths)

    def test_stop_marker_in_previous_training_execution_blocks_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=root/'original';training=root/'training';inputs=root/'inputs';output=root/'output'
            for p in [original,training,inputs,output,training/'execution']:p.mkdir()
            owner=root/'owner';owner.write_text('fixed-handoff')
            auth=root/'authorization';auth.write_text('fixed-authorization')
            c={**contract(),'source_root':str(inputs),'review_root':str(root),'owner_pointer':str(owner)}
            disk=type('Disk',(),{'free':30*1024**3})()
            with mock.patch.object(gates,'ORIGINAL',str(original)),mock.patch.object(gates,'TRAINING',str(training)), \
                 mock.patch('scripts.aia72_replay_contract.shutil.disk_usage',return_value=disk):
                gates.guard(c,output,auth,sha(auth),owner,sha(owner),time.monotonic()+30)
                (training/'execution/STOP_REQUESTED').write_text('stop')
                with self.assertRaisesRegex(ValueError,'STOP/failure'):
                    gates.guard(c,output,auth,sha(auth),owner,sha(owner),time.monotonic()+30)

    def test_explicit_return_retains_priority_and_rejects_live_descendants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'execution';output.mkdir()
            (output/'execution_receipt.json').write_text('{"status":"technical_checkpoint_replay_verified"}')
            owner=root/'owner.json';owner.write_text('initial-handoff');before=sha(owner)
            c={'owner_pointer':str(owner)};r={'status':'technical_checkpoint_replay_verified','worker_pid':4321,
                'worker_launch_state':'started','cleanup_verified':True,'worker_reaped':True,
                'process_group_empty':True,'cleanup_errors':[]}
            with mock.patch.object(controller,'ROOT',str(root)),mock.patch.object(controller,'gpu_pids',return_value=[]), \
                 mock.patch.object(controller,'live_group',return_value=[4321]):
                with self.assertRaisesRegex(ValueError,'descendants'):
                    controller.return_resource(c,r,output,before,time.monotonic()+30)
                self.assertEqual(sha(owner),before)
            with mock.patch.object(controller,'ROOT',str(root)),mock.patch.object(controller,'gpu_pids',return_value=[]), \
                 mock.patch.object(controller,'live_group',return_value=[]):
                returned=controller.return_resource(c,r,output,before,time.monotonic()+30)
            self.assertEqual(sha(owner),returned['sha256'])
            self.assertFalse(json.loads(owner.read_text())['total_analysis_priority_returned'])

    def test_unknown_or_unverified_worker_cannot_return_even_with_idle_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'execution';output.mkdir()
            (output/'execution_receipt.json').write_text('{}')
            owner=root/'owner.json';owner.write_text('initial-handoff');before=sha(owner)
            good={'status':'review_incomplete_or_failed','worker_launch_state':'started','worker_pid':4321,
                  'cleanup_verified':True,'worker_reaped':True,'process_group_empty':True,'cleanup_errors':[]}
            bad=[{}, {'worker_launch_state':'attempting'}, {**good,'worker_pid':None},
                 {**good,'worker_reaped':False}, {**good,'cleanup_verified':False},
                 {**good,'cleanup_errors':[{'error':'PermissionError'}]},
                 {**good,'worker_launch_state':'not_attempted'}]
            with mock.patch.object(controller,'ROOT',str(root)),mock.patch.object(controller,'gpu_pids',return_value=[]), \
                 mock.patch.object(controller,'live_group',return_value=[]):
                for receipt in bad:
                    with self.subTest(receipt=receipt),self.assertRaises(ValueError):
                        controller.return_resource({'owner_pointer':str(owner)},receipt,output,before,time.monotonic()+30)
                    self.assertEqual(sha(owner),before)
                    self.assertFalse((root/'handoff_return_to_aia.json').exists())


if __name__=='__main__':
    unittest.main()
