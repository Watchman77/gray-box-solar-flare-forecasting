"""CPU-only checks of changed continuation, allowance and loader-lease paths."""
from datetime import datetime,timedelta,timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,TensorDataset,get_worker_info
import scripts.aia72_continue as continuation

from scripts.aia72_continue import run_selected,leased_loader,validate_outputs
from scripts.aia72_continue_contract import (ROOT,PARENT,RESUME,RETURN,LOCK,LOCK_INODE,
    budget,check_allowance,verify_initial_training,verify_completed17,verify_bundle)
from scripts.aia72_replay_contract import sha,write_json
from scripts.run_aia72_continue import publish,handback
import scripts.train_aia72_gpu as training


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__();self.head=nn.Sequential(nn.Linear(2,4),nn.Dropout(.3),nn.Linear(4,1))
    def forward(self,x):return self.head(x).squeeze(-1)


class TinyCases(Dataset):
    def __init__(self):self.frame=pd.DataFrame({'forecast_case_id':[f'c{i}' for i in range(8)],'label':[0,1]*4})
    def __len__(self):return 8
    def __getitem__(self,i):return torch.tensor([i/10.,1.]),torch.tensor(float(i%2)),f'c{i}'


class LeaseCases(Dataset):
    def __len__(self):return 32
    def __getitem__(self,i):
        fd=continuation._LOADER_LEASE
        return i,get_worker_info().id,os.fstat(fd).st_ino,os.get_inheritable(fd)


def exact(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


class ContinuationTests(unittest.TestCase):
    def allowance(self):
        now=datetime.now(timezone.utc)
        shared={'contract_sha256':'contract','bundle_sha256':'bundle','review_root':ROOT,
                'parent_contract_sha256':PARENT,'initial_seed29_sha256':RESUME,'seeds':[29,43],'fitting_allowed':True}
        a={**shared,'status':'AUTHORIZED_GPU_TRAINING_CONTINUATION','invocation_count':1,'automatic_retry':False,
           'user_approval_reference':'synthetic test only','max_slot_seconds':3900,'max_fitting_seconds':3600,
           'cleanup_seconds':60,'charge_from':'handoff_utc','not_before_utc':(now-timedelta(minutes=2)).isoformat(),
           'expires_utc':(now+timedelta(hours=2)).isoformat()}
        h={**shared,'status':'GPU_RELEASED_TO_GRAYBOX_CONTINUATION','authorization_sha256':'auth',
           'prior_return_sha256':RETURN,'common_lock':LOCK,'common_lock_inode':LOCK_INODE,
           'utc':(now-timedelta(minutes=1)).isoformat()}
        return a,h,now

    def test_short_slot_charges_handoff_idle(self):
        a,h,now=self.allowance();r=check_allowance(a,h,{},'contract','bundle','auth',now)
        self.assertEqual(r['charged_before_launch_seconds'],60)
        self.assertEqual(r['remaining_total_seconds'],3840)
        self.assertEqual(r['remaining_work_seconds'],3780)

    def test_twelve_hour_proposal_or_repeated_invocation_rejected(self):
        a,h,now=self.allowance()
        for key,value in [('max_slot_seconds',43200),('max_fitting_seconds',3601),('invocation_count',2),
                          ('automatic_retry',True),('charge_from','model_start'),('status','PROPOSAL_ONLY')]:
            with self.subTest(key=key),self.assertRaises(ValueError):check_allowance({**a,key:value},h,{},'contract','bundle','auth',now)

    def test_changed_scope_owner_receipt_and_stale_handoff_rejected(self):
        a,h,now=self.allowance()
        for key,value in [('initial_seed29_sha256','other'),('seeds',[17,29,43]),('review_root','/tmp/copy'),
                          ('authorization_sha256','changed'),('common_lock_inode',1),('prior_return_sha256','old'),
                          ('utc',(now-timedelta(minutes=20)).isoformat())]:
            with self.subTest(key=key),self.assertRaises(ValueError):check_allowance(a,{**h,key:value},{},'contract','bundle','auth',now)

    def test_expiry_shortens_slot_and_no_remaining_time_rejected(self):
        a,h,now=self.allowance();a['expires_utc']=(now+timedelta(seconds=300)).isoformat()
        self.assertEqual(budget(a,h,now)['remaining_total_seconds'],300)
        a['expires_utc']=(now+timedelta(seconds=80)).isoformat()
        with self.assertRaisesRegex(ValueError,'exhausted'):budget(a,h,now)

    def test_initial_snapshot_and_seed17_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'seed_17_best.pt';p.write_bytes(b'original')
            c={'initial_training_files':{p.name:{'sha256':sha(p)}}}
            verify_initial_training(c,tmp);verify_completed17(c,tmp)
            p.write_bytes(b'altered')
            with self.assertRaises(ValueError):verify_initial_training(c,tmp)
            with self.assertRaises(ValueError):verify_completed17(c,tmp)

    def test_bundle_source_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'source.py';p.write_text('fixed')
            manifest=root/'continuation_bundle_manifest.json';write_json(manifest,{'files':{p.name:sha(p)}})
            pinned=sha(manifest);verify_bundle(root,pinned);p.write_text('altered')
            with self.assertRaises(ValueError):verify_bundle(root,pinned)

    def test_incomplete_second_seed_never_starts_third_or_seed17(self):
        trainer=mock.Mock(return_value={'status':'checkpointed_incomplete','seed':29})
        result=run_selected(None,None,{},None,'parent',time.monotonic()+30,lambda:None,trainer)
        self.assertEqual(trainer.call_count,1);self.assertEqual(trainer.call_args.args[3],29)
        self.assertFalse(result['remaining_seeds_fit_complete'])

    def test_mid_epoch_adapter_matches_original_models_optimizer_rng(self):
        torch.set_num_threads(1)
        config={'batch_size':2,'num_workers':0,'learning_rate':.001,'weight_decay':.0001,'max_epochs':2,
                'patience':4,'checkpoint_interval_seconds':999,'gradient_clip':1.}
        cases=TinyCases()
        with tempfile.TemporaryDirectory() as tmp:
            full=Path(tmp)/'full';resumed=Path(tmp)/'resumed'
            for seed in [29,43]:training.train_seed(cases,cases,config,seed,full,'fixed','cpu',time.monotonic()+60,model_factory=TinyModel)
            training.train_seed(cases,cases,config,29,resumed,'fixed','cpu',time.monotonic()+60,model_factory=TinyModel,stop_after_steps=2)
            before=torch.load(resumed/'seed_29_resume.pt',weights_only=True)
            self.assertEqual(before['state']['cursor'],4)
            def cpu_trainer(train,selection,config,seed,out,identity,device,deadline):
                return training.train_seed(train,selection,config,seed,out,identity,'cpu',deadline,model_factory=TinyModel)
            result=run_selected(cases,cases,config,resumed,'fixed',time.monotonic()+60,lambda:None,cpu_trainer)
            self.assertTrue(result['remaining_seeds_fit_complete'])
            for seed in [29,43]:
                a=torch.load(full/f'seed_{seed}_resume.pt',weights_only=True)
                b=torch.load(resumed/f'seed_{seed}_resume.pt',weights_only=True)
                for key in ['model','optimizer','rng']:self.assertTrue(exact(a[key],b[key]),key)
                self.assertEqual(a['state']['history'],b['state']['history'])
            self.assertFalse((resumed/'seed_17_resume.pt').exists())

    def test_loader_child_holds_lock_after_parent_descriptor_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lease';path.touch();parent=path.open('r+')
            fcntl.flock(parent,fcntl.LOCK_EX|fcntl.LOCK_NB)
            data=TensorDataset(torch.arange(16))
            def factory():return DataLoader(data,batch_size=2,num_workers=1,multiprocessing_context='spawn')
            value=leased_loader(factory,parent.fileno());iterator=iter(value)
            try:
                next(iterator);parent.close()
                with path.open('r+') as probe:
                    with self.assertRaises(BlockingIOError):fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:
                iterator._shutdown_workers();parent.close()
            with path.open('r+') as probe:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_four_loaders_hold_lease_without_leaking_to_exec_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lease';path.touch();parent=path.open('r+')
            fcntl.flock(parent,fcntl.LOCK_EX|fcntl.LOCK_NB);inode=path.stat().st_ino
            def factory():return DataLoader(LeaseCases(),batch_size=2,num_workers=4,multiprocessing_context='spawn')
            iterator=iter(leased_loader(factory,parent.fileno()))
            try:
                first=next(iterator);parent.close()
                with path.open('r+') as probe:
                    with self.assertRaises(BlockingIOError):fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                batches=[first,*list(iterator)]
                self.assertEqual([int(v) for b in batches for v in b[0]],list(range(32)))
                self.assertEqual({int(v) for b in batches for v in b[1]},{0,1,2,3})
                self.assertEqual({int(v) for b in batches for v in b[2]},{inode})
                self.assertFalse(any(bool(v) for b in batches for v in b[3]))
            finally:
                iterator._shutdown_workers();parent.close()
                for worker in iterator._workers:worker.join(timeout=2)
            self.assertTrue(all(not w.is_alive() for w in iterator._workers))
            with path.open('r+') as probe:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_leased_loader_preserves_original_training_result(self):
        from functools import partial
        torch.set_num_threads(1)
        config={'batch_size':2,'num_workers':1,'learning_rate':.001,'weight_decay':.0001,'max_epochs':1,
                'patience':4,'checkpoint_interval_seconds':999,'gradient_clip':1.}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lock';path.touch();cases=TinyCases()
            full=Path(tmp)/'original';leased=Path(tmp)/'leased'
            training.train_seed(cases,cases,config,29,full,'fixed','cpu',time.monotonic()+60,model_factory=TinyModel)
            with path.open('r+') as lease:
                fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                base=training.loader
                with mock.patch.object(training,'loader',partial(leased_loader,base,lease.fileno())):
                    training.train_seed(cases,cases,config,29,leased,'fixed','cpu',time.monotonic()+60,model_factory=TinyModel)
            a=torch.load(full/'seed_29_resume.pt',weights_only=True);b=torch.load(leased/'seed_29_resume.pt',weights_only=True)
            for key in ['model','optimizer','rng']:self.assertTrue(exact(a[key],b[key]),key)
            self.assertEqual(a['state']['history'],b['state']['history'])

    def test_guard_failure_prevents_training_call(self):
        trainer=mock.Mock()
        with self.assertRaisesRegex(ValueError,'STOP'):
            run_selected(None,None,{},None,'parent',time.monotonic()+30,mock.Mock(side_effect=ValueError('STOP')),trainer)
        trainer.assert_not_called()

    def test_late_success_write_is_invalidated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'receipt.json';stop=[False]
            def check():
                if stop[0]:raise ValueError('STOP during publication')
            def writing(p,v):
                write_json(p,v)
                if v['status']=='bounded_training_invocation_verified':stop[0]=True
            r={'status':'bounded_training_invocation_verified'}
            with mock.patch('scripts.run_aia72_continue.write_json',side_effect=writing):publish(path,r,check)
            self.assertEqual(json.loads(path.read_text())['status'],'continuation_failed_or_incomplete')

    def test_handback_cannot_overwrite_changed_owner_or_release_live_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);owner=root/'owner';owner.write_text('new-owner')
            c={'owner_pointer':str(owner)}
            with mock.patch('scripts.run_aia72_continue.live_group',return_value=[123]),mock.patch('scripts.run_aia72_continue.gpu_pids',return_value=[]):
                with self.assertRaisesRegex(ValueError,'descendants'):handback(c,{'worker_pid':123},root,'old-owner',time.monotonic()+30)
            with mock.patch('scripts.run_aia72_continue.live_group',return_value=[]),mock.patch('scripts.run_aia72_continue.gpu_pids',return_value=[]):
                with self.assertRaisesRegex(ValueError,'Owner changed'):handback(c,{'worker_pid':123},root,'old-owner',time.monotonic()+30)
            self.assertEqual(owner.read_text(),'new-owner')


if __name__=='__main__':unittest.main()
