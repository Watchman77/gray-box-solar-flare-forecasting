"""CPU collection boundary counterexamples and cancellation/resource safeguards."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest import mock

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Hide CUDA before these checks'
import scripts.archive_aia72_completion_training as collection


def state_for(losses):
    history=[];best=None;best_epoch=None;stale=0
    for epoch,loss in enumerate(losses,1):
        if best is None or loss<best:best,best_epoch,stale=loss,epoch,0
        else:stale+=1
        history.append({'seed':43,'epoch':epoch,'selection_log_loss':loss,'best_epoch':best_epoch})
    state={'history':history,'epoch':len(history)+1,'cursor':0,'loss_sum':0,
           'steps':1600*len(history),'best_epoch':best_epoch,'best_loss':best,
           'stale':stale,'elapsed_seconds':123.}
    complete={'best_epoch':best_epoch,'selection_log_loss':best,'epochs_completed':len(history),'elapsed_seconds':123.}
    return state,complete


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class CollectionControlTests(unittest.TestCase):
    def test_valid_first_patience_stop_and_equal_losses(self):
        for losses in [[.3,.4,.4,.4,.4],[.3,.3,.3,.3,.3]]:
            state,complete=state_for(losses)
            result=collection.validate_history(state,complete,43)
            self.assertEqual((result['first_stopping_epoch'],result['best_epoch']),(5,1))

    def test_preserved_six_and_ten_epoch_review_counterexamples_are_rejected(self):
        for losses in [[.3,.4,.4,.4,.4,.4],[.3,.4,.4,.4,.4,.2,.3,.3,.3,.3]]:
            with self.subTest(losses=losses), self.assertRaisesRegex(ValueError,'first mandatory stop at epoch 5'):
                collection.validate_history(*state_for(losses),seed=43)

    def test_each_historical_best_epoch_must_match(self):
        state,complete=state_for([.5,.4,.3,.4,.4,.4,.4])
        state['history'][1]['best_epoch']=1
        with self.assertRaisesRegex(ValueError,'Historical best_epoch'):
            collection.validate_history(state,complete,43)

    def test_premature_completion_and_inconsistent_final_state_rejected(self):
        state,complete=state_for([.3,.4,.4,.4])
        collection.validate_history(state,None,43)
        with self.assertRaisesRegex(ValueError,'precedes'):
            collection.validate_history(state,complete,43)
        for field,value in [('stale',2),('best_epoch',2),('best_loss',.2)]:
            state,complete=state_for([.3,.4,.4,.4,.4]);state[field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Final best/stale'):
                collection.validate_history(state,complete,43)

    def test_original_maximum_and_invalid_history_rejected(self):
        state,complete=state_for([1-i*.01 for i in range(20)])
        self.assertEqual(collection.validate_history(state,complete,43)['first_stopping_epoch'],20)
        with self.assertRaisesRegex(ValueError,'20-epoch'):
            collection.validate_history(*state_for([1-i*.01 for i in range(21)]),seed=43)
        for field,value in [('epoch',99),('seed',17),('selection_log_loss',float('nan'))]:
            state,complete=state_for([.3,.4,.4,.4,.4]);state['history'][2][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                collection.validate_history(state,complete,43)

    def test_known_accepted_seed17_history_matches_real_saved_state(self):
        root=Path(__file__).resolve().parents[1]/'outputs/compute_limit_archive_20261003/snapshot/outputs/training'
        if not (root/'seed_17_resume.pt').exists():self.skipTest('Private preserved checkpoint is not present')
        import torch
        self.assertFalse(torch.cuda.is_available())
        saved=torch.load(root/'seed_17_resume.pt',map_location='cpu',weights_only=True)
        complete=json.loads((root/'seed_17_complete.json').read_text())
        result=collection.validate_history(saved['state'],complete,17)
        self.assertEqual((result['first_stopping_epoch'],result['best_epoch']),(6,2))
        self.assertFalse(torch.cuda.is_initialized())

    def test_inner_deadline_interrupts_and_restores_signal_handlers(self):
        signals=[signal.SIGALRM,signal.SIGTERM,signal.SIGINT]
        before={number:signal.getsignal(number) for number in signals}
        with self.assertRaises(InterruptedError):
            with collection.bounded_collection(.03):time.sleep(.2)
        self.assertEqual(before,{number:signal.getsignal(number) for number in signals})
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.,0.))

    def test_outer_command_requires_CPU_and_both_reviewed_sources(self):
        command=collection.expected_supervisor_command('/source/collector.py','return','collector','launcher')
        self.assertEqual(command[:4],['/usr/bin/timeout','--signal=TERM','--kill-after=10s','600s'])
        self.assertIn('CUDA_VISIBLE_DEVICES=',command)
        self.assertEqual(command[-6:],['--return-sha256','return','--source-sha256','collector','--launcher-sha256','launcher'])

    def fixture(self,tmp):
        parent=Path(tmp);root=parent/'run';root.mkdir()
        (root/'training').mkdir()
        original=parent/'graybox_aia72_workers4_20261002';original.mkdir()
        returned=root/'handoff_return_to_aia.json';returned.write_text('same return')
        owner=original/'resource_reservation_status.json';owner.write_text('same return')
        source=parent/'collector.py';source.write_text('test collector')
        launcher=parent/'run_aia72_cpu_collection.sh';launcher.write_text('test launcher')
        lock=parent/'.aia19b2_run.lock';lock.write_text('persistent')
        return root,source,launcher,returned,owner,lock

    def test_continuing_deadline_source_owner_STOP_reserve_and_lease_checks(self):
        for changed in ['none','deadline','source','launcher','owner','return','stop','reserve','lease']:
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as tmp:
                root,source,launcher,returned,owner,lock=self.fixture(tmp)
                source_sha,launcher_sha,return_sha=digest(source),digest(launcher),digest(returned)
                disk=type('Disk',(),{'free':30*1024**3})()
                deadline=time.monotonic()+30
                with lock.open('r+') as lease,mock.patch.object(collection,'COMMON_INODE',lock.stat().st_ino), \
                     mock.patch.object(collection.shutil,'disk_usage',return_value=disk):
                    if changed in ['source','launcher','owner','return']:
                        {'source':source,'launcher':launcher,'owner':owner,'return':returned}[changed].write_text('changed')
                    elif changed=='deadline':deadline=time.monotonic()-1
                    elif changed=='stop':(root/'training/STOP_REQUESTED').write_text('stop')
                    elif changed=='reserve':disk.free=19*1024**3
                    elif changed=='lease':lock.unlink();lock.write_text('replaced')
                    args=(root,source,source_sha,launcher_sha,return_sha,lease.fileno(),deadline)
                    if changed=='none':collection.check_collection_guard(*args)
                    else:
                        with self.assertRaises((ValueError,TimeoutError)):collection.check_collection_guard(*args)


if __name__=='__main__':unittest.main()
