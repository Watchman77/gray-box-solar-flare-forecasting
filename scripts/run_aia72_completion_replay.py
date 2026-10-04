"""One bounded inference-only invocation after verified v5 fitting completion."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.aia72_replay_contract import (
    LOCK,LOCK_INODE,require,sha,read_hashed_json,write_json,verify_inputs,verify_bundle,utc)
from scripts.aia72_completion_replay_contract import (
    ROOT,CONSUMED,TRAINING,check_contract,check_allowance,guard)
from scripts.run_aia72_replay import (
    supervise,require_clean_worker,gpu_pids,live_group,interrupted,publish_receipt)


def check_finished_training(paths):
    receipt=json.loads(paths['training_execution'].read_text())
    result=json.loads(paths['training_result'].read_text())
    prior=json.loads(paths['prior_return'].read_text())
    require(receipt['status']=='bounded_training_invocation_verified' and receipt['returncode']==0,
            'Training invocation is not verified')
    require(receipt['process_group_empty'] and not receipt['work_deadline_reached'], 'Training cleanup incomplete')
    require(not receipt.get('cancellation_signals') and receipt.get('guard_error') is None, 'Training cancellation/guard failure')
    require(sha(paths['training_result'])==receipt['result_sha256'], 'Training result receipt differs')
    require(result['remaining_seeds_fit_complete'] and result['status']=='remaining_seeds_fit_pending_replay',
            'Remaining models have not completed fitting')
    require({row['seed'] for row in result['checkpoints']}=={29,43}
            and all(row['fit_complete'] for row in result['checkpoints']), 'Seed completion records incomplete')
    require(prior['status']=='GPU_RELEASED_TO_AIA' and prior['review_root']==TRAINING
            and prior['execution_receipt_sha256']==sha(paths['training_execution']), 'Training return differs')
    return prior


def verify_replay_result(output):
    path=Path(output)/'replay/result.json'
    result=json.loads(path.read_text())
    require(result['status']=='technical_three_seed_replay_verified_pending_controller', 'Replay incomplete')
    require(result['replayed_seeds']==[29,43] and result['seed17_replay_reused']
            and result['seed17_repeated'] is False and result['fitting_steps']==0, 'Replay scope changed')
    require(result['selection_cases']==3905 and result['inputs_unchanged'], 'Replay support/input integrity differs')
    reports=result['comparisons']
    require([r['seed'] for r in reports]==[29,43]
            and all(r['cases']==3905 and r['all_logits_within_tolerance'] and r['mismatched_cases']==0
                    and r['atol']==1e-6 and r['rtol']==1e-5 and r['fitting_steps']==0 for r in reports),
            'Both saved-model comparisons must pass')
    expected={'three_seed_selection_predictions.csv.gz'}
    expected |= {f'seed_{seed}_comparison.{suffix}' for seed in [29,43] for suffix in ['csv.gz','json']}
    require(set(result['output_sha256'])==expected,'Replay artifact set incomplete')
    for name,digest in result['output_sha256'].items():
        artifact=(path.parent/name).resolve()
        require(artifact.is_relative_to(path.parent.resolve()) and artifact.is_file()
                and sha(artifact)==digest,'Replay output changed: '+name)
    return result,sha(path)


def return_resource(c, receipt, output, handoff_sha, hard_end):
    require(time.monotonic()<hard_end, 'No cleanup time remains')
    worker=receipt.get('worker_pid')
    require(not worker or not live_group(worker), 'Owned worker descendants remain')
    require(not gpu_pids(), 'GPU still occupied')
    owner=Path(c['owner_pointer'])
    require(sha(owner)==handoff_sha, 'Cannot return a changed reservation')
    returned={'status':'GPU_RELEASED_TO_AIA','utc':datetime.now(timezone.utc).isoformat(),
              'reason':'Completed-model replay invocation ended; Gray total-analysis scheduling priority remains retained',
              'review_root':ROOT,'common_lock':LOCK,'common_lock_inode':LOCK_INODE,
              'supersedes_reservation_sha256':handoff_sha,'worker_pid':worker,
              'worker_process_group_empty':True,'gpu_empty_under_exclusive_lock':True,
              'execution_receipt':str(Path(output)/'execution_receipt.json'),
              'execution_receipt_sha256':sha(Path(output)/'execution_receipt.json'),
              'execution_status':receipt['status'],'automatic_retry':False,
              'scientific_acceptance':False,'total_analysis_priority_returned':False,
              'new_GPU_work_allowed_by_this_receipt':False}
    path=Path(ROOT)/'handoff_return_to_aia.json'
    require(not path.exists(),'Preserve prior return')
    with path.open('x') as stream:stream.write(json.dumps(returned,indent=2)+'\n')
    temporary=owner.with_suffix('.completion_replay_return.tmp')
    with temporary.open('xb') as stream:stream.write(path.read_bytes())
    require(sha(owner)==handoff_sha,'Owner changed during return')
    temporary.replace(owner)
    require(sha(owner)==sha(path),'Return pointer differs')
    return {'path':str(path),'sha256':sha(path),'utc':returned['utc']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['contract-sha256','bundle-sha256','authorization','handoff']:
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    bundle=Path(__file__).resolve().parents[1]
    require(bundle==(Path(ROOT)/'bundle').resolve(),'Only the canonical reviewed bundle may run')
    c,csha=read_hashed_json(bundle/'configs/aia72_completion_replay_v1.json');check_contract(c)
    require(csha==a.contract_sha256,'Unreviewed contract')
    authorization,asha=read_hashed_json(a.authorization)
    handoff,hsha=read_hashed_json(a.handoff)
    check_allowance(authorization,handoff,c,csha,a.bundle_sha256,asha)
    with Path(LOCK).open('r+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(os.fstat(lease.fileno()).st_ino==LOCK_INODE,'Persistent lease inode changed')
        bounds=check_allowance(authorization,handoff,c,csha,a.bundle_sha256,asha)
        started=time.monotonic();hard_end=started+bounds['remaining_total_seconds'];work_end=hard_end-60
        claimroot=Path(CONSUMED);claimroot.mkdir(exist_ok=True)
        with (claimroot/(asha+'.json')).open('x') as stream:
            json.dump({'status':'consumed_before_preflight','authorization_sha256':asha,
                       'utc':datetime.now(timezone.utc).isoformat(),'automatic_retry':False,
                       'charged_before_launch_seconds':bounds['charged_before_launch_seconds']},stream)
        output=Path(ROOT)/'execution';output.mkdir(exist_ok=False)
        receipt={'status':'completed_model_replay_started_unverified','started_utc':datetime.now(timezone.utc).isoformat(),
                 'contract_sha256':csha,'bundle_sha256':a.bundle_sha256,'authorization_sha256':asha,
                 'handoff_sha256':hsha,'handoff_utc':handoff['utc'],'max_slot_seconds':authorization['max_slot_seconds'],
                 'charged_before_launch_seconds':bounds['charged_before_launch_seconds'],
                 'common_lock_inode':LOCK_INODE,'fitting_steps':0,'scientific_acceptance':False,'automatic_retry':False}
        write_json(output/'execution_receipt.json',receipt)
        def check(deadline=work_end):
            guard(c,output,a.authorization,asha,a.handoff,hsha,deadline)
        old={sig:signal.signal(sig,interrupted) for sig in [signal.SIGALRM,signal.SIGTERM,signal.SIGINT]}
        signal.setitimer(signal.ITIMER_REAL,max(.001,work_end-time.monotonic()))
        try:
            verify_bundle(bundle,a.bundle_sha256)
            paths=verify_inputs(c)
            prior=check_finished_training(paths)
            require(utc(prior['utc'])<utc(handoff['utc']),'Old or simultaneous handoff rejected')
            check();require(not gpu_pids(),'GPU occupied under persistent lease')
            context={'mode':'authorized_completed_model_replay','bundle_root':str(bundle),'output':str(output),
                     'work_end':work_end,'authorization':a.authorization,'authorization_sha256':asha,
                     'handoff':a.handoff,'handoff_sha256':hsha}
            env=dict(os.environ,PYTHONPATH=str(bundle),PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                     OMP_NUM_THREADS='1',GRAYBOX_COMPLETION_REPLAY_LEASE_FD=str(lease.fileno()),
                     GRAYBOX_COMPLETION_REPLAY_CONTEXT=json.dumps(context))
            with (output/'execution.log').open('x') as log:
                receipt.update(supervise([sys.executable,'-u','-m','scripts.aia72_completion_replay_worker'],
                                         env,bundle,log,work_end,hard_end,(lease.fileno(),),check))
            signal.setitimer(signal.ITIMER_REAL,max(.001,hard_end-time.monotonic()-10))
            require_clean_worker(receipt);check(hard_end-10)
            verify_bundle(bundle,a.bundle_sha256);verify_inputs(c)
            _,result_sha=verify_replay_result(output)
            receipt.update(status='technical_checkpoint_replay_verified',result_sha256=result_sha,
                           replayed_seeds=[29,43],seed17_repeated=False,total_analysis_complete=False)
        except BaseException as error:
            receipt.update(status='review_incomplete_or_failed',error=repr(error))
        finally:
            signal.setitimer(signal.ITIMER_REAL,max(.001,hard_end-time.monotonic()-5))
            receipt['finished_utc']=datetime.now(timezone.utc).isoformat()
            receipt['charged_reservation_seconds_before_return']=bounds['charged_before_launch_seconds']+time.monotonic()-started
            publish_receipt(output/'execution_receipt.json',receipt,lambda:check(hard_end-5))
            signal.setitimer(signal.ITIMER_REAL,max(.001,hard_end-time.monotonic()))
            try:
                require(sha(a.authorization)==asha and sha(a.handoff)==hsha,'Control bytes changed; return needs review')
                returned=return_resource(c,receipt,output,hsha,hard_end)
                charged=bounds['charged_before_launch_seconds']+time.monotonic()-started
                write_json(Path(ROOT)/'slot_ledger.json',{'status':'reservation_returned','authorization_sha256':asha,
                    'charged_total_reserved_seconds':charged,'maximum_slot_seconds':authorization['max_slot_seconds'],
                    'remaining_allowance_seconds':0,'invocation_consumed':True,'return_receipt':returned,
                    'overrun_seconds':max(0,charged-authorization['max_slot_seconds']),'automatic_retry':False})
            except BaseException as error:
                write_json(Path(ROOT)/'return_requires_review.json',{'status':'return_unverified','error':repr(error),
                    'authorization_sha256':asha,'automatic_retry':False})
                receipt['return_error']=repr(error)
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,handler in old.items():signal.signal(sig,handler)
            print(json.dumps(receipt,indent=2),flush=True)
        if receipt['status']!='technical_checkpoint_replay_verified' or receipt.get('return_error'):
            raise SystemExit(1)


if __name__=='__main__':
    main()
