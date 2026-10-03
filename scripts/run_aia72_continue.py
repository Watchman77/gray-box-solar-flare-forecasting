"""Exactly one externally approved short training slot, with explicit GPU handback."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.aia72_replay_contract import (require,sha,read_hashed_json,write_json,verify_inputs,LOCK,LOCK_INODE)
from scripts.aia72_continue_contract import (ROOT,CONSUMED,check_contract,check_allowance,verify_initial_training,
                                           verify_completed17,verify_bundle,guard)
from scripts.run_aia72_replay import supervise,require_clean_worker,gpu_pids,live_group,interrupted


def publish(path,receipt,check):
    if receipt['status']=='bounded_training_invocation_verified':
        try:
            check();write_json(path,receipt);check();return
        except BaseException as exc:
            receipt.update(status='continuation_failed_or_incomplete',error=repr(exc))
    write_json(path,receipt)


def handback(c,receipt,output,handoff_sha,hard_end):
    """Explicit return under the still-held original lease; never infer release from exit."""
    require(time.monotonic()<hard_end,'No handback time remains')
    worker=receipt.get('worker_pid')
    require(not worker or not live_group(worker),'Owned worker descendants remain')
    require(not gpu_pids(),'GPU still occupied')
    owner=Path(c['owner_pointer'])
    require(sha(owner)==handoff_sha,'Owner changed; cannot hand back another reservation')
    returned={'status':'GPU_RELEASED_TO_AIA','utc':datetime.now(timezone.utc).isoformat(),
              'reason':'One bounded training continuation invocation ended; no automatic next invocation',
              'common_lock':LOCK,'common_lock_inode':LOCK_INODE,'review_root':ROOT,
              'supersedes_reservation_sha256':handoff_sha,'worker_pid':worker,
              'worker_process_group_empty':True,'gpu_empty_under_exclusive_lock':True,
              'execution_receipt':str(Path(output)/'execution_receipt.json'),
              'execution_receipt_sha256':sha(Path(output)/'execution_receipt.json'),
              'execution_status':receipt['status'],'scientific_acceptance':False,
              'automatic_retry':False,'new_fitting_allowed_by_this_receipt':False,
              'AIA_must_acquire_common_lock_and_recheck_GPU':True}
    path=Path(ROOT)/'handoff_return_to_aia.json'
    require(not path.exists(),'Prior return must be preserved')
    write_json(path,returned)
    temporary=owner.with_suffix('.continuation_return.tmp')
    with temporary.open('xb') as stream:stream.write(path.read_bytes())
    require(sha(owner)==handoff_sha,'Owner changed during return')
    temporary.replace(owner)
    require(sha(owner)==sha(path),'Return pointer differs')
    return {'path':str(path),'sha256':sha(path),'utc':returned['utc']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['contract-sha256','bundle-sha256','authorization','handoff']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    require(root==(Path(ROOT)/'bundle').resolve(),'Only canonical continuation bundle may run')
    c,csha=read_hashed_json(root/'configs/aia72_continuation_v1.json');check_contract(c)
    require(csha==args.contract_sha256,'Unreviewed continuation contract')
    a,asha=read_hashed_json(args.authorization);h,hsha=read_hashed_json(args.handoff)
    bounds=check_allowance(a,h,c,csha,args.bundle_sha256,asha)
    require(sha(c['prior_return_path'])==c['prior_return_sha256'],'Prior AIA return changed')
    with Path(LOCK).open('r+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(os.fstat(lease.fileno()).st_ino==LOCK_INODE,'Persistent lock inode differs')
        # Recompute charged handoff-to-launch idle after acquiring the original lease.
        bounds=check_allowance(a,h,c,csha,args.bundle_sha256,asha)
        started=time.monotonic();hard_end=started+bounds['remaining_total_seconds'];work_end=hard_end-60
        claimroot=Path(CONSUMED);claimroot.mkdir(exist_ok=True)
        claim=claimroot/(asha+'.json')
        with claim.open('x') as stream:
            json.dump({'utc':datetime.now(timezone.utc).isoformat(),'status':'consumed_before_preflight',
                       'authorization_sha256':asha,'charged_before_launch_seconds':bounds['charged_before_launch_seconds'],
                       'automatic_retry':False,'maximum_total_reservation_seconds':a['max_slot_seconds']},stream)
        output=Path(ROOT)/'execution';output.mkdir(exist_ok=False)
        training=Path(ROOT)/'training'
        receipt={'status':'continuation_started_unverified','started_utc':datetime.now(timezone.utc).isoformat(),
                 'contract_sha256':csha,'bundle_sha256':args.bundle_sha256,'authorization_sha256':asha,
                 'handoff_sha256':hsha,'handoff_utc':h['utc'],'common_lock_inode':LOCK_INODE,
                 'max_slot_seconds':a['max_slot_seconds'],'charged_before_launch_seconds':bounds['charged_before_launch_seconds'],
                 'automatic_retry':False,'scientific_acceptance':False}
        write_json(output/'execution_receipt.json',receipt)
        def check(deadline=work_end):
            guard(c,output,training,args.authorization,asha,args.handoff,hsha,deadline)
        old={sig:signal.signal(sig,interrupted) for sig in [signal.SIGALRM,signal.SIGTERM,signal.SIGINT]}
        signal.setitimer(signal.ITIMER_REAL,max(.001,work_end-time.monotonic()))
        try:
            verify_bundle(root,args.bundle_sha256);verify_inputs(c)
            verify_initial_training(c,Path(ROOT)/'initial_training');verify_initial_training(c,training)
            check();require(not gpu_pids(),'GPU occupied under shared lock')
            # Bound the whole worker (including notebook setup/checkpointing) by
            # max_fitting_seconds, so a batch cannot overrun the fitting allowance.
            child_end=min(work_end,time.monotonic()+a['max_fitting_seconds'])
            context={'mode':'authorized_training_continuation','bundle_root':str(root),'output':str(output),
                     'training':str(training),'work_end':child_end,'max_fitting_seconds':a['max_fitting_seconds'],
                     'authorization':args.authorization,'authorization_sha256':asha,'handoff':args.handoff,'handoff_sha256':hsha}
            env=dict(os.environ,PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                     OMP_NUM_THREADS='1',GRAYBOX_CONTINUATION_LEASE_FD=str(lease.fileno()),
                     GRAYBOX_CONTINUATION_CONTEXT=json.dumps(context))
            with (output/'execution.log').open('x') as log:
                receipt.update(supervise([sys.executable,'-u','-m','scripts.aia72_continue'],env,root,log,
                                         child_end,hard_end,(lease.fileno(),),lambda:check(child_end)))
            signal.setitimer(signal.ITIMER_REAL,max(.001,hard_end-time.monotonic()-10))
            require_clean_worker(receipt);check(hard_end-10)
            verify_bundle(root,args.bundle_sha256);verify_inputs(c)
            verify_initial_training(c,Path(ROOT)/'initial_training');verify_completed17(c,training)
            result=json.loads((output/'training_result.json').read_text())
            require(result['status'] in ['checkpointed_incomplete','remaining_seeds_fit_pending_replay'],'Missing bounded training result')
            for row in result['checkpoints']:
                require(sha(training/f"seed_{row['seed']}_resume.pt")==row['resume_sha256'],'Checkpoint changed after worker')
            receipt.update(status='bounded_training_invocation_verified',training_status=result['status'],
                           additional_steps=result['additional_steps'],result_sha256=sha(output/'training_result.json'))
        except BaseException as exc:
            receipt.update(status='continuation_failed_or_incomplete',error=repr(exc))
        finally:
            signal.setitimer(signal.ITIMER_REAL,max(.001,hard_end-time.monotonic()-5))
            receipt['finished_utc']=datetime.now(timezone.utc).isoformat()
            receipt['charged_reservation_seconds_before_handback']=bounds['charged_before_launch_seconds']+time.monotonic()-started
            try:
                receipt['output_files']={str(p.relative_to(training)):sha(p) for p in training.glob('*') if p.is_file()}
                publish(output/'execution_receipt.json',receipt,lambda:check(hard_end-5))
            except BaseException as exc:
                receipt.update(status='continuation_failed_or_incomplete',error=repr(exc))
                write_json(output/'execution_receipt.json',receipt)
            signal.setitimer(signal.ITIMER_REAL,max(.001,hard_end-time.monotonic()))
            # A STOP can end work yet still permit clean explicit return; fixed owner
            # and live child/GPU checks remain mandatory even after a failed invocation.
            try:
                require(sha(args.authorization)==asha and sha(args.handoff)==hsha,'Control bytes changed; manual review required')
                returned=handback(c,receipt,output,hsha,hard_end)
                charged=bounds['charged_before_launch_seconds']+time.monotonic()-started
                write_json(Path(ROOT)/'slot_ledger.json',{'authorization_sha256':asha,'status':'reservation_returned',
                    'charged_total_reserved_seconds':charged,'maximum_slot_seconds':a['max_slot_seconds'],
                    'remaining_allowance_seconds':0,'invocation_consumed':True,'return_receipt':returned,
                    'overrun_seconds':max(0,charged-a['max_slot_seconds']),'automatic_retry':False})
            except BaseException as exc:
                write_json(Path(ROOT)/'return_requires_review.json',{'status':'return_unverified','error':repr(exc),
                    'authorization_sha256':asha,'automatic_retry':False})
                receipt['handback_error']=repr(exc)
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,handler in old.items():signal.signal(sig,handler)
            print(json.dumps(receipt,indent=2),flush=True)
        if receipt['status']!='bounded_training_invocation_verified' or receipt.get('handback_error'):raise SystemExit(1)


if __name__=='__main__':main()
