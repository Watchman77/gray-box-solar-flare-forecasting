"""One reviewed frozen-model inference invocation, with bounded cleanup and return."""
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
from scripts.aia72_replay_contract import LOCK,LOCK_INODE,require,sha,read_hashed_json,write_json,verify_inputs,verify_bundle
from scripts.aia72_inference_contract import ROOT,RETURN,check_contract,check_allowance,guard
from scripts.run_aia72_replay import require_clean_worker,gpu_pids,live_group,interrupted
from scripts.aia72_replay_supervisor import supervise,require_cleanup_proof


def publish_receipt(path,receipt,guard):
    if receipt['status']=='all_inference_blocks_verified_pending_collection':
        try:
            guard();write_json(path,receipt);guard()
        except BaseException as error:
            receipt.update(status='inference_incomplete_or_failed',error=repr(error),final_publication_gate_failed=True)
    write_json(path,receipt)


def verify_result(output):
    path=Path(output)/'predictions/result.json';result=json.loads(path.read_text())
    require(result['status']=='all_prediction_blocks_complete_pending_collection','Prediction phase incomplete')
    require(result['fitting_steps']==0 and result['models_unchanged'] is True,'Inference changed fitting state')
    require(result['inferred_cases']==92691 and result['accepted_selection_cases_reused']==3905,'Case support incomplete')
    require(len(result['blocks'])==184 and len({r['block_id'] for r in result['blocks']})==184,'Block set incomplete')
    for r in result['blocks']:
        require(sha(path.parent/(r['block_id']+'_predictions.csv.gz'))==r['predictions_sha256'],'Prediction block changed')
        require(sha(path.parent/(r['block_id']+'_sources.json'))==r['sources_sha256'],'Source receipt changed')
    return sha(path)


def return_resource(c,receipt,output,handoff_sha,hard_end):
    require(time.monotonic()<hard_end,'No cleanup time remains');require_cleanup_proof(receipt)
    if receipt['worker_launch_state']=='started':require(not live_group(receipt['worker_pid']),'Owned worker group remains')
    require(not gpu_pids(),'GPU occupied after inference cleanup')
    owner=Path(c['owner_pointer']);require(sha(owner)==handoff_sha,'Owner changed before return')
    returned={'status':'GPU_RELEASED_TO_AIA','utc':datetime.now(timezone.utc).isoformat(),
              'reason':'Inference invocation ended; Gray total-analysis priority retained',
              'review_root':ROOT,'common_lock':LOCK,'common_lock_inode':LOCK_INODE,
              'supersedes_reservation_sha256':handoff_sha,'worker_pid':receipt['worker_pid'],
              'worker_launch_state':receipt['worker_launch_state'],'cleanup_verified':receipt.get('cleanup_verified',False),
              'worker_process_group_empty':True,'gpu_empty_under_exclusive_lock':True,
              'execution_receipt':str(Path(output)/'execution_receipt.json'),
              'execution_receipt_sha256':sha(Path(output)/'execution_receipt.json'),'execution_status':receipt['status'],
              'automatic_retry':False,'scientific_acceptance':False,'total_analysis_priority_returned':False,
              'new_GPU_work_allowed_by_this_receipt':False}
    path=Path(ROOT)/'handoff_return_to_aia.json'
    with path.open('x') as stream:stream.write(json.dumps(returned,indent=2)+'\n')
    temporary=owner.with_suffix('.inference_return.tmp')
    with temporary.open('xb') as stream:stream.write(path.read_bytes());stream.flush();os.fsync(stream.fileno())
    require(sha(owner)==handoff_sha,'Owner changed during return');temporary.replace(owner)
    require(sha(owner)==sha(path),'Return publication differs')
    return {'path':str(path),'sha256':sha(path),'utc':returned['utc']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['contract-sha256','bundle-sha256','authorization','handoff']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args();bundle=Path(__file__).resolve().parents[1]
    require(bundle==(Path(ROOT)/'bundle').resolve(),'Only canonical reviewed source may run')
    c,csha=read_hashed_json(bundle/'configs/aia72_inference_v1.json');check_contract(c)
    require(csha==args.contract_sha256,'Unreviewed inference contract')
    a,asha=read_hashed_json(args.authorization);h,hsha=read_hashed_json(args.handoff)
    check_allowance(a,h,c,csha,args.bundle_sha256,asha)
    with Path(LOCK).open('r+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(os.fstat(lease.fileno()).st_ino==LOCK_INODE,'Original lock inode changed')
        bounds=check_allowance(a,h,c,csha,args.bundle_sha256,asha)
        started=time.monotonic();hard_end=started+bounds['remaining_total_seconds'];work_end=hard_end-60
        claims=Path('/home/abmoses2000/.graybox_gpu_review_authorizations');claims.mkdir(exist_ok=True)
        with (claims/(asha+'.json')).open('x') as stream:
            json.dump({'status':'consumed_before_preflight','authorization_sha256':asha,
                       'utc':datetime.now(timezone.utc).isoformat(),'automatic_retry':False},stream)
        output=Path(ROOT)/'execution';output.mkdir(exist_ok=False)
        receipt={'status':'inference_started_unverified','started_utc':datetime.now(timezone.utc).isoformat(),
                 'contract_sha256':csha,'bundle_sha256':args.bundle_sha256,'authorization_sha256':asha,
                 'handoff_sha256':hsha,'handoff_utc':h['utc'],'max_slot_seconds':a['max_slot_seconds'],
                 'charged_before_launch_seconds':bounds['charged_before_launch_seconds'],
                 'worker_launch_state':'not_attempted','worker_pid':None,'cleanup_verified':False,
                 'common_lock_inode':LOCK_INODE,'fitting_steps':0,'scientific_acceptance':False,'automatic_retry':False}
        write_json(output/'execution_receipt.json',receipt)
        def check(deadline=work_end):guard(c,output,args.authorization,asha,args.handoff,hsha,deadline)
        old={sig:signal.signal(sig,interrupted) for sig in [signal.SIGALRM,signal.SIGTERM,signal.SIGINT]}
        signal.setitimer(signal.ITIMER_REAL,max(.001,work_end-time.monotonic()))
        try:
            verify_bundle(bundle,args.bundle_sha256);verify_inputs(c);check();require(not gpu_pids(),'GPU occupied')
            require(not Path(c['scratch_root']).exists(),'Prior scratch must be preserved, no automatic retry')
            context={'bundle_root':str(bundle),'output':str(output),'work_end':work_end,
                     'authorization':args.authorization,'authorization_sha256':asha,'handoff':args.handoff,'handoff_sha256':hsha}
            env=dict(os.environ,PYTHONPATH=str(bundle),PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                     OMP_NUM_THREADS='1',GRAYBOX_INFERENCE_LEASE_FD=str(lease.fileno()),GRAYBOX_INFERENCE_CONTEXT=json.dumps(context))
            with (output/'execution.log').open('x') as log:
                supervise([sys.executable,'-u','-m','scripts.aia72_inference_worker'],env,bundle,log,work_end,hard_end,
                          (lease.fileno(),),check,receipt=receipt)
            signal.setitimer(signal.ITIMER_REAL,max(.001,hard_end-time.monotonic()-10))
            require_cleanup_proof(receipt);require_clean_worker(receipt);check(hard_end-10)
            verify_bundle(bundle,args.bundle_sha256);verify_inputs(c)
            result_sha=verify_result(output)
            receipt.update(status='all_inference_blocks_verified_pending_collection',result_sha256=result_sha,
                           total_analysis_complete=False)
        except BaseException as error:
            receipt.update(status='inference_incomplete_or_failed',error=repr(error))
        finally:
            signal.setitimer(signal.ITIMER_REAL,max(.001,hard_end-time.monotonic()-5))
            receipt['finished_utc']=datetime.now(timezone.utc).isoformat()
            receipt['charged_reservation_seconds_before_return']=bounds['charged_before_launch_seconds']+time.monotonic()-started
            publish_receipt(output/'execution_receipt.json',receipt,lambda:check(hard_end-5))
            signal.setitimer(signal.ITIMER_REAL,max(.001,hard_end-time.monotonic()))
            try:
                require(sha(args.authorization)==asha and sha(args.handoff)==hsha,'Control bytes changed')
                returned=return_resource(c,receipt,output,hsha,hard_end)
                charged=bounds['charged_before_launch_seconds']+time.monotonic()-started
                write_json(Path(ROOT)/'slot_ledger.json',{'status':'reservation_returned','authorization_sha256':asha,
                    'charged_total_reserved_seconds':charged,'maximum_slot_seconds':a['max_slot_seconds'],
                    'remaining_allowance_seconds':0,'invocation_consumed':True,'return_receipt':returned,
                    'overrun_seconds':max(0,charged-a['max_slot_seconds']),'automatic_retry':False})
            except BaseException as error:
                write_json(Path(ROOT)/'return_requires_review.json',{'status':'return_unverified','error':repr(error),
                                                                  'authorization_sha256':asha,'automatic_retry':False})
                receipt['return_error']=repr(error)
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,handler in old.items():signal.signal(sig,handler)
            print(json.dumps(receipt,indent=2),flush=True)
        if receipt['status']!='all_inference_blocks_verified_pending_collection' or receipt.get('return_error'):
            raise SystemExit(1)


if __name__=='__main__':main()
