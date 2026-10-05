"""Issue and launch one reviewed inference phase under the standing user priority.

This source is bound to the frozen prior issuer's runtime/lock/launch helpers and
the completed CPU staging receipt by a local preparation step before execution.
"""
import ast
import base64
from contextlib import ExitStack
from datetime import datetime,timedelta,timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import types


def main(c):
    require(__debug__ and not sys.flags.optimize,'Normal Python execution required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Issuer is CPU-only')
    require(sys.executable=='/home/abmoses2000/aia_gpu_venv/bin/python','Exact VM interpreter required')
    root=Path('/home/abmoses2000/graybox_aia72_inference_v1_20261004');bundle=root/'bundle';coord=root/'coordination'
    prior=Path('/home/abmoses2000/graybox_aia72_completion_replay_v1_20261004')
    owner=Path('/home/abmoses2000/graybox_aia72_workers4_20261002/resource_reservation_status.json')
    helper=base64.b64decode(c['inspection_source_base64'],validate=True)
    require(hashlib.sha256(helper).hexdigest()=='c7de4c6b153010250ee1b9f59c13d3ed2adc86a8fe1f4d78b029768ad2723d60','Lock helper changed')
    L=types.ModuleType('accepted_lock_helpers');exec(compile(helper,'accepted_lock_helpers','exec'),L.__dict__)
    sys.path.insert(0,str(bundle))
    from scripts.aia72_inference_contract import check_contract,check_allowance,RETURN
    from scripts.aia72_replay_contract import verify_bundle,verify_inputs,check_guards
    started=time.monotonic();deadline=started+120
    def interrupted(number,frame):raise InterruptedError('Issuer interrupted: '+str(number))
    for sig in [signal.SIGALRM,signal.SIGTERM,signal.SIGINT]:signal.signal(sig,interrupted)
    signal.setitimer(signal.ITIMER_REAL,120)
    require(sha(bundle/'configs/aia72_inference_v1.json')==c['contract_sha256'],'Inference contract changed')
    contract=check_contract(json.loads((bundle/'configs/aia72_inference_v1.json').read_text()))
    require(sys.version==contract['runtime']['python'],'Python runtime differs');check_runtime(contract)
    verify_bundle(bundle,c['bundle_sha256']);verify_inputs(contract)
    require(sha(root/'cpu_staging_receipt.json')==c['cpu_staging_receipt_sha256'],'CPU staging changed')
    staged=json.loads((root/'cpu_staging_receipt.json').read_text())
    require(staged['status']=='CPU_metadata_models_and_source_staged_not_launched' and staged['GPU_launches']==0
            and staged['contract_sha256']==c['contract_sha256'] and staged['bundle_sha256']==c['bundle_sha256'],
            'CPU staging receipt differs')
    with ExitStack() as stack:
        held=[]
        for name in L.LOCK_NAMES:
            path=L.HOME/name;before=L.regular(path);fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
            stack.enter_context(os.fdopen(fd,'r+'))
            require(L.identity(before)==L.identity(os.fstat(fd))==L.identity(L.regular(path))
                    and before.st_ino==L.LOCK_INODES[name],'Original lock identity changed')
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append((path,fd,L.identity(before)))
        def check():
            require(time.monotonic()<deadline,'Issuer deadline')
            require(sha(owner)==sha(prior/'handoff_return_to_aia.json')==RETURN,'Prior resource return changed')
            for path,fd,identity in held:
                require(L.identity(L.regular(path))==identity==L.identity(os.fstat(fd)),'Held lock changed')
            for source in [root/'inputs',prior,L.HOME/'graybox_aia72_workers4_20261002',
                           L.HOME/'graybox_aia72_completion_training_v5_20261004']:
                check_guards({**contract,'source_root':str(source)},root/'execution',RETURN,deadline)
            L.idle_gpu_and_workers()
        check()
        require(not coord.exists() and not (root/'execution').exists() and not Path(contract['scratch_root']).exists(),
                'Prior attempt exists; preserve and do not retry')
        require(not (root/'handoff_return_to_aia.json').exists() and not (root/'slot_ledger.json').exists(),
                'Prior invocation preserved')
        require(subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True,timeout=8).strip()
                ==contract['runtime']['device'],'GPU device changed')
        free={str(p):shutil.disk_usage(p).free for p in [L.HOME,Path('/mnt/disks/aia-cache')]}
        coord.mkdir()
        evidence={name:save(coord,name,base64.b64decode(value,validate=True)) for name,value in c['evidence_base64'].items()}
        evidence['previous_owner.json']=save(coord,'previous_owner.json',owner.read_bytes())
        now=datetime.now(timezone.utc)
        common={'kind':contract['kind'],'review_root':str(root),'contract_sha256':c['contract_sha256'],
                'bundle_sha256':c['bundle_sha256'],'fitting_allowed':False,'seeds':[17,29,43]}
        auth={**common,'status':'AUTHORIZED_GPU_INFERENCE','utc':now.isoformat(),'not_before_utc':now.isoformat(),
              'expires_utc':(now+timedelta(seconds=14400)).isoformat(),'max_slot_seconds':14400,'cleanup_seconds':60,
              'charge_from':'handoff_utc','invocation_count':1,'automatic_retry':False,
              'user_approval_reference':'Standing user direction: coordinate with AIA, reuse same VM/GPU one job at a time; Gray finishes its full agreed training/analysis before AIA resumes.',
              'numeric_duration_specified_by_user':False,'duration_interpretation':'Agent-selected four-hour safety ceiling; finish and return promptly.',
              'previous_allowances_remain_consumed':True,'acceptance_sha256':evidence,
              'prior_selection_cases_reused':3905,'new_inference_cases':92691,
              'maximum_download_bytes':contract['maximum_download_bytes'],'scratch_cap_bytes':contract['scratch_cap_bytes'],
              'Gray_total_analysis_priority_retained':True,'AIA_must_remain_Mac_only':True,'scientific_acceptance':False}
        auth_sha=save(coord,'authorization.json',auth)
        handoff={**common,'status':'GPU_RELEASED_TO_GRAYBOX_INFERENCE','utc':now.isoformat(),
                 'authorization_sha256':auth_sha,'prior_return_sha256':RETURN,
                 'common_lock':contract['common_lock'],'common_lock_inode':519274,'free_bytes':free,
                 'GPU_empty_under_common_lock':True,'total_analysis_priority_returned':False,'automatic_retry':False}
        handoff_sha=save(coord,'handoff_to_graybox_inference.json',handoff)
        check_allowance(auth,handoff,contract,c['contract_sha256'],c['bundle_sha256'],auth_sha)
        require(not (L.HOME/'.graybox_gpu_review_authorizations'/(auth_sha+'.json')).exists(),'Allowance already consumed')
        verify_bundle(bundle,c['bundle_sha256']);verify_inputs(contract);check()
        tmp=owner.with_suffix('.inference_handoff.tmp')
        with tmp.open('xb') as stream:stream.write((coord/'handoff_to_graybox_inference.json').read_bytes());stream.flush();os.fsync(stream.fileno())
        require(sha(owner)==RETURN,'Owner changed before publication');os.replace(tmp,owner)
        require(sha(owner)==handoff_sha,'Published handoff differs')
        save(coord,'handoff_completed.json',{'authorization_sha256':auth_sha,'handoff_sha256':handoff_sha,'held_original_locks':
             [{'path':str(path),'inode':os.fstat(fd).st_ino} for path,fd,_ in held]})
    require(time.monotonic()<deadline and sha(owner)==handoff_sha,'Issuance changed before launch')
    check_allowance(auth,handoff,contract,c['contract_sha256'],c['bundle_sha256'],auth_sha)
    argv=[sys.executable,'-u',str(bundle/'scripts/run_aia72_inference.py'),'--contract-sha256',c['contract_sha256'],
          '--bundle-sha256',c['bundle_sha256'],'--authorization',str(coord/'authorization.json'),
          '--handoff',str(coord/'handoff_to_graybox_inference.json')]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('CUDA_VISIBLE_DEVICES',None)
    launched=attempt_launch(argv,env,root,coord)
    result={'status':'FRESH_INFERENCE_RESERVATION_AND_LAUNCH_REQUESTED','authorization_sha256':auth_sha,
            'handoff_sha256':handoff_sha,'expires_utc':auth['expires_utc'],'launch_receipt':launched,
            'GPU_execution_observed':False,'automatic_retry':False}
    save(coord,'issuance_and_launch_completed.json',result)
    signal.setitimer(signal.ITIMER_REAL,0);print(json.dumps(result,indent=2),flush=True)
