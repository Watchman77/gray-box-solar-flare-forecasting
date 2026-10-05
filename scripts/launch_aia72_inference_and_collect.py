"""Execute one frozen issuer, then replace this process with its exact Mac relay.

No automatic retry. An ambiguous issuance outcome remains recorded for review.
This driver is separately frozen beside the reviewed issuer before actual use.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(package):
    package=Path(package).resolve();issuer=package/'issuance';protocol=json.loads((issuer/'protocol.json').read_text())
    source=(issuer/'issue_and_launch_bound.py').read_bytes()
    require(hashlib.sha256(source).hexdigest()==protocol['bound_source_sha256'],'Issuer source changed')
    bundle=package/'bundle'
    require(hashlib.sha256((bundle/'replay_bundle_manifest.json').read_bytes()).hexdigest()==protocol['bundle_sha256'],
            'Local bundle changed')
    for name,digest in json.loads((bundle/'replay_bundle_manifest.json').read_text())['files'].items():
        require(hashlib.sha256((bundle/name).read_bytes()).hexdigest()==digest,'Local reviewed source changed')
    require(not (package/'collected_blocks').exists(),'Prior relay archive preserved; no retry')
    argv=['/Users/mac/google-cloud-sdk/bin/gcloud','compute','ssh','abmoses2000@aia-temporal-l4',
          '--project=sonorous-shore-450510-i4','--zone=europe-west4-c','--quiet','--ssh-flag=-T',
          '--command=env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /usr/bin/timeout --signal=TERM --kill-after=10s 150s /home/abmoses2000/aia_gpu_venv/bin/python -']
    started={'utc':datetime.now(timezone.utc).isoformat(),'argv':argv,'stdin_sha256':protocol['bound_source_sha256'],
             'automatic_retry':False}
    save(issuer/'actual_start.json',started)
    begin=time.monotonic();returncode=None;error=None
    try:
        with (issuer/'actual_stdout.json').open('xb') as out,(issuer/'actual_stderr.txt').open('xb') as err:
            result=subprocess.run(argv,input=source,stdout=out,stderr=err,timeout=180)
            returncode=result.returncode
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        receipt={**started,'finished_utc':datetime.now(timezone.utc).isoformat(),
                 'elapsed_seconds':time.monotonic()-begin,'returncode':returncode,'error':error}
        save(issuer/'actual_local_receipt.json',receipt)
        print(json.dumps(receipt),flush=True)
    require(returncode==0,'Issuer did not complete successfully; inspect and never retry blindly')
    result=json.loads((issuer/'actual_stdout.json').read_text())
    require(result['status']=='FRESH_INFERENCE_RESERVATION_AND_LAUNCH_REQUESTED','Unexpected issuer status')
    require(result['automatic_retry'] is False and result['launch_receipt']['controller_pid']>1,'Launch evidence missing')
    require(len(result['handoff_sha256'])==64 and set(result['handoff_sha256'])<=set('0123456789abcdef'),'Invalid handoff digest')
    expires=datetime.fromisoformat(result['expires_utc']);require(expires.tzinfo is not None,'Missing reservation timezone')
    remaining=(expires-datetime.now(timezone.utc)).total_seconds();require(60<remaining<=14400,'No original reservation remains')
    relay=[sys.executable,'-u',str(bundle/'scripts/aia72_inference_archive_relay.py'),'--mode','local',
           '--contract-sha256',protocol['contract_sha256'],'--bundle-sha256',protocol['bundle_sha256'],
           '--handoff-sha256',result['handoff_sha256'],'--expires-utc',result['expires_utc'],
           '--package',str(package),'--output',str(package/'collected_blocks')]
    save(issuer/'local_relay_exec_intent.json',{'utc':datetime.now(timezone.utc).isoformat(),'argv':relay,
                                             'process_pid':os.getpid(),'automatic_retry':False})
    print(json.dumps({'status':'issuer_exit0_starting_exact_local_relay','controller_pid':result['launch_receipt']['controller_pid'],
                      'handoff_sha256':result['handoff_sha256'],'expires_utc':result['expires_utc']}),flush=True)
    try:os.execv(sys.executable,relay)
    except BaseException as exc:
        save(issuer/'local_relay_exec_failure.json',{'error':repr(exc),'automatic_retry':False});raise


def require(value,message):
    if not value:raise ValueError(message)


def save(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())


if __name__=='__main__':main(sys.argv[1])
