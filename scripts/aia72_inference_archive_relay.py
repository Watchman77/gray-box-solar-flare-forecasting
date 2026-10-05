"""Bounded two-host block-result relay; never transfers/deletes image pixels."""
import argparse
import base64
import csv
from datetime import datetime,timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import select
import shlex
import signal
import subprocess
import sys
import tarfile
import time

ROOT=Path('/home/abmoses2000/graybox_aia72_inference_v1_20261004')
OWNER=Path('/home/abmoses2000/graybox_aia72_workers4_20261002/resource_reservation_status.json')


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def durable(path,raw):
    path=Path(path)
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    descriptor=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def encoded(value):
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def operation_deadlines(expires_utc):
    expires=datetime.fromisoformat(expires_utc.replace('Z','+00:00'))
    require(expires.tzinfo is not None,'Timezone required for immutable relay deadline')
    remaining=(expires-datetime.now(timezone.utc)).total_seconds()
    require(60<remaining<=14400,'Relay outside original reservation')
    hard_end=time.monotonic()+remaining
    return hard_end-60,hard_end


def write_ack(fd,payload,deadline,cancellations):
    os.set_blocking(fd,False);offset=0
    while offset<len(payload):
        require(not cancellations and time.monotonic()<deadline,'Acknowledgement cancelled or deadline reached')
        if not select.select([],[fd],[],min(.2,max(0,deadline-time.monotonic())))[1]:continue
        try:offset+=os.write(fd,payload[offset:])
        except BlockingIOError:continue


def remote(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Relay is CPU-only')
    require(sha(ROOT/'bundle/configs/aia72_inference_v1.json')==args.contract_sha256,'Contract changed')
    require(sha(ROOT/'bundle/replay_bundle_manifest.json')==args.bundle_sha256,'Bundle changed')
    manifest=json.loads((ROOT/'bundle/replay_bundle_manifest.json').read_text())['files']
    require(sha(Path(__file__))==manifest['scripts/aia72_inference_archive_relay.py'],'Relay source changed')
    c=json.loads((ROOT/'bundle/configs/aia72_inference_v1.json').read_text())
    blocks=json.loads((ROOT/'inputs/blocks.json').read_text())
    require(sha(ROOT/'inputs/blocks.json')==c['inputs']['blocks.json']['sha256'],'Block support changed')
    schedule=[b for role in c['role_order'] for b in blocks if b['role']==role]
    output=ROOT/'execution/predictions';deadline,hard_end=operation_deadlines(args.expires_utc)
    def guard():
        require(time.monotonic()<deadline,'Relay deadline reached')
        require(sha(OWNER)==args.handoff_sha256,'Resource owner changed; stop relay')
        for name in ['STOP','STOP_REQUESTED','CANCEL','FAILURE']:
            require(not (ROOT/name).exists() and not (output/name).exists(),'Stop marker present')
    def interrupted(number,frame):raise InterruptedError('Relay interrupted: '+str(number))
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(sig,interrupted)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    for block in schedule:
        name=block['block_id'];ready_path=output/(name+'_ready.json')
        while not ready_path.exists():guard();time.sleep(.2)
        guard();ready=json.loads(ready_path.read_text());require(ready['block_id']==name,'Block identity differs')
        members={name+'_predictions.csv.gz':ready['predictions_sha256'],name+'_sources.json':ready['sources_sha256']}
        payloads={}
        for member,digest in members.items():
            path=output/member;require(path.is_file() and not path.is_symlink() and path.stat().st_size<8*1024**2,'Unexpected result member')
            payload=path.read_bytes();require(hashlib.sha256(payload).hexdigest()==digest,'Result member changed')
            payloads[member]=payload
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            for member,payload in payloads.items():
                info=tarfile.TarInfo(member);info.size=len(payload);archive.addfile(info,io.BytesIO(payload))
        raw=buffer.getvalue();require(len(raw)<4*1024**2,'Block archive exceeds bound')
        proofs=output/'proofs';proofs.mkdir(exist_ok=True)
        durable(proofs/(name+'.tar.gz'),raw)
        archive_sha=hashlib.sha256(raw).hexdigest()
        print(json.dumps({'block_id':name,'archive_sha256':archive_sha,'archive_base64':base64.b64encode(raw).decode()}),flush=True)
        ack_deadline=min(deadline,time.monotonic()+180)
        while not select.select([sys.stdin],[],[],.2)[0]:
            guard();require(time.monotonic()<ack_deadline,'Local acknowledgement timeout; preserve scratch')
        line=sys.stdin.readline(65537);require(line and len(line)<65537,'Missing/bounded acknowledgement required')
        ack=json.loads(line)
        require(ack['status']=='local_block_archive_verified' and ack['block_id']==name
                and ack['archive_sha256']==archive_sha and ack['predictions_sha256']==ready['predictions_sha256']
                and ack['sources_sha256']==ready['sources_sha256'],'Local acknowledgement differs')
        guard();temporary=output/(name+'_local_ack.json.tmp');durable(temporary,encoded(ack))
        temporary.replace(output/(name+'_local_ack.json'))
    signal.setitimer(signal.ITIMER_REAL,0)


def verify_archive(raw,message,block,pins,case_rows):
    require(hashlib.sha256(raw).hexdigest()==message['archive_sha256'],'Transferred archive hash differs')
    name=block['block_id'];require(message['block_id']==name,'Block sequence differs')
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        members=archive.getmembers()
        require([m.name for m in members]==[name+'_predictions.csv.gz',name+'_sources.json'],'Unexpected archive members')
        require(all(m.isfile() and m.size<8*1024**2 for m in members),'Unsafe archive entry')
        payloads={m.name:archive.extractfile(m).read() for m in members}
    sources=json.loads(payloads[name+'_sources.json']);require(set(sources)==set(block['uris']),'Block image support differs')
    for uri,r in sources.items():
        require(all(r[k]==pins[uri][k] for k in ['uri','generation','bytes','md5_base64']),'Block source pin differs')
        require(r['status'] in ['ok','missing_input','invalid_input'],'Unexpected source status')
        if r.get('owned'):
            p=Path(r['path'])
            require(p.parent==Path('/mnt/disks/aia-cache/graybox_aia72_inference_v1_20261004')/name
                    and len(r['sha256'])==64,'Fresh scratch source is not correctly bound')
    predictions=list(csv.DictReader(io.StringIO(gzip.decompress(payloads[name+'_predictions.csv.gz']).decode())))
    require([r['forecast_case_id'] for r in predictions]==block['case_ids'],'Output support/order differs')
    for r in predictions:
        expected=case_rows[r['forecast_case_id']]
        require(r['role']==expected['role'] and float(r['label'])==float(expected['label'])
                and r['issue_utc']==expected['issue_utc'] and r['last_observation_utc']==expected['history_96_UTC'],
                'Case metadata changed')
        status=r['input_status'];require(status in ['ok','missing_input','invalid_input'],'Prediction status differs')
        columns=['probability',*[f'logit_seed_{s}' for s in [17,29,43]],*[f'probability_seed_{s}' for s in [17,29,43]]]
        if status!='ok':
            require(all(r[k]=='' for k in columns) and r['input_failure_reason'],'Failed input has a prediction or lacks explanation')
        else:
            values=[float(r[f'probability_seed_{s}']) for s in [17,29,43]]
            require(all(math.isfinite(v) and 0<=v<=1 for v in values),'Invalid seed probability')
            p=float(r['probability']);require(math.isfinite(p) and abs(p-sum(values)/3)<1e-12,'Wrong ensemble mean')
            for s,p in zip([17,29,43],values):
                v=float(r[f'logit_seed_{s}']);require(math.isfinite(v),'Invalid logit')
                expected_p=1/(1+math.exp(-v)) if v>=0 else math.exp(v)/(1+math.exp(v))
                require(abs(p-expected_p)<1e-12,'Logit/probability mismatch')
    return {'status':'local_block_archive_verified','block_id':name,'archive_sha256':message['archive_sha256'],
            'predictions_sha256':hashlib.sha256(payloads[name+'_predictions.csv.gz']).hexdigest(),
            'sources_sha256':hashlib.sha256(payloads[name+'_sources.json']).hexdigest(),
            'cases':len(predictions),'source_objects':len(sources),'scientific_acceptance':False}


def local(args):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from scripts.aia72_replay_supervisor import stop_group
    deadline,cleanup_end=operation_deadlines(args.expires_utc)
    cancellations=[]
    old_handlers={sig:signal.signal(sig,lambda number,_frame:cancellations.append(number))
                  for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]}
    package=Path(args.package).resolve();output=Path(args.output).resolve();output.mkdir(exist_ok=False)
    config=package/'bundle/configs/aia72_inference_v1.json'
    require(sha(config)==args.contract_sha256,'Local contract changed')
    require(sha(package/'bundle/replay_bundle_manifest.json')==args.bundle_sha256,'Local bundle changed')
    c=json.loads(config.read_text())
    for r in c['inputs'].values():require(sha(package/'inputs'/r['path'])==r['sha256'],'Local frozen input changed')
    blocks=json.loads((package/'inputs/blocks.json').read_text())
    schedule=[b for role in c['role_order'] for b in blocks if b['role']==role]
    pins=json.loads((package/'inputs/objects.json').read_text())
    with gzip.open(package/'inputs/cases.csv.gz','rt') as stream:case_rows={r['forecast_case_id']:r for r in csv.DictReader(stream)}
    command=['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/usr/bin/timeout','--signal=TERM','--kill-after=10s','14430s',
             '/home/abmoses2000/aia_gpu_venv/bin/python',str(ROOT/'bundle/scripts/aia72_inference_archive_relay.py'),
             '--mode','remote','--contract-sha256',args.contract_sha256,'--bundle-sha256',args.bundle_sha256,
             '--handoff-sha256',args.handoff_sha256,'--expires-utc',args.expires_utc]
    argv=['/Users/mac/google-cloud-sdk/bin/gcloud','compute','ssh','abmoses2000@aia-temporal-l4',
          '--project=sonorous-shore-450510-i4','--zone=europe-west4-c','--quiet','--ssh-flag=-T','--command',shlex.join(command)]
    durable(output/'start.json',encoded({'argv':argv,'automatic_retry':False,'handoff_sha256':args.handoff_sha256}))
    completed=0;buffer=b'';process=None;error=None;cleanup={'worker_launch_state':'not_attempted','worker_pid':None}
    try:
        with (output/'stderr.txt').open('xb') as err:
            process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,start_new_session=True)
            cleanup.update(worker_launch_state='started',worker_pid=process.pid)
            durable(output/'process.json',encoded({'pid':process.pid}))
            for block in schedule:
                while b'\n' not in buffer:
                    require(not cancellations and time.monotonic()<deadline,'Local relay cancelled or deadline reached')
                    if not select.select([process.stdout],[],[],.2)[0]:continue
                    data=os.read(process.stdout.fileno(),65536);require(data,'Remote relay ended before all blocks')
                    buffer+=data;require(len(buffer)<6*1024**2,'Relay message exceeds bound')
                line,buffer=buffer.split(b'\n',1);message=json.loads(line)
                raw=base64.b64decode(message['archive_base64'],validate=True)
                ack=verify_archive(raw,message,block,pins,case_rows)
                path=output/(block['block_id']+'.tar.gz');durable(path,raw)
                require(sha(path)==ack['archive_sha256'],'Local saved archive differs')
                ack['local_archive_path']=str(path);durable(output/(block['block_id']+'_ack.json'),encoded(ack))
                write_ack(process.stdin.fileno(),json.dumps(ack).encode()+b'\n',deadline,cancellations)
                completed+=1;print(json.dumps({'archived_blocks':completed,'block_id':block['block_id'],'cases':ack['cases']}),flush=True)
            process.stdin.close()
            require(not cancellations and time.monotonic()<deadline,'No relay completion time remains')
            require(process.wait(timeout=min(30,deadline-time.monotonic()))==0,'Remote relay failed after final acknowledgement')
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc);raise
    finally:
        if process is not None:
            # Includes normal leader exit: descendants must still be inspected.
            try:cleanup.update(stop_group(process,cleanup_end))
            except BaseException as exc:cleanup.update(cleanup_verified=False,cleanup_error=repr(exc))
            for pipe in [process.stdin,process.stdout]:
                try:pipe.close()
                except BaseException as exc:cleanup.setdefault('pipe_close_errors',[]).append(repr(exc))
        success=(completed==len(schedule) and error is None and not cancellations and cleanup.get('cleanup_verified')
                 and not cleanup.get('pipe_close_errors'))
        try:
            durable(output/'result.json',encoded({'status':'all_blocks_archived' if success else 'relay_incomplete',
                'blocks_archived':completed,'error':error,'returncode':None if process is None else process.returncode,
                'cleanup':cleanup,'cancellation_signals':cancellations,'automatic_retry':False}))
        finally:
            for sig,handler in old_handlers.items():signal.signal(sig,handler)
        require(success,'Block relay did not finish with verified cleanup')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['local','remote'],required=True)
    for name in ['contract-sha256','bundle-sha256','handoff-sha256','expires-utc']:p.add_argument('--'+name,required=True)
    p.add_argument('--package');p.add_argument('--output');args=p.parse_args()
    (remote if args.mode=='remote' else local)(args)
