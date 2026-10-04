"""Stage pinned replay bytes and run one stdlib CPU canary; never launch a model.

The frozen local launcher fills PAYLOAD with the accepted source tar. Execute with
CUDA hidden and GNU timeout 180s, kill-after 10s. No nested unit tests run on VM.
"""
import base64
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

PAYLOAD = None
ROOT = Path('/home/abmoses2000/graybox_aia72_completion_replay_v1_20261004')
TRAINING = Path('/home/abmoses2000/graybox_aia72_completion_training_v5_20261004')
ORIGINAL = Path('/home/abmoses2000/graybox_aia72_workers4_20261002')
LOCK = Path('/home/abmoses2000/.aia19b2_run.lock')
OWNER_SHA = 'b74c994c3102deef05d8e1c545bd87159ffd53ab829bf30de353543f24c7b9d9'
TAR_SHA = '633b0296db6bad4edebe7cf533a118e9163feb80ed5eca314f27baebd014b9d5'
MANIFEST_SHA = '4b8c1dd072aa24c4e4ca5b5c588bc59c7b581c7041c60c089174c0264589f5ba'
CONTRACT_SHA = '92dcaf0f14052f6c7d26b5ed9b3a6bec7a60317389798e048672f3251c0af41e'
CANARY = """import fcntl,json,os,sys
fd=int(sys.argv[1])
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.fstat(fd).st_ino==519274
fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert 'torch' not in sys.modules
print(json.dumps({'inherited_lock_inode':519274,'CUDA_hidden':True,'torch_imported':False}),flush=True)
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interrupted(number, _frame):
    raise InterruptedError('CPU staging interrupted by signal ' + str(number))


def check_canary(receipt, require_cleanup_proof):
    require_cleanup_proof(receipt)
    require(receipt.get('worker_launch_state') == 'started' and receipt.get('returncode') == 0,
            'CPU canary did not complete')
    require(not receipt.get('cancellation_signals') and not receipt.get('work_deadline_reached')
            and receipt.get('guard_error') is None and receipt.get('supervisor_error') is None,
            'CPU canary cancelled, timed out, or failed')


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden before Python')
    require(sys.executable == '/home/abmoses2000/aia_gpu_venv/bin/python', 'Unexpected interpreter')
    require(PAYLOAD is not None, 'No frozen payload supplied')
    started = time.monotonic(); deadline = started+120; cleanup_end = started+150
    for sig in [signal.SIGALRM,signal.SIGINT,signal.SIGTERM]:
        signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL,120)
    raw = base64.b64decode(PAYLOAD, validate=True)
    require(hashlib.sha256(raw).hexdigest() == TAR_SHA, 'Source archive differs')
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        for member in archive.getmembers():
            p = Path(member.name)
            require(member.isfile() and not p.is_absolute() and '..' not in p.parts
                    and p.parts[0]=='bundle' and member.name not in files, 'Unsafe source member')
            files[member.name] = archive.extractfile(member).read()
    manifest_bytes = files['bundle/replay_bundle_manifest.json']
    require(hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA, 'Manifest differs')
    manifest = json.loads(manifest_bytes)['files']
    require(set(files)=={'bundle/'+name for name in manifest}|{'bundle/replay_bundle_manifest.json'},
            'Source member set differs')
    for name,digest in manifest.items():
        require(hashlib.sha256(files['bundle/'+name]).hexdigest()==digest, 'Source member differs: '+name)
    config_bytes = files['bundle/configs/aia72_completion_replay_v1.json']
    require(hashlib.sha256(config_bytes).hexdigest()==CONTRACT_SHA, 'Contract differs')
    config = json.loads(config_bytes)
    require(not ROOT.exists(), 'Preserve an earlier staging attempt; no automatic retry')
    owner = ORIGINAL/'resource_reservation_status.json'
    with LOCK.open('r+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(os.fstat(lease.fileno()).st_ino==519274,'Shared inode differs')
        def guard():
            require(time.monotonic()<deadline,'CPU staging deadline exceeded')
            require(os.fstat(lease.fileno()).st_ino==LOCK.stat().st_ino==519274,'Shared inode changed')
            require(sha(owner)==sha(TRAINING/'handoff_return_to_aia.json')==OWNER_SHA,'Owner/return changed')
            for directory in [ORIGINAL,ORIGINAL/'outputs/training',TRAINING,TRAINING/'training',
                              TRAINING/'execution',ROOT,ROOT/'bundle',ROOT/'inputs']:
                if directory.is_dir():
                    require(not any(p.name.casefold().startswith(('stop','.stop','cancel','.cancel','failure','failed'))
                                    for p in directory.iterdir()),'STOP/failure marker in '+str(directory))
                    for name in ['notebook_status.json','queue_status.json','execution_receipt.json']:
                        path=directory/name
                        if path.is_file():
                            status=str(json.loads(path.read_text()).get('status','')).lower()
                            require('fail' not in status and 'cancel' not in status,'Recorded failure: '+str(path))
            require(all(shutil.disk_usage(p).free>=20*1024**3
                        for p in ['/home/abmoses2000','/mnt/disks/aia-cache']),'20GiB reserve violated')
        def empty_gpu():
            require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                                                 text=True,timeout=8).strip(),'GPU occupied')
        guard()
        require(sha(TRAINING/'execution/execution_receipt.json')==
                'ec0db7ed550fc73c0c1355210a9f3a0a21d3b8de4c4e466a8da8b7384cdc1f43','Training receipt differs')
        table=subprocess.check_output(['ps','-eo','pid,pgid,stat'],text=True,timeout=3)
        require(not any(int(p[0]) in {1699559,1699562} or int(p[1]) in {1699559,1699562}
                        for line in table.splitlines()[1:] if len(p:=line.split())>=3
                        and not p[2].startswith('Z')),'Prior training group remains live')
        empty_gpu();ROOT.mkdir(exist_ok=False)
        canary={'worker_launch_state':'not_attempted','worker_pid':None,'cleanup_verified':False}
        record={'status':'CPU_staging_started_unverified','owner_unchanged_sha256':OWNER_SHA,
                'common_lock_inode':519274,'source_tar_sha256':TAR_SHA,'bundle_sha256':MANIFEST_SHA,
                'contract_sha256':CONTRACT_SHA,'model_runs':0,'image_reads':0,'GPU_launches':0,
                'new_allowance':False,'automatic_retry':False,'CPU_canary':canary}
        try:
            for name,data in files.items():
                guard();target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream:stream.write(data)
            inputs=ROOT/'inputs';inputs.mkdir()
            for key,entry in config['inputs'].items():
                guard();source=Path(config['input_sources_on_VM'][key]);target=inputs/entry['path']
                require(source.is_file() and not source.is_symlink() and sha(source)==entry['sha256'],
                        'Existing VM input differs: '+key)
                require(target.parent==inputs,'Input destination differs')
                with source.open('rb') as src,target.open('xb') as dest:shutil.copyfileobj(src,dest)
                require(sha(target)==entry['sha256'],'Copied input differs: '+key)
            sys.path.insert(0,str(ROOT/'bundle'))
            from scripts import aia72_replay_supervisor as supervisor
            require(Path(supervisor.__file__).resolve().is_relative_to((ROOT/'bundle').resolve()),
                    'Supervisor imported from another source')
            with (ROOT/'CPU_canary.log').open('x') as log:
                supervisor.supervise([sys.executable,'-c',CANARY,str(lease.fileno())],os.environ.copy(),
                    ROOT/'bundle',log,min(deadline,time.monotonic()+10),min(cleanup_end,time.monotonic()+30),
                    (lease.fileno(),),guard,receipt=canary)
            signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
            check_canary(canary,supervisor.require_cleanup_proof)
            require(not supervisor.inspect_group(canary['worker_pid'],deadline),'CPU canary group remains')
            require(json.loads((ROOT/'CPU_canary.log').read_text())==
                    {'inherited_lock_inode':519274,'CUDA_hidden':True,'torch_imported':False},'Canary output differs')
            require('torch' not in sys.modules,'CPU preparation imported Torch')
            for name,digest in manifest.items():require(sha(ROOT/'bundle'/name)==digest,'Staged source changed')
            for entry in config['inputs'].values():require(sha(inputs/entry['path'])==entry['sha256'],'Staged input changed')
            empty_gpu();guard()
            record.update(status='CPU_staging_and_Linux_canary_passed',copied_inputs=19,source_files=len(files),
                          canary_log_sha256=sha(ROOT/'CPU_canary.log'),torch_imported=False)
        except BaseException as error:
            record.update(status='CPU_staging_or_canary_failed',error=repr(error))
            raise
        finally:
            record['elapsed_seconds']=time.monotonic()-started
            with (ROOT/'cpu_staging_receipt.json').open('x') as stream:stream.write(json.dumps(record,indent=2)+'\n')
            print(json.dumps(record,indent=2),flush=True)
            signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    main()
