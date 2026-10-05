"""Stage an exact reviewed metadata/model/source bundle, CPU only and no launch."""
import base64
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import signal
import subprocess
import sys
import tarfile
import time


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(c):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA must be hidden during staging')
    require(sys.executable=='/home/abmoses2000/aia_gpu_venv/bin/python','Frozen VM interpreter required')
    root=Path('/home/abmoses2000/graybox_aia72_inference_v1_20261004')
    owner=Path('/home/abmoses2000/graybox_aia72_workers4_20261002/resource_reservation_status.json')
    lock=Path('/home/abmoses2000/.aia19b2_run.lock');started=time.monotonic();deadline=started+180
    def interrupted(number,frame):raise InterruptedError('Staging interrupted: '+str(number))
    for sig in [signal.SIGALRM,signal.SIGTERM,signal.SIGINT]:signal.signal(sig,interrupted)
    signal.setitimer(signal.ITIMER_REAL,180)
    raw=base64.b64decode(c['archive_base64'],validate=True)
    require(hashlib.sha256(raw).hexdigest()==c['archive_sha256'] and len(raw)<40*1024**2,'Staging archive changed/oversized')
    manifest=c['manifest'];require(sum(r['bytes'] for r in manifest.values())<200*1024**2,'Expanded bundle exceeds bound')
    with lock.open('r+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def guard():
            require(time.monotonic()<deadline,'Staging deadline')
            require(os.fstat(lease.fileno()).st_ino==lock.stat().st_ino==519274,'Original lock changed')
            require(sha(owner)==c['owner_sha256'],'Owner changed before staging')
            require(all(shutil.disk_usage(p).free>=20*1024**3 for p in ['/home/abmoses2000','/mnt/disks/aia-cache']),
                    'Disk reserve violated')
        guard();require(not root.exists(),'Preserve prior staging attempt; no automatic retry')
        require(shutil.disk_usage(root.parent).free-sum(r['bytes'] for r in manifest.values())>=20*1024**3,
                'Expanded staging would violate boot reserve')
        require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                                             text=True,timeout=8).strip(),'GPU occupied')
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
            members=archive.getmembers()
            require(len(members)==len(manifest) and {m.name for m in members}==set(manifest),'Staging member set differs')
            for m in members:
                p=PurePosixPath(m.name)
                require(m.isfile() and not p.is_absolute() and '..' not in p.parts and p.parts[0] in ['bundle','inputs'],
                        'Unsafe bundle member')
                require(m.size==manifest[m.name]['bytes'],'Expanded member size differs')
            root.mkdir()
            for m in members:
                guard();payload=archive.extractfile(m).read()
                require(hashlib.sha256(payload).hexdigest()==manifest[m.name]['sha256'],'Staged member hash differs')
                path=root/m.name;path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as stream:stream.write(payload)
                if path.suffix=='.py':compile(payload,str(path),'exec')
        for name,record in manifest.items():guard();require(sha(root/name)==record['sha256'],'Written staging content differs')
        guard();require('torch' not in sys.modules,'CPU staging imported Torch')
        result={'status':'CPU_metadata_models_and_source_staged_not_launched','elapsed_seconds':time.monotonic()-started,
                'archive_sha256':c['archive_sha256'],'contract_sha256':sha(root/'bundle/configs/aia72_inference_v1.json'),
                'bundle_sha256':sha(root/'bundle/replay_bundle_manifest.json'),'files':len(manifest),
                'owner_sha256':c['owner_sha256'],'common_lock_inode':519274,'image_downloads':0,'GPU_launches':0,
                'torch_imported':False,'existing_sources_modified':False,'automatic_retry':False}
        with (root/'cpu_staging_receipt.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    signal.setitimer(signal.ITIMER_REAL,0)
