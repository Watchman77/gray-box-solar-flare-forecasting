"""Read-only, bounded snapshot of the completed replay; streams gzip to stdout.

No VM files are created or modified. Requires the exact successful final receipts,
unchanged source/input bytes, the original exclusive lock and an idle GPU.
"""
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time

ROOT = Path('/home/abmoses2000/graybox_aia72_completion_replay_v1_20261004')
OWNER = Path('/home/abmoses2000/graybox_aia72_workers4_20261002/resource_reservation_status.json')
LOCK = Path('/home/abmoses2000/.aia19b2_run.lock')
RETURN = 'd0ee440ed05554281ab1b5730ae6ce1fe0270c91f560ffba69afb848dfc56afe'
EXPECTED = {
    'execution/execution_receipt.json':'1a837a62f8d70c3bdf435461d7064eeab550554464f11a530a1093392dcbc65f',
    'execution/replay/result.json':'c56e23141506a4194b195976e8a1d28233c4f461b58dcd2e815b122e7b3909ef',
    'handoff_return_to_aia.json':RETURN,
    'bundle/replay_bundle_manifest.json':'4b8c1dd072aa24c4e4ca5b5c588bc59c7b581c7041c60c089174c0264589f5ba',
    'bundle/configs/aia72_completion_replay_v1.json':'92dcaf0f14052f6c7d26b5ed9b3a6bec7a60317389798e048672f3251c0af41e',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Hide CUDA before collection')
    require(sys.executable=='/home/abmoses2000/aia_gpu_venv/bin/python','Exact interpreter required')
    started=time.monotonic();deadline=started+60
    def interrupted(number, frame):
        raise InterruptedError('Read-only collection interrupted: '+str(number))
    for sig in [signal.SIGALRM,signal.SIGTERM,signal.SIGINT]:signal.signal(sig,interrupted)
    signal.setitimer(signal.ITIMER_REAL,60)
    with LOCK.open('r+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def guard():
            require(time.monotonic()<deadline,'Collection deadline exceeded')
            require(os.fstat(lease.fileno()).st_ino==LOCK.stat().st_ino==519274,'Shared inode changed')
            require(sha(OWNER)==sha(ROOT/'handoff_return_to_aia.json')==RETURN,'Final owner/return changed')
            require(all(shutil.disk_usage(p).free>=20*1024**3
                        for p in ['/home/abmoses2000','/mnt/disks/aia-cache']),'Disk reserve violated')
        guard()
        require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                                             text=True,timeout=8).strip(),'GPU occupied')
        table=subprocess.check_output(['ps','-eo','pid,pgid,stat'],text=True,timeout=3)
        require(not any(int(p[0]) in {1705207,1705210} or int(p[1]) in {1705207,1705210}
                        for line in table.splitlines()[1:] if len(p:=line.split())>=3
                        and not p[2].startswith('Z')),'Completed replay descendants still live')
        for name,digest in EXPECTED.items():require(sha(ROOT/name)==digest,'Final pin differs: '+name)
        receipt=json.loads((ROOT/'execution/execution_receipt.json').read_text())
        require(receipt['status']=='technical_checkpoint_replay_verified' and receipt['returncode']==0
                and receipt['cleanup_verified'] and receipt['worker_reaped'] and receipt['process_group_empty']
                and receipt['cleanup_errors']==[] and not receipt['cancellation_signals']
                and not receipt['work_deadline_reached'] and receipt['guard_error'] is None,'Replay not cleanly complete')
        result=json.loads((ROOT/'execution/replay/result.json').read_text())
        require(receipt['result_sha256']==EXPECTED['execution/replay/result.json']
                and result['replayed_seeds']==[29,43] and result['seed17_repeated'] is False
                and result['fitting_steps']==0 and result['inputs_unchanged'],'Result scope differs')
        ledger=json.loads((ROOT/'slot_ledger.json').read_text())
        require(ledger['status']=='reservation_returned' and ledger['overrun_seconds']==0
                and ledger['remaining_allowance_seconds']==0
                and ledger['return_receipt']['sha256']==RETURN,'Final ledger differs')
        config=json.loads((ROOT/'bundle/configs/aia72_completion_replay_v1.json').read_text())
        expected=dict(EXPECTED)
        expected.update({'bundle/'+name:digest for name,digest in
                         json.loads((ROOT/'bundle/replay_bundle_manifest.json').read_text())['files'].items()})
        expected.update({'inputs/'+entry['path']:entry['sha256'] for entry in config['inputs'].values()})
        expected.update({'execution/replay/'+name:digest for name,digest in result['output_sha256'].items()})
        paths=[]
        for p in ROOT.rglob('*'):
            require(not p.is_symlink(),'Preserve and reject symlink: '+str(p))
            if p.is_dir():continue
            require(stat.S_ISREG(p.stat().st_mode) and p.resolve().is_relative_to(ROOT),'Nonregular snapshot member')
            require(not p.name.casefold().startswith(('stop','.stop','cancel','.cancel','failure','failed')),
                    'STOP/failure marker requires review')
            paths.append(p)
        require(len(paths)<=120 and sum(p.stat().st_size for p in paths)<=64*1024**2,'Unexpected snapshot size')
        data={}
        for path in sorted(paths):
            guard();raw=path.read_bytes();name=path.relative_to(ROOT).as_posix()
            if name in expected:require(hashlib.sha256(raw).hexdigest()==expected[name],'Member changed: '+name)
            data[name]=raw
        require(set(expected)<=set(data),'Required snapshot members missing')
        for name,digest in expected.items():require(sha(ROOT/name)==digest,'Live member changed during collection')
        guard()
        manifest={name:{'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)} for name,raw in data.items()}
        summary={'status':'completed_replay_read_only_snapshot','VM_files_written':0,'model_runs':0,
                 'Torch_imported':'torch' in sys.modules,'source_files':len(data),'common_lock_inode':519274,
                 'current_owner_sha256':RETURN,'elapsed_verification_seconds':time.monotonic()-started}
        data['snapshot_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
        data['snapshot_collection.json']=(json.dumps(summary,indent=2)+'\n').encode()
        with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
            for name,raw in data.items():
                guard();member=tarfile.TarInfo(name);member.size=len(raw);archive.addfile(member,io.BytesIO(raw))
        guard()
    signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    main()
