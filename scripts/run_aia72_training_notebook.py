"""Execute one bounded notebook invocation under the existing shared GPU lock."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--release', type=Path, required=True)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    release = json.loads(args.release.read_text())
    if release.get('status') != 'GPU_RELEASED_TO_GRAYBOX' or release.get('common_lock') != '/home/abmoses2000/.aia19b2_run.lock':
        raise ValueError('An explicit agreed GPU release receipt is required')
    for path in [Path('/home/abmoses2000'), Path('/mnt/disks/aia-cache')]:
        if shutil.disk_usage(path).free < 20 * 1024**3:
            raise ValueError('The shared 20-GiB disk reserve would be violated')
    bundle = json.loads((root / 'bundle_manifest.json').read_text())
    for name, digest in bundle['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError('Execution bundle changed: ' + name)
    lock_path = Path(release['common_lock'])
    # r+ intentionally refuses to create a replacement lock if the agreed file is absent.
    with lock_path.open('r+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.fstat(lease.fileno()).st_ino != lock_path.stat().st_ino:
            raise ValueError('Shared lock inode changed')
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=15)
        if occupied.strip():
            raise ValueError('GPU occupied after exclusive lock acquisition')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run = root / 'outputs' / f'invocation_{stamp}'; run.mkdir(parents=True)
        source = root / 'notebooks/09_AIA_72h_Shared_GPU_Training.ipynb'
        nb = nbformat.read(source, as_version=4); nbformat.validate(nb)
        target = run / '09_AIA_72h_Shared_GPU_Training_EXECUTED.ipynb'
        def preserve(**_):
            temporary = target.with_suffix('.tmp'); nbformat.write(nb, temporary); temporary.replace(target)
        class StreamingClient(NotebookClient):
            def process_message(self, msg, cell, cell_index):
                if msg['msg_type'] == 'stream':
                    print(msg['content'].get('text', ''), end='', flush=True)
                result = super().process_message(msg, cell, cell_index)
                if msg['msg_type'] == 'stream':
                    preserve()
                return result
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['GRAYBOX_SHARED_GPU_LEASE'] = str(root)
        os.environ['OMP_NUM_THREADS'] = '1'
        manager = KernelManager(kernel_name='python3'); manager.kernel_spec.argv[0] = sys.executable
        client = StreamingClient(nb, timeout=2400, km=manager, resources={'metadata': {'path': str(root)}})
        # The training cell has its own one-hour bound; allow checkpoint/cleanup overhead.
        client.timeout = 4200
        client.on_cell_executed = preserve
        receipt = {'status': 'running', 'pid': os.getpid(), 'utc': datetime.now(timezone.utc).isoformat(),
                   'common_lock': str(lock_path), 'lock_inode': os.fstat(lease.fileno()).st_ino,
                   'release_sha256': sha(args.release), 'bundle_sha256': sha(root / 'bundle_manifest.json')}
        (run / 'execution.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)
        try:
            client.execute()
            receipt['status'] = 'invocation_executed_consult_training_status'
        except BaseException:
            receipt['status'] = 'failed_partial_notebook_preserved'
            traceback.print_exc(); raise
        finally:
            preserve()
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
            receipt['executed_notebook_sha256'] = sha(target)
            (run / 'execution.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
