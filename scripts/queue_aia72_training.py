"""Wait once for the agreed AIA release, then run one bounded notebook invocation."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--wait-seconds', type=int, default=1800)
    args = parser.parse_args(); root = args.root.resolve()
    if not 1 <= args.wait_seconds <= 1800:
        raise ValueError('The waiting bound may not exceed thirty minutes')
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        intent = root / 'queue_intent.json'
        with intent.open('x') as stream:
            json.dump({'utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                       'queue_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       'wait_seconds': args.wait_seconds, 'automatic_retry': False}, stream, indent=2)
        status = root / 'queue_status.json'
        def record(state, **extra):
            result = {'utc': datetime.now(timezone.utc).isoformat(), 'status': state, 'pid': os.getpid(), **extra}
            temporary = status.with_suffix('.tmp'); temporary.write_text(json.dumps(result, indent=2) + '\n'); temporary.replace(status)
            print(json.dumps(result), flush=True)
        release = root / 'handoff_release.json'
        deadline = time.monotonic() + args.wait_seconds
        record('waiting_for_AIA_release_no_GPU_use')
        while not release.is_file():
            if (root / 'CANCEL_QUEUE').exists():
                record('cancelled_before_start'); return
            if time.monotonic() >= deadline:
                record('release_wait_expired_no_GPU_launch'); return
            time.sleep(min(20, max(0, deadline - time.monotonic())))
        # The notebook runner independently validates the release, shared lock,
        # absence of GPU processes, disk reserves and exact bundle hashes.
        record('starting_shared_GPU_guarded_notebook')
        with (root / 'notebook_execution.log').open('x') as log:
            result = subprocess.run([sys.executable, '-u', str(root / 'scripts/run_aia72_training_notebook.py'),
                                     '--release', str(release)], cwd=root, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
        record('notebook_invocation_exited' if result.returncode == 0 else 'notebook_failed_no_retry',
               returncode=result.returncode, full_training_complete=False,
               next='Read outputs/training/invocation_result.json and execution receipt; coordinate before any further GPU launch')


if __name__ == '__main__':
    main()
