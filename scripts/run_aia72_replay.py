"""One approved inference-only slot, with inherited lock and bounded process cleanup."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.aia72_replay_contract import (LOCK, LOCK_INODE, REVIEW_ROOT, CONSUMED_ROOT, check_authorization, check_contract, check_guards,
                                          check_fixed_review_guards, read_hashed_json,
                                          require, sha, verify_bundle, verify_inputs, write_json)


def gpu_pids():
    value = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                    text=True, timeout=8)
    return [int(x.strip()) for x in value.splitlines() if x.strip()]


def live_group(pgid):
    table = subprocess.check_output(['ps', '-eo', 'pid,pgid,stat'], text=True, timeout=3)
    return [int(parts[0]) for line in table.splitlines()[1:] if len(parts := line.split()) >= 3
            and int(parts[1]) == pgid and not parts[2].startswith('Z')]


def stop_group(process, end):
    """Kill only this launcher's new process group, including spawned loader workers."""
    require(process.pid != os.getpgrp(), 'Refusing to signal controller process group')
    for sig, wait_seconds in [(signal.SIGTERM, 3), (signal.SIGKILL, 3)]:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        until = min(end, time.monotonic() + wait_seconds)
        while time.monotonic() < until:
            process.poll()
            try:
                if not live_group(process.pid):
                    return True
            except Exception:
                # Inspection failure must not skip the last-resort signal or imply clean release.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=max(.001, min(2, end-time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
                return False
            time.sleep(.1)
    return not live_group(process.pid)


def supervise(command, environment, cwd, log, work_end, slot_end, pass_fds=(), guard=None):
    # During the worker lifetime signals request cancellation; they never interrupt cleanup.
    watched = [signal.SIGALRM, signal.SIGINT, signal.SIGTERM]
    blocked = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
    arrived = []
    handlers = {sig: signal.signal(sig, lambda number, _frame: arrived.append(number)) for sig in watched}
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
    process = None
    timed_out, clean, guard_error = False, True, None
    try:
        require(time.monotonic() < work_end and not arrived, 'Slot cancelled before worker launch')
        process = subprocess.Popen(command, env=environment, cwd=cwd, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=pass_fds)
        try:
            while process.poll() is None and not arrived:
                remaining = work_end - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                if guard:
                    try:
                        guard()
                    except Exception as error:
                        guard_error = type(error).__name__ + ': ' + str(error)
                        break
                try:
                    process.wait(timeout=min(.1, remaining))
                except subprocess.TimeoutExpired:
                    pass
            timed_out = timed_out or time.monotonic() >= work_end
        finally:
            clean = stop_group(process, slot_end)
    finally:
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
    return {'worker_pid': process.pid, 'returncode': process.poll(), 'work_deadline_reached': timed_out,
            'process_group_empty': clean, 'cancellation_signals': arrived, 'guard_error': guard_error}


def require_clean_worker(result):
    require(result['process_group_empty'] and not gpu_pids(), 'Cleanup incomplete; do not release reservation')
    require(not result['work_deadline_reached'] and not result['cancellation_signals']
            and result.get('guard_error') is None and result['returncode'] == 0,
            'Replay incomplete, cancelled or failed')


def check_launch_location(root):
    require(Path(root).resolve() == (Path(REVIEW_ROOT) / 'bundle').resolve(), 'Launch only the canonical prepared bundle')


def publish_receipt(path, receipt, guard):
    """A late STOP/owner/disk/deadline change invalidates even an already written success."""
    if receipt['status'] != 'technical_checkpoint_replay_verified':
        write_json(path, receipt)
        return
    try:
        guard()
        write_json(path, receipt)
        guard()
    except BaseException as error:
        receipt['status'] = 'review_incomplete_or_failed'
        receipt['error'] = type(error).__name__ + ': ' + str(error)
        receipt['final_publication_gate_failed'] = True
        write_json(path, receipt)


def interrupted(signum, _frame):
    raise InterruptedError('Replay controller received signal ' + str(signum))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--bundle-sha256', required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--handoff', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    check_launch_location(root)
    contract_path = root / 'configs/aia72_seed17_replay_v1.json'
    contract, contract_sha = read_hashed_json(contract_path)
    require(contract_sha == args.contract_sha256, 'Unapproved replay contract')
    contract = check_contract(contract)
    authorization, authorization_sha = read_hashed_json(args.authorization)
    handoff, handoff_sha = read_hashed_json(args.handoff)
    # This small pinned receipt proves that the earlier training handoff was returned.
    prior_path = Path(contract['source_root']) / contract['inputs']['prior_return']['path']
    require(sha(prior_path) == contract['inputs']['prior_return']['sha256'], 'Prior return changed')
    check_authorization(authorization, handoff, args.contract_sha256, args.bundle_sha256,
                        authorization_sha, json.loads(prior_path.read_text()))
    with Path(LOCK).open('r+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(os.fstat(lease.fileno()).st_ino == LOCK_INODE, 'Shared lock inode changed')
        started = time.monotonic()
        slot_end = started + contract['slot_seconds']
        work_end = slot_end - contract['cleanup_reserve_seconds']
        consumed = Path(CONSUMED_ROOT)
        consumed.mkdir(exist_ok=True)
        # Even a failed preflight consumes this one-time review authorization.
        with (consumed / (authorization_sha + '.json')).open('x') as stream:
            json.dump({'utc': datetime.now(timezone.utc).isoformat(), 'automatic_retry': False,
                       'authorization_sha256': authorization_sha}, stream)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output = root.parent / 'executions' / stamp
        output.mkdir(parents=True, exist_ok=False)
        receipt = {'status': 'review_started_unverified', 'started_utc': datetime.now(timezone.utc).isoformat(),
                   'contract_sha256': args.contract_sha256, 'bundle_sha256': args.bundle_sha256,
                   'authorization_sha256': authorization_sha, 'handoff_sha256': handoff_sha,
                   'slot_seconds': 1800, 'cleanup_reserve_seconds': 60, 'common_lock_inode': LOCK_INODE,
                   'fitting_steps': 0, 'scientific_acceptance': False}
        write_json(output / 'execution_receipt.json', receipt)
        def guard(deadline):
            check_fixed_review_guards(contract, output, args.authorization, authorization_sha,
                                      args.handoff, handoff_sha, deadline)
        old_handlers = {sig: signal.signal(sig, interrupted) for sig in [signal.SIGALRM, signal.SIGINT, signal.SIGTERM]}
        signal.setitimer(signal.ITIMER_REAL, max(.001, work_end - time.monotonic()))
        try:
            verify_bundle(root, args.bundle_sha256)
            require(not gpu_pids(), 'GPU occupied after lock acquisition')
            guard(work_end)
            free = {p: shutil.disk_usage(p).free for p in ['/home/abmoses2000', '/mnt/disks/aia-cache']}
            require(min(free.values()) >= contract['minimum_free_bytes'], 'Shared disk reserve violated')
            verify_inputs(contract)
            require(time.monotonic() < work_end, 'Loading exhausted work budget')
            env = dict(os.environ)
            env.update({'PYTHONPATH': str(root), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '1',
                        'GRAYBOX_REPLAY_LEASE_FD': str(lease.fileno()),
                        'GRAYBOX_REPLAY_CONTEXT': json.dumps({'mode': 'authorized_gpu_replay', 'bundle_root': str(root),
                            'output': str(output), 'deadline': work_end, 'owner_sha256': handoff_sha})})
            # supervise supplies the worker deadline; SIGALRM still protects preflight and final verification.
            with (output / 'execution.log').open('x') as log:
                receipt.update(supervise([sys.executable, '-u', '-m', 'scripts.aia72_replay'], env, root, log,
                                         work_end, slot_end, (lease.fileno(),),
                                         lambda: guard(work_end)))
            signal.setitimer(signal.ITIMER_REAL, max(.001, slot_end - time.monotonic() - 5))
            require_clean_worker(receipt)
            guard(slot_end - 5)
            verify_inputs(contract)
            verify_bundle(root, args.bundle_sha256)
            result = json.loads((output / 'comparison_result.json').read_text())
            require(result['status'] == 'comparison_passed_pending_controller' and result['cases'] == 3905,
                    'No full successful comparison')
            require(time.monotonic() < slot_end, 'Overall review allowance exhausted')
            guard(slot_end - 5)
            receipt['status'] = 'technical_checkpoint_replay_verified'
            receipt['result_sha256'] = sha(output / 'comparison_result.json')
        except BaseException as error:
            receipt['status'] = 'review_incomplete_or_failed'
            receipt['error'] = type(error).__name__ + ': ' + str(error)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if receipt['status'] == 'technical_checkpoint_replay_verified':
                try:
                    guard(slot_end - 5)
                    require_clean_worker(receipt)
                except Exception as error:
                    receipt['status'] = 'review_incomplete_or_failed'
                    receipt['error'] = type(error).__name__ + ': ' + str(error)
            receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
            receipt['elapsed_slot_seconds'] = time.monotonic() - started
            receipt['reservation_return'] = 'requires explicit coordination; this receipt does not release AIA reservation'
            signal.setitimer(signal.ITIMER_REAL, max(.001, slot_end-time.monotonic()))
            try:
                publish_receipt(output / 'execution_receipt.json', receipt, lambda: guard(slot_end))
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                for sig, handler in old_handlers.items():
                    signal.signal(sig, handler)
            print(json.dumps(receipt, indent=2))
        if receipt['status'] != 'technical_checkpoint_replay_verified':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
