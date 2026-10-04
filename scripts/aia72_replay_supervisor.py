"""Replay-only process supervision with persistent launch and cleanup evidence."""
import os
import signal
import subprocess
import time

from scripts.aia72_replay_contract import require


def inspect_group(pgid, deadline):
    remaining = deadline - time.monotonic()
    require(remaining > 0, 'No time remains to inspect worker group')
    table = subprocess.check_output(['ps', '-eo', 'pid,pgid,stat'], text=True,
                                    timeout=min(3, remaining))
    return [int(parts[0]) for line in table.splitlines()[1:]
            if len(parts := line.split()) >= 3 and int(parts[1]) == pgid
            and not parts[2].startswith('Z')]


def stop_group(process, deadline):
    """Always attempt bounded reaping and final inspection, retaining every error."""
    require(type(process.pid) is int and process.pid > 1 and process.pid != os.getpgrp(),
            'Refusing to signal an invalid or controller process group')
    errors = []
    reaped = False
    empty = False

    def record(stage, error):
        errors.append({'stage': stage, 'error': type(error).__name__ + ': ' + str(error)})

    for sig in [signal.SIGTERM, signal.SIGKILL]:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        except Exception as error:
            record('signal_' + str(sig), error)
        stage_end = min(deadline, time.monotonic() + 3)
        while time.monotonic() < stage_end:
            try:
                process.wait(timeout=max(.001, min(.1, stage_end - time.monotonic())))
                reaped = True
            except subprocess.TimeoutExpired:
                pass
            except Exception as error:
                record('wait', error)
                break
            try:
                empty = not inspect_group(process.pid, stage_end)
            except Exception as error:
                record('inspect', error)
                break
            if empty and reaped:
                break
            # wait() returns immediately for an exited parent with live descendants.
            time.sleep(max(0, min(.05, stage_end - time.monotonic())))
        if empty and reaped:
            break

    # Neither a signalling error nor an inspection error may skip these attempts.
    try:
        remaining = deadline - time.monotonic()
        require(remaining > 0, 'Cleanup deadline reached before final reap')
        process.wait(timeout=min(.5, remaining))
        reaped = True
    except Exception as error:
        record('final_wait', error)
    try:
        empty = not inspect_group(process.pid, deadline)
    except Exception as error:
        empty = False
        record('final_inspect', error)
    return {'worker_reaped': reaped, 'process_group_empty': empty,
            'cleanup_errors': errors, 'returncode': process.returncode,
            'cleanup_verified': reaped and empty and not errors and time.monotonic() < deadline}


def require_cleanup_proof(receipt):
    """Unknown launch state is never equivalent to a worker that was not launched."""
    state = receipt.get('worker_launch_state')
    if state == 'not_attempted':
        require(receipt.get('worker_pid') is None, 'No-launch receipt contains a worker PID')
        return
    require(state == 'started' and type(receipt.get('worker_pid')) is int
            and receipt['worker_pid'] > 1, 'Worker launch state is unknown')
    require(receipt.get('cleanup_verified') is True and receipt.get('worker_reaped') is True
            and receipt.get('process_group_empty') is True
            and receipt.get('cleanup_errors') == [], 'Completed worker cleanup proof is missing')


def supervise(command, environment, cwd, log, work_end, slot_end, pass_fds=(), guard=None, *, receipt):
    """Record PID in the caller immediately; do not rely on a successful return."""
    receipt.update(worker_launch_state='not_attempted', worker_pid=None, cleanup_verified=False,
                   worker_reaped=False, process_group_empty=False, cleanup_errors=[],
                   returncode=None, work_deadline_reached=False, cancellation_signals=[], guard_error=None)
    watched = [signal.SIGALRM, signal.SIGINT, signal.SIGTERM]
    blocked = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
    arrived = receipt['cancellation_signals']
    handlers = {sig: signal.signal(sig, lambda number, _frame: arrived.append(number)) for sig in watched}
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
    process = None
    try:
        require(time.monotonic() < work_end and not arrived, 'Slot cancelled before worker launch')
        receipt['worker_launch_state'] = 'attempting'
        process = subprocess.Popen(command, env=environment, cwd=cwd, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   pass_fds=pass_fds)
        receipt.update(worker_launch_state='started', worker_pid=process.pid)
        while process.poll() is None and not arrived:
            remaining = work_end - time.monotonic()
            if remaining <= 0:
                receipt['work_deadline_reached'] = True
                break
            if guard:
                try:
                    guard()
                except Exception as error:
                    receipt['guard_error'] = type(error).__name__ + ': ' + str(error)
                    break
            try:
                process.wait(timeout=min(.1, remaining))
            except subprocess.TimeoutExpired:
                pass
        receipt['work_deadline_reached'] |= time.monotonic() >= work_end
    except BaseException as error:
        receipt['supervisor_error'] = repr(error)
        raise
    finally:
        try:
            if process is not None:
                # Also retain identity if an unexpected exception interrupted bookkeeping.
                receipt.update(worker_launch_state='started', worker_pid=process.pid)
                try:
                    receipt.update(stop_group(process, slot_end))
                except BaseException as error:
                    receipt.update(cleanup_verified=False, process_group_empty=False)
                    receipt['cleanup_errors'].append({'stage': 'cleanup_exception', 'error': repr(error)})
                    raise
        finally:
            blocked = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
    return receipt
