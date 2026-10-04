"""Owned temporary CPU children exercise cleanup failures without VM/GPU work."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA before CPU checks'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import aia72_replay_supervisor as supervisor


class SupervisorTests(unittest.TestCase):
    def child(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                   start_new_session=True, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        def reap_fixture():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
        self.addCleanup(reap_fixture)
        return process

    def test_permission_error_does_not_skip_kill_reap_or_final_inspection(self):
        process = self.child()
        actual_kill = os.killpg
        def deny_term(pgid, sig):
            if sig == signal.SIGTERM:
                raise PermissionError('synthetic TERM denial')
            return actual_kill(pgid, sig)
        with mock.patch.object(supervisor.os, 'killpg', side_effect=deny_term), \
             mock.patch.object(process, 'wait', wraps=process.wait) as waits, \
             mock.patch.object(supervisor, 'inspect_group', wraps=supervisor.inspect_group) as inspections:
            result = supervisor.stop_group(process, time.monotonic() + 8)
        self.assertGreater(waits.call_count, 0)
        self.assertGreater(inspections.call_count, 0)
        self.assertEqual(process.returncode, -signal.SIGKILL)
        self.assertTrue(result['worker_reaped'])
        self.assertTrue(result['process_group_empty'])
        self.assertFalse(result['cleanup_verified'])
        self.assertIn('PermissionError', result['cleanup_errors'][0]['error'])

    def test_inspection_error_retained_after_successful_reaping(self):
        process = self.child()
        real_inspect = supervisor.inspect_group
        count = 0
        def fail_once(pgid, deadline):
            nonlocal count
            count += 1
            if count == 1:
                raise PermissionError('synthetic inspection denial')
            return real_inspect(pgid, deadline)
        with mock.patch.object(supervisor, 'inspect_group', side_effect=fail_once):
            result = supervisor.stop_group(process, time.monotonic() + 5)
        self.assertTrue(result['worker_reaped'])
        self.assertTrue(result['process_group_empty'])
        self.assertFalse(result['cleanup_verified'])
        self.assertGreaterEqual(count, 2)

    def test_all_signals_denied_still_attempts_wait_and_cannot_claim_cleanup(self):
        process = self.child()
        start = time.monotonic()
        with mock.patch.object(supervisor.os, 'killpg', side_effect=PermissionError('denied')), \
             mock.patch.object(process, 'wait', wraps=process.wait) as waits:
            result = supervisor.stop_group(process, start + .25)
        self.assertGreater(waits.call_count, 0)
        self.assertLess(time.monotonic() - start, 1)
        self.assertIsNone(process.poll())
        self.assertFalse(result['cleanup_verified'])
        self.assertFalse(result['process_group_empty'])

    def test_normal_exited_worker_has_complete_proof(self):
        receipt = {}
        with tempfile.TemporaryFile(mode='w+') as log:
            supervisor.supervise([sys.executable, '-c', 'pass'], os.environ.copy(), Path.cwd(), log,
                                 time.monotonic() + 5, time.monotonic() + 8, receipt=receipt)
        self.assertEqual(receipt['returncode'], 0)
        self.assertTrue(receipt['cleanup_verified'])
        supervisor.require_cleanup_proof(receipt)

    def test_cleanup_exception_keeps_pid_in_callers_receipt(self):
        receipt = {}
        with tempfile.TemporaryFile(mode='w+') as log, \
             mock.patch.object(supervisor, 'stop_group', side_effect=PermissionError('unexpected cleanup fault')):
            with self.assertRaises(PermissionError):
                supervisor.supervise([sys.executable, '-c', 'pass'], os.environ.copy(), Path.cwd(), log,
                                     time.monotonic() + 5, time.monotonic() + 8, receipt=receipt)
        self.assertEqual(receipt['worker_launch_state'], 'started')
        self.assertGreater(receipt['worker_pid'], 1)
        self.assertFalse(receipt['cleanup_verified'])
        with self.assertRaises(ValueError):
            supervisor.require_cleanup_proof(receipt)

    def test_spawn_exception_is_unknown_not_no_launch(self):
        receipt = {}
        with tempfile.TemporaryFile(mode='w+') as log, \
             mock.patch.object(supervisor.subprocess, 'Popen', side_effect=OSError('spawn failure')):
            with self.assertRaises(OSError):
                supervisor.supervise(['unused'], os.environ.copy(), Path.cwd(), log,
                                     time.monotonic() + 5, time.monotonic() + 8, receipt=receipt)
        self.assertEqual(receipt['worker_launch_state'], 'attempting')
        with self.assertRaisesRegex(ValueError, 'unknown'):
            supervisor.require_cleanup_proof(receipt)

    def test_guard_failure_still_cleans_worker_and_retains_guard_error(self):
        receipt = {}
        with tempfile.TemporaryFile(mode='w+') as log:
            supervisor.supervise([sys.executable, '-c', 'import time; time.sleep(30)'],
                                 os.environ.copy(), Path.cwd(), log, time.monotonic() + 5,
                                 time.monotonic() + 8, guard=mock.Mock(side_effect=ValueError('STOP')),
                                 receipt=receipt)
        self.assertTrue(receipt['cleanup_verified'])
        self.assertIn('STOP', receipt['guard_error'])
        self.assertNotEqual(receipt['returncode'], 0)


if __name__ == '__main__':
    unittest.main()
