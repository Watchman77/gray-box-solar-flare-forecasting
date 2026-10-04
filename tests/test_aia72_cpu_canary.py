"""The staging canary binds cancellation to its actual supervisor receipt."""
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.aia72_replay_supervisor import supervise,require_cleanup_proof
from scripts.stage_aia72_completion_replay import check_canary


class CanaryTests(unittest.TestCase):
    def run_canary(self,code,guard=None):
        receipt={}
        with tempfile.TemporaryFile(mode='w+') as log:
            supervise([sys.executable,'-c',code],os.environ.copy(),Path.cwd(),log,
                      time.monotonic()+5,time.monotonic()+8,guard=guard,receipt=receipt)
        return receipt

    def test_normal_cpu_child_passes(self):
        receipt=self.run_canary('pass')
        check_canary(receipt,require_cleanup_proof)

    def test_signal_received_by_supervisor_cannot_be_swallowed_by_staging(self):
        receipt=self.run_canary('import time;time.sleep(30)',lambda:signal.raise_signal(signal.SIGTERM))
        self.assertEqual(receipt['cancellation_signals'],[signal.SIGTERM])
        self.assertTrue(receipt['cleanup_verified'])
        with self.assertRaises(ValueError):
            check_canary(receipt,require_cleanup_proof)
        # Even a coincident successful child exit cannot hide the cancellation.
        with self.assertRaisesRegex(ValueError,'cancelled'):
            check_canary({**receipt,'returncode':0},require_cleanup_proof)

    def test_unknown_launch_cannot_pass_canary(self):
        with self.assertRaises(ValueError):
            check_canary({'returncode':0},require_cleanup_proof)


if __name__=='__main__':
    unittest.main()
