"""CPU metadata tests: an unfinished or altered snapshot cannot become a replay package."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA before CPU checks'
from scripts.aia72_replay_contract import sha
from scripts.prepare_aia72_completion_replay import verified_snapshot, prepare


class ReplayPreparationTests(unittest.TestCase):
    def fixture(self, root):
        root.mkdir()
        payload = root/'checkpoint.pt'
        payload.write_bytes(b'synthetic metadata test; not a trained model')
        manifest = {'checkpoint.pt':{'sha256':sha(payload),'bytes':payload.stat().st_size}}
        verification = {
            'status':'bounded_invocation_independently_verified_and_explicit_return_confirmed',
            'remaining_seeds_fit_complete':True, 'CUDA_initialized_by_verification':False,
            'seed17_unchanged':True, 'reservation_overrun_seconds':0}
        (root/'snapshot_manifest.json').write_text(json.dumps(manifest))
        (root/'independent_verification.json').write_text(json.dumps(verification))
        return self.digests(root)

    def digests(self, root):
        return sha(root/'snapshot_manifest.json'),sha(root/'independent_verification.json')

    def test_import_has_no_torch_or_GPU_side_effect(self):
        result = subprocess.run([sys.executable,'-c',
            "import scripts.prepare_aia72_completion_replay, scripts.archive_aia72_completion_training, sys; assert 'torch' not in sys.modules"],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_intact_snapshot_is_read_only_and_untrusted_model_bytes_are_not_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'snapshot'; digests=self.fixture(root)
            before={p.name:sha(p) for p in root.iterdir()}
            manifest, verification=verified_snapshot(root,*digests)
            self.assertEqual(set(manifest),{'checkpoint.pt'})
            self.assertTrue(verification['remaining_seeds_fit_complete'])
            self.assertEqual(before,{p.name:sha(p) for p in root.iterdir()})

    def test_changed_weights_or_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'snapshot'; digests=self.fixture(root)
            for supplied in [('0'*64,digests[1]),(digests[0],'0'*64)]:
                with self.assertRaisesRegex(ValueError,'differs'):
                    verified_snapshot(root,*supplied)
            (root/'checkpoint.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'member changed'):
                verified_snapshot(root,*digests)

    def test_unsafe_or_missing_members_rejected(self):
        for name in ['../checkpoint.pt','/tmp/checkpoint.pt','missing.pt','link.pt']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'snapshot';self.fixture(root)
                if name=='link.pt':(root/name).symlink_to(root/'checkpoint.pt')
                manifest={name:{'sha256':sha(root/'checkpoint.pt'),'bytes':(root/'checkpoint.pt').stat().st_size}}
                (root/'snapshot_manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):verified_snapshot(root,*self.digests(root))

    def test_partial_or_unverified_archive_cannot_create_package(self):
        for field,value in [('remaining_seeds_fit_complete',False),('status','still_training'),
                            ('seed17_unchanged',False),('reservation_overrun_seconds',1)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'snapshot';self.fixture(root)
                path=root/'independent_verification.json';v=json.loads(path.read_text());v[field]=value
                path.write_text(json.dumps(v));output=Path(tmp)/'package'
                with self.assertRaises(ValueError):
                    prepare(Path.cwd(),root,Path(tmp)/'original',output,*self.digests(root))
                self.assertFalse(output.exists())

    def test_required_training_artifacts_must_be_in_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'snapshot';digests=self.fixture(root);output=Path(tmp)/'package'
            with self.assertRaisesRegex(ValueError,'Required artifact missing'):
                prepare(Path.cwd(),root,Path(tmp)/'original',output,*digests)
            self.assertFalse(output.exists())

    def test_preexisting_package_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'package';output.mkdir();sentinel=output/'existing';sentinel.write_text('preserve')
            with self.assertRaisesRegex(ValueError,'Preserve every'):
                prepare(Path.cwd(),Path(tmp)/'absent',Path(tmp)/'absent',output,'bad','bad')
            self.assertEqual(sentinel.read_text(),'preserve')


if __name__=='__main__':
    unittest.main()
