import json
from pathlib import Path
import tempfile
import unittest

from scripts.reconcile_aia72_vm_cache import inventory, probe


class CacheReconciliationTests(unittest.TestCase):
    def test_scan_recovers_missing_hint_and_preserves_missing_record(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve(); (root/'kept').mkdir(); (root/'kept/a.npz').write_bytes(b'abc')
            rows=[{'uri':'gs://bucket/'+name,'generation':'1','bytes':'3','md5_base64':'placeholder',
                   'candidate_paths_json':'[]','expected_locations_not_presence_evidence_json':'[]'}
                  for name in ['a.npz','b.npz']]
            result,_=inventory(rows,[root],root,lambda:None)
            self.assertEqual([r['uri'] for r in result],[r['uri'] for r in rows])
            self.assertEqual(result[0]['status'],'candidate_requires_content_verification')
            self.assertEqual(result[1]['status'],'no_size_matching_candidate')
            self.assertFalse(any(r['image_bytes_verified'] for r in result))

    def test_size_and_link_do_not_count_as_verified_file(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve(); source=root/'a.npz'; source.write_bytes(b'abc'); link=root/'link.npz';link.symlink_to(source)
            self.assertEqual(probe(source,4,[root])['status'],'size_mismatch')
            self.assertEqual(probe(link,3,[root])['status'],'nonregular_or_symlink')

    def test_bad_hint_or_traversal_bound_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve(); (root/'a.npz').write_bytes(b'abc')
            row={'uri':'gs://bucket/a.npz','generation':'1','bytes':'3','md5_base64':'x',
                 'candidate_paths_json':json.dumps(['/unrelated/a.npz']),
                 'expected_locations_not_presence_evidence_json':'[]'}
            with self.assertRaisesRegex(ValueError,'outside allowed'):
                inventory([row],[root],root,lambda:None)
            row['candidate_paths_json']='[]'
            with self.assertRaisesRegex(ValueError,'bound exceeded'):
                inventory([row],[root],root,lambda:None,max_entries=0)
