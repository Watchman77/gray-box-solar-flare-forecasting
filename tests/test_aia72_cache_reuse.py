"""Historical path hints must not become live or wrong-generation cache claims."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from scripts.prepare_aia72_cache_reuse import join_candidates, safe_path


class CacheReuseTests(unittest.TestCase):
    def setUp(self):
        self.uri='gs://suryabench-sharp-pipeline-bamidele/samples_npz/2024/image.npz'
        self.path='/mnt/disks/aia-cache/yearwise/2024/image.npz'
        self.pins={self.uri:{'generation':'123','bytes':10,'md5_base64':'fixed'}}
        self.row={'uri':self.uri,'generation':'123','size':'10','md5':'fixed',
                  'status':'SIZE_MATCH_REQUIRES_CONTENT_HASH','local_path':self.path,
                  'candidate_path':self.path,'candidate_count':'1'}

    def test_matching_pin_remains_unverified_candidate(self):
        got=join_candidates(self.pins,{}, {},pd.DataFrame([self.row])).iloc[0]
        self.assertEqual(got.candidate_category,'historical_matching_metadata_candidate')
        self.assertFalse(got.live_presence_verified)
        self.assertFalse(got.image_bytes_verified_for_inference)

    def test_same_size_or_hash_with_wrong_generation_does_not_accept_candidate(self):
        for key,value in [('generation','124'),('md5','changed'),('size','11')]:
            got=join_candidates(self.pins,{}, {},pd.DataFrame([{**self.row,key:value}])).iloc[0]
            self.assertEqual(got.candidate_category,'no_historical_candidate')
            self.assertNotEqual(got.metadata_mismatch_sources_json,'[]')

    def test_missing_original_location_is_not_presence_evidence(self):
        missing={**self.row,'status':'MISSING','candidate_path':'','candidate_count':'0'}
        got=join_candidates(self.pins,{}, {},pd.DataFrame([missing])).iloc[0]
        self.assertEqual(got.candidate_category,'no_historical_candidate')
        self.assertEqual(got.candidate_paths_json,'[]')
        self.assertTrue(got.historically_reported_missing)
        self.assertIn(self.path,got.expected_locations_not_presence_evidence_json)

    def test_lineage_hint_not_promoted_to_generation_proof(self):
        other={**self.row,'uri':self.uri.replace('image.npz','other.npz')}
        got=join_candidates(self.pins,{}, {self.uri:{self.path}},pd.DataFrame([other])).iloc[0]
        self.assertEqual(got.candidate_category,'historical_unversioned_path_hint')
        self.assertEqual(got.matching_metadata_paths_json,'[]')

    def test_duplicate_uri_inconsistent_missing_and_invalid_paths_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            join_candidates(self.pins,{}, {},pd.DataFrame([self.row,self.row]))
        with self.assertRaisesRegex(ValueError,'Missing/candidate'):
            join_candidates(self.pins,{}, {},pd.DataFrame([{**self.row,'status':'MISSING'}]))
        for path in ['/tmp/image.npz','/mnt/disks/aia-cache/../image.npz','/mnt/disks/aia-cache/wrong.npz']:
            with self.subTest(path=path),self.assertRaises(ValueError):
                safe_path(path,self.uri)


if __name__=='__main__':
    unittest.main()
