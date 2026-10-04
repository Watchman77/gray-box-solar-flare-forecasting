"""CPU-only checks of source restrictions and bounded, batch-preserving blocks."""
import base64
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from scripts.prepare_aia72_inference_inputs import BUCKET, parse_uri, validate_object, make_blocks


class InferencePreparationTests(unittest.TestCase):
    def fixture(self, count=51):
        frame=pd.DataFrame({'forecast_case_id':[f'case{i}' for i in range(count)],
                            'role':['first']*35+['second']*(count-35)})
        objects={}
        for lag in [288,192,96]:
            frame[f'history_uri_tminus{lag}']=[f'image_{i}_{lag}' for i in range(count)]
            objects.update({uri:{'bytes':10} for uri in frame[f'history_uri_tminus{lag}']})
        return frame,objects

    def test_preserves_role_order_and_partial_batches_while_shrinking(self):
        frame,objects=self.fixture()
        blocks=make_blocks(frame,objects,max_cases=512,max_bytes=16*3*10)
        self.assertEqual([b['case_count'] for b in blocks],[16,16,3,16])
        self.assertEqual([v for b in blocks for v in b['case_ids']],frame.forecast_case_id.tolist())
        self.assertTrue(all(b['all_images_bytes_upper_bound']<=480 for b in blocks))
        self.assertEqual([b['final_within_role'] for b in blocks],[False,False,True,True])

    def test_shared_objects_are_counted_once_per_block(self):
        frame,objects=self.fixture()
        for lag in [288,192,96]:
            frame[f'history_uri_tminus{lag}']='shared'
        blocks=make_blocks(frame,{'shared':{'bytes':10}},max_bytes=10)
        self.assertEqual([b['case_count'] for b in blocks],[35,16])
        self.assertTrue(all(b['all_images_bytes_upper_bound']==10 for b in blocks))

    def test_refuses_missing_pin_duplicate_case_or_too_small_scratch(self):
        frame,objects=self.fixture()
        bad=dict(objects);bad.pop(next(iter(bad)))
        with self.assertRaisesRegex(ValueError,'Object pin missing'):
            make_blocks(frame,bad)
        with self.assertRaisesRegex(ValueError,'original batch exceeds'):
            make_blocks(frame,objects,max_bytes=479)
        frame.loc[1,'forecast_case_id']=frame.loc[0,'forecast_case_id']
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            make_blocks(frame,objects)

    def test_source_bucket_prefix_and_generation_must_be_explicit(self):
        uri=f'gs://{BUCKET}/samples_npz/2024/example.npz'
        self.assertEqual(parse_uri(uri),'samples_npz/2024/example.npz')
        for bad in [uri.replace(BUCKET,'other'),uri.replace('/2024/','/../'),uri.replace('samples_npz/','secret/'),uri+'.txt']:
            with self.subTest(uri=bad),self.assertRaises(ValueError):
                parse_uri(bad)
        record={'uri':uri,'generation':'123456','bytes':16,'md5_base64':base64.b64encode(b'a'*16).decode()}
        validate_object(record)
        for field,value in [('generation','latest'),('bytes',True),('bytes',0),('bytes',16*1024**2+1),('md5_base64','AAAA')]:
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_object({**record,field:value})


if __name__=='__main__':
    unittest.main()
