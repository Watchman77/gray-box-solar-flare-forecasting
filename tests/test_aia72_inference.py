import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.aia72_inference import object_url, payload_matches, stage_one, retire_owned, predict_block


class ConstantModel(torch.nn.Module):
    def __init__(self,value):
        super().__init__();self.value=value
    def forward(self,x):
        return torch.full((len(x),),self.value)


class InferenceTests(unittest.TestCase):
    def pin(self,payload=b'abc'):
        return {'uri':'gs://suryabench-sharp-pipeline-bamidele/samples_npz/2026/a.npz',
                'generation':'123456','bytes':len(payload),
                'md5_base64':base64.b64encode(hashlib.md5(payload).digest()).decode()}

    def test_download_is_generation_bound_and_checksum_checked(self):
        pin=self.pin();self.assertIn('generation=123456',object_url(pin))
        self.assertIn('samples_npz%2F2026%2Fa.npz',object_url(pin))
        self.assertFalse(payload_matches(b'abd',pin))
        with tempfile.TemporaryDirectory() as d:
            count=[]
            bad=stage_one(pin,[],Path(d),'secret',lambda:None,fetch=lambda _:b'abd',reserve_download=count.append)
            self.assertEqual(bad['status'],'invalid_input');self.assertEqual(list(Path(d).iterdir()),[])
            self.assertEqual(count,[3])

    def test_only_owned_objects_retire_after_unchanged_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();block=root/'block';block.mkdir()
            preserved=root/'existing.npz';preserved.write_bytes(b'abc')
            record=stage_one(self.pin(),[],block,'secret',lambda:None,fetch=lambda _:b'abc')
            output=root/'prediction.csv';output.write_bytes(b'one,two\n')
            records={'owned':record,'foreign':{'path':str(preserved),'owned':False}}
            with self.assertRaisesRegex(ValueError,'output changed'):
                retire_owned(records,block,output,'0'*64,lambda:None)
            digest=hashlib.sha256(output.read_bytes()).hexdigest()
            self.assertEqual(retire_owned(records,block,output,digest,lambda:None),1)
            self.assertEqual(preserved.read_bytes(),b'abc');self.assertFalse(block.exists())

    def test_probability_mean_missing_rows_and_case_order(self):
        models={s:ConstantModel(v).eval() for s,v in zip([17,29,43],[-2.,0.,1.])}
        frame=pd.DataFrame({'forecast_case_id':['a','b'],'role':['test']*2,'label':[0,1],
                            'issue_utc':['2026-01-01']*2,'history_96_UTC':['2025-12-31']*2})
        batches=[(torch.zeros((2,1)),['a','b'],['ok','missing_input'],['','missing'])]
        result=predict_block(models,batches,frame,lambda:None,device='cpu')
        expected=np.mean(1/(1+np.exp(-np.array([-2.,0.,1.]))))
        self.assertAlmostEqual(result.iloc[0].probability,expected,places=14)
        self.assertTrue(np.isnan(result.iloc[1].probability))
        self.assertEqual(result.forecast_case_id.tolist(),['a','b'])
        wrong=[(torch.zeros((2,1)),['b','a'],['ok','ok'],['',''])]
        with self.assertRaisesRegex(ValueError,'support reordered'):
            predict_block(models,wrong,frame,lambda:None,device='cpu')

    def test_download_budget_precedes_network_or_file_creation(self):
        with tempfile.TemporaryDirectory() as d:
            def reject(_):raise ValueError('budget exhausted')
            with self.assertRaisesRegex(ValueError,'budget exhausted'):
                stage_one(self.pin(),[],Path(d),'secret',lambda:None,
                          fetch=lambda _:self.fail('request should not occur'),reserve_download=reject)
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_cached_bytes_do_not_reserve_download_budget(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();path=root/'a.npz';path.write_bytes(b'abc')
            with patch('scripts.aia72_inference.EXISTING_ROOTS',(str(root),)):
                record=stage_one(self.pin(),[{'path':str(path)}],root,'secret',lambda:None,
                    fetch=lambda _:self.fail('no fetch for cached bytes'),
                    reserve_download=lambda _:self.fail('no budget charge for cached bytes'))
            self.assertFalse(record['owned']);self.assertEqual(record['status'],'ok')
