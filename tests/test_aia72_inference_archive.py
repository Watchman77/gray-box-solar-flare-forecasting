import csv
import gzip
import hashlib
import io
import json
import os
import tarfile
import time
import unittest
from scripts.aia72_inference_archive_relay import verify_archive,write_ack


class InferenceArchiveTests(unittest.TestCase):
    def fixture(self, probability='.5', bad_member=False):
        uri='gs://suryabench-sharp-pipeline-bamidele/samples_npz/2026/a.npz'
        pin={'uri':uri,'generation':'123','bytes':3,'md5_base64':'abc'}
        block={'block_id':'test_000000_000001','case_ids':['a'],'uris':[uri]}
        case={'forecast_case_id':'a','role':'test','label':'1','issue_utc':'2026-01-01T00:00:00Z','history_96_UTC':'2025-12-31T22:24:00Z'}
        row={k:v for k,v in case.items() if k!='history_96_UTC'}
        row.update(last_observation_utc=case['history_96_UTC'],input_status='ok',input_failure_reason='',probability=probability)
        for seed in [17,29,43]:row[f'logit_seed_{seed}']='0';row[f'probability_seed_{seed}']='.5'
        csv_text=io.StringIO();writer=csv.DictWriter(csv_text,fieldnames=list(row));writer.writeheader();writer.writerow(row)
        members={block['block_id']+'_predictions.csv.gz':gzip.compress(csv_text.getvalue().encode()),
                 block['block_id']+'_sources.json':json.dumps({uri:{**pin,'owned':False,'status':'ok'}}).encode()}
        if bad_member:members['../outside']=b'x'
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            for name,payload in members.items():
                info=tarfile.TarInfo(name);info.size=len(payload);archive.addfile(info,io.BytesIO(payload))
        raw=buffer.getvalue();message={'block_id':block['block_id'],'archive_sha256':hashlib.sha256(raw).hexdigest()}
        return raw,message,block,{uri:pin},{'a':case}

    def test_exact_archive_accepted_with_support_and_probability_checks(self):
        ack=verify_archive(*self.fixture());self.assertEqual(ack['cases'],1)
        self.assertEqual(ack['status'],'local_block_archive_verified')

    def test_wrong_ensemble_mean_is_not_acknowledged(self):
        with self.assertRaisesRegex(ValueError,'ensemble mean'):verify_archive(*self.fixture(probability='.6'))

    def test_unexpected_member_is_not_extracted_or_acknowledged(self):
        with self.assertRaisesRegex(ValueError,'archive members'):verify_archive(*self.fixture(bad_member=True))

    def test_full_ack_pipe_stops_at_immutable_deadline(self):
        reader,writer=os.pipe()
        try:
            os.set_blocking(writer,False)
            try:
                while True:os.write(writer,b'x'*4096)
            except BlockingIOError:pass
            started=time.monotonic()
            with self.assertRaisesRegex(ValueError,'deadline'):
                write_ack(writer,b'ack',started+.05,[])
            self.assertLess(time.monotonic()-started,1)
        finally:os.close(reader);os.close(writer)

    def test_cancelled_ack_writes_no_bytes(self):
        reader,writer=os.pipe()
        try:
            with self.assertRaisesRegex(ValueError,'cancelled'):
                write_ack(writer,b'ack',time.monotonic()+1,[15])
            os.set_blocking(reader,False)
            with self.assertRaises(BlockingIOError):os.read(reader,3)
        finally:os.close(reader);os.close(writer)
