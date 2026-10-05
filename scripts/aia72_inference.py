"""Frozen-model inference on bounded image blocks, retaining every requested case.

This module has no launcher and no optimizer. A reviewed controller supplies the
lease, immutable configuration, output directory and deadline guard.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import threading
import time
import zipfile
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from scripts.aia72_data import LAGS, transform_image, validate_case_times, validate_frame_identity
from scripts.aia72_model import TemporalAIACNNGRU
from scripts.aia72_vm_data import read_source
from scripts.aia72_replay_contract import require, sha, write_json
from scripts.aia72_continue import leased_loader

SEEDS = (17,29,43)
BUCKET = 'suryabench-sharp-pipeline-bamidele'
EXISTING_ROOTS = ('/mnt/disks/aia-cache','/home/abmoses2000')


def payload_matches(payload, pin):
    return len(payload) == pin['bytes'] and base64.b64encode(hashlib.md5(payload).digest()).decode() == pin['md5_base64']


def object_url(pin):
    prefix='gs://'+BUCKET+'/'
    require(pin['uri'].startswith(prefix), 'Unexpected source bucket')
    name=pin['uri'][len(prefix):]
    require(name.endswith('.npz') and '..' not in name.split('/') and not name.startswith('/'), 'Unsafe object name')
    require(isinstance(pin['generation'],str) and pin['generation'].isdigit(), 'Immutable generation required')
    return 'https://storage.googleapis.com/storage/v1/b/'+BUCKET+'/o/'+quote(name,safe='')+'?'+urlencode({'alt':'media','generation':pin['generation']})


def token_from_metadata():
    # Token remains in process memory and is never included in logs or receipts.
    request=Request('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',
                    headers={'Metadata-Flavor':'Google'})
    with urlopen(request,timeout=10) as response:
        raw=response.read(65537)
    require(len(raw)<=65536, 'Unexpected credential response size')
    result=json.loads(raw)
    require(isinstance(result.get('access_token'),str) and result['access_token'], 'Missing VM storage credentials')
    return result['access_token']


def stage_one(pin, candidates, directory, token, guard, fetch=None, reserve_download=None):
    """Reuse exact cached bytes or restore only the pinned source generation."""
    guard()
    defects=[]
    for candidate in candidates:
        path=Path(candidate['path'])
        require(path.is_absolute() and '..' not in path.parts and path.name==Path(pin['uri']).name,
                'Unsafe candidate path')
        require(any(path.is_relative_to(root) for root in EXISTING_ROOTS),
                'Candidate outside existing storage')
        try:
            require(not path.is_symlink() and path.resolve()==path, 'Candidate symlink prohibited')
            before=path.stat()
            if not stat.S_ISREG(before.st_mode) or before.st_size!=pin['bytes']:
                defects.append({'path':str(path),'reason':'nonregular_or_size_mismatch'});continue
            payload=path.read_bytes();after=path.stat()
            require((before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                    (after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'Cache changed during read')
            if not payload_matches(payload,pin):
                defects.append({'path':str(path),'reason':'checksum_mismatch'});continue
            return {**pin,'path':str(path),'status':'ok','owned':False,'sha256':hashlib.sha256(payload).hexdigest(),
                    'prior_candidate_defects':defects}
        except (FileNotFoundError, PermissionError) as error:
            defects.append({'path':str(path),'reason':type(error).__name__})
    guard()
    if reserve_download:
        reserve_download(pin['bytes'])
    try:
        if fetch is None:
            with urlopen(Request(object_url(pin),headers={'Authorization':'Bearer '+token}),timeout=30) as response:
                payload=response.read(pin['bytes']+1)
        else:
            payload=fetch(pin)
    except HTTPError as error:
        if error.code in (404,410):
            return {**pin,'status':'missing_input','owned':False,'reason':'pinned_generation_unavailable',
                    'HTTP_status':error.code,'prior_candidate_defects':defects}
        # Authentication/rate/server failures stop the job rather than biasing
        # availability by classifying a service incident as missing observations.
        raise RuntimeError('Storage request failed with HTTP '+str(error.code)) from None
    guard()
    if not payload_matches(payload,pin):
        return {**pin,'status':'invalid_input','owned':False,'reason':'pinned_download_checksum_mismatch',
                'prior_candidate_defects':defects}
    path=Path(directory)/(hashlib.sha256(pin['uri'].encode()).hexdigest()+'.npz')
    require(path.parent.resolve()==Path(directory).resolve(), 'Scratch path escaped owned block')
    with path.open('xb') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    st=path.stat()
    return {**pin,'path':str(path),'status':'ok','owned':True,'sha256':hashlib.sha256(payload).hexdigest(),
            'inode':st.st_ino,'device':st.st_dev,'mtime_ns':st.st_mtime_ns,
            'prior_candidate_defects':defects}


def retire_owned(records, directory, output, expected_output_sha, guard, evidence=None):
    """Retire only exact objects created by this invocation after durable output."""
    directory=Path(directory)
    require(sha(output)==expected_output_sha, 'Prediction output changed before scratch retirement')
    count=0
    def check_evidence():
        for path,digest in (evidence or {}).items():
            require(sha(path)==digest,'Retirement evidence changed: '+str(path))
    check_evidence()
    for record in records.values():
        if not record.get('owned'):
            continue
        guard();path=Path(record['path'])
        require(not path.is_symlink() and path.parent==directory and path.resolve().parent==directory.resolve(),
                'Retirement path is not an owned regular block object')
        st=path.stat()
        require(stat.S_ISREG(st.st_mode) and (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==
                (record['device'],record['inode'],record['bytes'],record['mtime_ns']), 'Owned object identity changed')
        require(sha(path)==record['sha256'], 'Owned object bytes changed before retirement')
        check_evidence();guard();path.unlink();count+=1
    require(not any(directory.iterdir()), 'Unexpected scratch entry; preserve for review')
    directory.rmdir()
    return count


def wait_for_local_ack(output, block_id, ready, guard):
    """A bounded two-host result receipt must precede duplicate-cache retirement."""
    deadline=time.monotonic()+240
    path=Path(output)/(block_id+'_local_ack.json')
    while not path.exists():
        guard();require(time.monotonic()<deadline,'Local result acknowledgement absent; preserve scratch')
        time.sleep(.2)
    ack_raw=path.read_bytes();ack=json.loads(ack_raw)
    require(ack['status']=='local_block_archive_verified' and ack['block_id']==block_id
            and ack['predictions_sha256']==ready['predictions_sha256']
            and ack['sources_sha256']==ready['sources_sha256'],'Local acknowledgement differs')
    require(sha(Path(output)/(block_id+'_sources.json'))==ready['sources_sha256'],'Source receipt changed before retirement')
    archive=Path(output)/'proofs'/(block_id+'.tar.gz')
    require(sha(archive)==ack['archive_sha256'],'VM proof archive differs from local acknowledgement')
    ack['acknowledgement_file_sha256']=hashlib.sha256(ack_raw).hexdigest()
    return ack


class InferenceDataset(Dataset):
    def __init__(self, frame, records, normalization):
        from collections import OrderedDict
        self.frame=frame.reset_index(drop=True);self.records=records;self.normalization=normalization
        self.cache=OrderedDict()
        validate_case_times(self.frame)
        require(self.frame.forecast_case_id.is_unique,'Duplicate block case')

    def __len__(self):
        return len(self.frame)

    def __getitem__(self,index):
        row=self.frame.iloc[index];images=[];failures=[]
        for lag in LAGS:
            uri=row[f'history_uri_tminus{lag}'];r=self.records[uri]
            if r['status']!='ok':
                failures.append((r['status'],uri+': '+r.get('reason','unavailable')));continue
            try:
                st=Path(r['path']).stat();stamp=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
                cached=self.cache.get(uri)
                if cached is not None and cached[0]==stamp:
                    image=cached[1];self.cache.move_to_end(uri)
                else:
                    raw,_=read_source(r);image=transform_image(raw,self.normalization)
                    self.cache[uri]=(stamp,image)
                    while len(self.cache)>96:self.cache.popitem(last=False)
                images.append(image)
            except FileNotFoundError:
                failures.append(('missing_input',uri+': file disappeared'))
            except (ValueError, OSError, EOFError, zipfile.BadZipFile) as error:
                failures.append(('invalid_input',uri+': '+type(error).__name__+': '+str(error)))
        if failures:
            status='invalid_input' if any(x[0]=='invalid_input' for x in failures) else 'missing_input'
            # Preserve batching: invalid input produces no probability. In eval
            # mode no other case depends on these placeholder pixels.
            return torch.zeros((3,6,256,256)),row.forecast_case_id,status,json.dumps(failures)
        return torch.stack(images),row.forecast_case_id,'ok',''


def loader(dataset):
    return DataLoader(dataset,batch_size=16,shuffle=False,num_workers=4,pin_memory=True,
                      drop_last=False,multiprocessing_context='spawn',prefetch_factor=2,
                      generator=torch.Generator().manual_seed(200017))


def predict_block(models, batches, frame, guard, device='cuda'):
    """One input traversal for the three fixed models; probability mean, no fitting."""
    rows=[]
    with torch.no_grad():
        for images,identities,statuses,reasons in batches:
            guard();offset=len(rows)
            require(list(identities)==frame.forecast_case_id.iloc[offset:offset+len(identities)].tolist(),
                    'Inference support reordered or changed')
            logits={};images=images.to(device,non_blocking=True)
            for seed in SEEDS:
                model=models[seed];require(not model.training,'Model must be frozen in evaluation mode')
                values=model(images).detach().cpu().numpy().astype(np.float64)
                require(values.shape==(len(identities),) and np.isfinite(values).all(),'Nonfinite or malformed logits')
                logits[seed]=values
            probabilities={seed:np.exp(-np.logaddexp(0.,-values)) for seed,values in logits.items()}
            for i,(identity,status,reason) in enumerate(zip(identities,statuses,reasons)):
                require(status in ['ok','missing_input','invalid_input'],'Unknown input status')
                ok=status=='ok'
                r={'forecast_case_id':identity,'input_status':status,'input_failure_reason':reason,
                   'probability':float(np.mean([probabilities[s][i] for s in SEEDS])) if ok else np.nan}
                for seed in SEEDS:
                    r[f'logit_seed_{seed}']=float(logits[seed][i]) if ok else np.nan
                    r[f'probability_seed_{seed}']=float(probabilities[seed][i]) if ok else np.nan
                rows.append(r)
            guard()
    require(len(rows)==len(frame),'Incomplete inference block')
    result=pd.DataFrame(rows)
    for key in ['role','label','issue_utc','history_96_UTC']:
        result[key]=frame[key].to_numpy()
    result=result.rename(columns={'history_96_UTC':'last_observation_utc'})
    return result


def run_inference(c, output, lease_fd, guard):
    output=Path(output);output.mkdir(exist_ok=False)
    source=Path(c['source_root'])
    for record in c['inputs'].values():
        require(sha(source/record['path'])==record['sha256'],'Frozen input differs: '+record['path'])
    frame=pd.read_csv(source/'cases.csv.gz',low_memory=False);validate_case_times(frame)
    require(len(frame)==96596 and frame.forecast_case_id.is_unique,'Comparison case support changed')
    pins=json.loads((source/'objects.json').read_text());blocks=json.loads((source/'blocks.json').read_text())
    cache={r['uri']:r for r in map(json.loads,(source/'cache_records.jsonl').read_text().splitlines())}
    require(set(cache)==set(pins),'Cache reconciliation support differs')
    normalization=json.loads((source/'normalization.json').read_text())
    require(normalization['fit_role']=='train' and normalization['training_case_ids']==
            sorted(frame.loc[frame.role.eq('train'),'forecast_case_id']),'Normalization fit support changed')
    for uri,pin in pins.items():
        require(all(str(cache[uri][k])==str(pin[k]) for k in ['uri','generation','bytes','md5_base64']),
                'Cache metadata differs')
    models={};initial={}
    for seed in SEEDS:
        # Original best files are plain state_dict objects. Their complete file
        # hashes are bound to the accepted training/replay provenance above.
        state=torch.load(source/f'seed_{seed}_best.pt',map_location='cpu',weights_only=True)
        model=TemporalAIACNNGRU();model.load_state_dict(state);model.eval()
        initial[seed]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        models[seed]=model.to('cuda')
    scratch=Path(c['scratch_root']);scratch.mkdir(exist_ok=False)
    accepted=pd.read_csv(source/'three_seed_selection_predictions.csv.gz')
    require(len(accepted)==3905 and accepted.forecast_case_id.is_unique,'Accepted selection export differs')
    # Existing accepted selection predictions are copied without GPU repetition.
    (output/'accepted_selection_predictions.csv.gz').write_bytes((source/'three_seed_selection_predictions.csv.gz').read_bytes())
    schedule=[b for role in c['role_order'] for b in blocks if b['role']==role]
    require(set(c['role_order'])==set(frame.role)-{'model_validation'},'Inference role support differs')
    completed=[];started=time.monotonic();download_bytes=0
    budget_lock=threading.Lock();reserved_download_bytes=0
    def reserve_download(size):
        nonlocal reserved_download_bytes
        with budget_lock:
            require(reserved_download_bytes+size<=c['maximum_download_bytes'],'Transfer bound reached before request')
            reserved_download_bytes+=size
    by_role={role:g.reset_index(drop=True) for role,g in frame.groupby('role',sort=False)}
    for block in schedule:
        guard();block_id=block['block_id'];selected=by_role[block['role']].iloc[block['start']:block['stop']].reset_index(drop=True)
        require(selected.forecast_case_id.tolist()==block['case_ids'],'Block identity/order differs')
        require(block['all_images_bytes_upper_bound']<=c['scratch_cap_bytes'],'Block exceeds scratch bound')
        directory=scratch/block_id;directory.mkdir(exist_ok=False)
        guard();token=token_from_metadata()
        with ThreadPoolExecutor(max_workers=8) as pool:
            records=list(pool.map(lambda uri:stage_one(pins[uri],cache[uri]['candidates'],directory,token,guard,
                                                      reserve_download=reserve_download),block['uris']))
        del token
        records={r['uri']:r for r in records};download_bytes+=sum(r['bytes'] for r in records.values() if r.get('owned'))
        require(download_bytes<=c['maximum_download_bytes'],'Cumulative transfer bound exceeded')
        require(sum(p.stat().st_size for p in directory.iterdir())<=c['scratch_cap_bytes'],'Owned scratch cap exceeded')
        write_json(output/(block_id+'_sources.json'),records)
        with (output/(block_id+'_sources.json')).open('rb') as durable:os.fsync(durable.fileno())
        dataset=InferenceDataset(selected,records,normalization)
        batches=leased_loader(loader,lease_fd,dataset)
        prediction=predict_block(models,batches,selected,guard)
        del batches,dataset;gc.collect()
        path=output/(block_id+'_predictions.csv.gz')
        prediction.to_csv(path,index=False,compression={'method':'gzip','mtime':0})
        with path.open('rb') as durable:os.fsync(durable.fileno())
        output_fd=os.open(output,os.O_RDONLY)
        try:os.fsync(output_fd)
        finally:os.close(output_fd)
        digest=sha(path)
        ready={'block_id':block_id,'predictions_sha256':digest,
               'sources_sha256':sha(output/(block_id+'_sources.json')),'cases':len(selected)}
        write_json(output/(block_id+'_ready.json'),ready)
        ack=wait_for_local_ack(output,block_id,ready,guard)
        evidence={path:digest,output/(block_id+'_sources.json'):ready['sources_sha256'],
                  output/(block_id+'_local_ack.json'):ack['acknowledgement_file_sha256'],
                  output/'proofs'/(block_id+'.tar.gz'):ack['archive_sha256']}
        retired=retire_owned(records,directory,path,digest,guard,evidence)
        record={'block_id':block_id,'role':block['role'],'cases':len(prediction),'predictions_sha256':digest,
                'sources_sha256':sha(output/(block_id+'_sources.json')),'statuses':prediction.input_status.value_counts().to_dict(),
                'owned_images_retired':retired,'cumulative_download_bytes':download_bytes,
                'local_archive_sha256':ack['archive_sha256'],
                'reserved_download_bytes_including_failed_requests':reserved_download_bytes,
                'elapsed_seconds':time.monotonic()-started}
        completed.append(record)
        write_json(output/'progress.json',{'status':'inference_in_progress','blocks_completed':len(completed),
                   'blocks_requested':len(schedule),'completed_cases':sum(r['cases'] for r in completed),
                   'accepted_selection_cases_reused':3905,'latest':record})
        print(json.dumps(record),flush=True)
    require(all(torch.equal(initial[s][k],v.detach().cpu()) for s in SEEDS for k,v in models[s].state_dict().items()),
            'Inference changed trained parameters or model buffers')
    for record in c['inputs'].values():
        require(sha(source/record['path'])==record['sha256'],'Input changed during inference')
    require(sum(r['cases'] for r in completed)+3905==96596,'Full comparison coverage incomplete')
    result={'status':'all_prediction_blocks_complete_pending_collection','blocks':completed,
            'inferred_cases':sum(r['cases'] for r in completed),'accepted_selection_cases_reused':3905,
            'fitting_steps':0,'models_unchanged':True,'download_bytes':download_bytes,
            'elapsed_seconds':time.monotonic()-started,'scientific_acceptance':False,
            'historical_availability':'unverified_retrospective','fusion_or_calibration_performed':False}
    write_json(output/'result.json',result)
    return result
