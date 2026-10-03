"""Visible-notebook worker: original train_seed, unchanged data and short external budget."""
import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import fcntl
from functools import partial
from multiprocessing.reduction import DupFd

import nbformat
import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.aia72_replay_contract import require, sha, write_json, verify_inputs, LOCK_INODE
from scripts.aia72_continue_contract import check_contract, verify_initial_training, verify_completed17, guard
from scripts.aia72_vm_data import ExistingAIA72Dataset
from scripts.aia72_data import LAGS
from scripts.dataset_io import file_sha256
from scripts.aia72_model import TemporalAIACNNGRU
import scripts.train_aia72_gpu as original
from scripts.train_aia72_gpu import atomic_save, rng_state, restore_rng, set_seed, validation_predictions

# Spawned loader workers each receive a duplicated reference to the same open-file
# description. The lock therefore survives both controller and model-worker failure.
_LOADER_LEASE = None


def receive_lease(worker_id, descriptors, expected_inode):
    global _LOADER_LEASE
    _LOADER_LEASE = descriptors[worker_id].detach()
    # SCM_RIGHTS reception does not promise FD_CLOEXEC. Keep this loader's
    # reference, but do not leak it into detached torch_shm_manager execs.
    os.set_inheritable(_LOADER_LEASE, False)
    require(os.fstat(_LOADER_LEASE).st_ino == expected_inode, 'Loader lease changed')
    fcntl.flock(_LOADER_LEASE, fcntl.LOCK_EX | fcntl.LOCK_NB)


def leased_loader(factory, lease_fd, *args, **kwargs):
    value = factory(*args, **kwargs)
    require(value.worker_init_fn is None, 'Original loader initializer changed')
    if value.num_workers:
        descriptors = [DupFd(lease_fd) for _ in range(value.num_workers)]
        value.worker_init_fn = partial(receive_lease, descriptors=descriptors,
                                      expected_inode=os.fstat(lease_fd).st_ino)
    return value


def inspect_inputs(c, source, initial):
    paths = verify_inputs(c, source)
    verify_initial_training(c, initial)
    frame = pd.read_csv(paths['cases'])
    require(frame.forecast_case_id.is_unique and frame.label.isin([0,1]).all(), 'Case identities/outcomes changed')
    require(frame.role.value_counts().to_dict() == {'train':25586,'model_validation':3905}, 'Support changed')
    config = json.loads(paths['configuration'].read_text())
    parent = json.loads(paths['run_contract'].read_text())
    require(config == parent['configuration'], 'Original science configuration changed')
    normalization = json.loads(paths['normalization'].read_text())
    require(normalization['training_case_ids'] == sorted(frame.loc[frame.role.eq('train'),'forecast_case_id']), 'Normalization support changed')
    records = json.loads(paths['sources'].read_text())
    wanted = set(frame[[f'history_uri_tminus{x}' for x in LAGS]].to_numpy().ravel())
    require(set(records) == wanted, 'Image bindings changed')
    for record in records.values():
        require(Path(record['path']).is_relative_to('/mnt/disks/aia-cache/cycle24'), 'Unexpected image path')
    resume = torch.load(Path(initial)/'seed_29_resume.pt',map_location='cpu',weights_only=True)
    require(resume['contract_sha256'] == c['parent_contract_sha256'], 'Checkpoint identity changed')
    require({k:resume['state'][k] for k in c['initial_seed29_state']} == c['initial_seed29_state'], 'Resume cursor/state changed')
    require(not (Path(initial)/'seed_43_resume.pt').exists(), 'Seed43 already exists')
    return frame, records, normalization, config, resume['state']


def run_selected(train, selection, config, output, parent_sha, deadline, guard_fn,
                 trainer=original.train_seed):
    """No scientific tuning and no call to the old exhausted run_training scheduler."""
    results = []
    for seed in [29,43]:
        guard_fn()
        if time.monotonic() >= deadline:
            break
        result = trainer(train,selection,config,seed,output,parent_sha,'cuda',deadline)
        results.append(result)
        if result['status'] != 'seed_fit_complete_pending_replay':
            break
    complete = len(results)==2 and all(r['status']=='seed_fit_complete_pending_replay' for r in results)
    return {'status':'remaining_seeds_fit_pending_replay' if complete else 'checkpointed_incomplete',
            'seeds':results,'remaining_seeds_fit_complete':complete,'scientific_acceptance':False,
            'seed17_repeated':False,'later_period_evaluation':False}


def validate_outputs(c, training):
    verify_completed17(c, training)
    summaries=[]
    for seed in [29,43]:
        path=Path(training)/f'seed_{seed}_resume.pt'
        if not path.exists():
            require(seed==43,'Resumed seed29 disappeared'); continue
        saved=torch.load(path,map_location='cpu',weights_only=True)
        require(saved['contract_sha256']==c['parent_contract_sha256'],'Output checkpoint contract changed')
        state=saved['state']
        minimum=c['initial_seed29_state']['steps'] if seed==29 else 0
        require(state['steps']>=minimum and math.isfinite(state['elapsed_seconds']), 'Progress moved backwards')
        require(0<=state['cursor']<=25586 and (state['cursor']%16==0 or state['cursor']==25586),'Invalid saved cursor')
        for tensor in saved['model'].values():
            require(torch.isfinite(tensor).all().item(),'Nonfinite saved model')
        for opt in saved['optimizer']['state'].values():
            for value in opt.values():
                if isinstance(value,torch.Tensor): require(torch.isfinite(value).all().item(),'Nonfinite optimizer')
        done=Path(training)/f'seed_{seed}_complete.json'
        if done.exists():
            d=json.loads(done.read_text());require(d['contract_sha256']==c['parent_contract_sha256'],'Completion contract changed')
            require(sha(Path(training)/f'seed_{seed}_best.pt')==d['best_checkpoint_sha256'],'Best checkpoint changed')
        summaries.append({'seed':seed,'epoch':state['epoch'],'cursor':state['cursor'],'steps':state['steps'],
                          'fit_complete':done.exists(),'resume_sha256':sha(path),'elapsed_seconds':state['elapsed_seconds']})
    return summaries


def run_training(c, context):
    fd=int(os.environ['GRAYBOX_CONTINUATION_LEASE_FD'])
    require(os.fstat(fd).st_ino==LOCK_INODE,'Inherited lease missing')
    require(json.loads(os.environ['GRAYBOX_CONTINUATION_CONTEXT'])==context,'Launcher context differs')
    training, output = Path(context['training']),Path(context['output'])
    def check():
        guard(c,output,training,context['authorization'],context['authorization_sha256'],
              context['handoff'],context['handoff_sha256'],context['work_end'])
    check()
    frame,records,normalization,config,_=inspect_inputs(c,c['source_root'],Path(c['review_root'])/'initial_training')
    verify_initial_training(c,training)  # This adapter is intentionally one initial slot only.
    runtime={'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
             'python':sys.version,'device':torch.cuda.get_device_name(0)}
    require(runtime==c['runtime'],'Training runtime changed')
    torch.set_num_threads(config['cpu_threads'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    check()
    deadline=min(time.monotonic()+context['max_fitting_seconds'],context['work_end']-20)
    require(deadline>time.monotonic(),'No fitting time remains')
    train=ExistingAIA72Dataset(frame[frame.role.eq('train')],records,normalization,config['max_cached_images_per_worker'])
    selection=ExistingAIA72Dataset(frame[frame.role.eq('model_validation')],records,normalization,config['max_cached_images_per_worker'])
    base_loader=original.loader
    original.loader=partial(leased_loader,base_loader,fd)
    try:
        result=run_selected(train,selection,config,training,c['parent_contract_sha256'],deadline,check,
                            trainer=original.train_seed)
    finally:
        original.loader=base_loader
    check()
    result['checkpoints']=validate_outputs(c,training)
    result['initial_seed29_steps']=c['initial_seed29_state']['steps']
    result['additional_steps']=sum(r['steps'] for r in result['checkpoints'])-c['initial_seed29_state']['steps']
    result['original_training_loop_sha256']=c['inputs']['training_loop']['sha256']
    result['loader_amendment']='Only pass persistent lock descriptors to spawned loaders; generator/order/preprocessing unchanged'
    write_json(output/'training_result.json',result);check()
    return result


def execute_notebook(bundle,output,context):
    """Same supervised plain-Python engine as accepted replay; no detached kernel."""
    nb=nbformat.read(Path(bundle)/'notebooks/11_AIA_72h_Training_Continuation.ipynb',as_version=4)
    target=Path(output)/'11_AIA_72h_Training_Continuation_EXECUTED.ipynb'
    namespace={'__name__':'__continuation_notebook__','CONTINUATION_CONTEXT':context}
    for index,cell in enumerate(c for c in nb.cells if c.cell_type=='code'):
        cell.execution_count=index+1;cell.outputs=[]
        def display(value):
            data={'text/plain':str(value)}
            if isinstance(value,pd.DataFrame):data['text/html']=value.to_html(index=False)
            cell.outputs.append(nbformat.v4.new_output('display_data',data=data))
        namespace['display']=display
        class Tee(io.StringIO):
            def write(self,text):
                sys.__stdout__.write(text);sys.__stdout__.flush();return super().write(text)
        capture=Tee()
        try:
            with contextlib.redirect_stdout(capture):exec(compile(cell.source,str(target),'exec'),namespace)
        except BaseException as error:
            cell.outputs.append(nbformat.v4.new_output('error',ename=type(error).__name__,evalue=str(error),traceback=traceback.format_exc().splitlines()))
            raise
        finally:
            if capture.getvalue():cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=capture.getvalue()))
            temp=target.with_suffix('.tmp');nbformat.write(nb,temp);temp.replace(target)


if __name__=='__main__':
    fd=int(os.environ['GRAYBOX_CONTINUATION_LEASE_FD'])
    require(os.fstat(fd).st_ino==LOCK_INODE,'Inherited lock absent')
    os.set_inheritable(fd,False)
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    context=json.loads(os.environ['GRAYBOX_CONTINUATION_CONTEXT'])
    require(context['mode']=='authorized_training_continuation','Unapproved worker')
    execute_notebook(context['bundle_root'],context['output'],context)
