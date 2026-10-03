"""Prepare the first short continuation slot locally; no VM, pixels, allowance or GPU."""
import argparse
import inspect
import json
from pathlib import Path
import shutil
import tarfile
import nbformat as nbf
from scripts.aia72_replay_contract import sha,require,write_json
from scripts.aia72_continue_contract import ROOT,CONSUMED,LOCK,LOCK_INODE,PARENT,RESUME,RETURN,check_contract
from scripts.train_aia72_gpu import train_seed


def build_notebook(root):
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('''# 11 · Resume the AIA 72-hour experiment

## tl;dr
**Prepared continuation notebook. Preparation does not restart training.** Seed 17 is completed and its saved predictions reproduced. Seed 29 resumes epoch 2 after 18,256 cases; seed 43 begins only after seed 29 completes its original stopping rule.

## Context & Methods
The first separate allowance may grant **at most 65 minutes total**, with at most 60 minutes fitting and 60 seconds reserved for cleanup. The 12-hour planning proposal is **not an allowance**. All time from the new handoff—including held idle and setup—is charged. This notebook runs once per reviewed bundle; a later continuation requires the checked completion receipt and a new allowance/handoff.

The original CNN–GRU, float32 settings, batch 16, AdamW learning rate 0.0003, weight decay 0.0001, four loader workers, patience 4 and maximum 20 epochs remain unchanged. Preserve model, optimizer, RNG, epoch permutation/cursor, accumulated loss and selection history. Reuse the original training-only normalization.

### Key assumptions and limits
Candidate outcomes and historical data availability remain provisional. Training uses 2010–2013 and model selection uses January–June 2014. No later-year performance, calibration or fusion is evaluated. A time-capped checkpoint is incomplete, not a completed model. The first short slot cannot be assumed to finish either remaining seed.

### 1. Read the frozen configuration
Normal execution defaults to a CPU metadata preview. Authorized GPU execution requires the reviewed controller and its inherited persistent lock.''')
    code('''from pathlib import Path
import json, os, sys, time, math, random
import numpy as np
import pandas as pd
import torch
from torch import nn

if 'CONTINUATION_CONTEXT' not in globals():
    ROOT = next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'scripts/aia72_continue.py').is_file())
    archive = ROOT/'outputs/compute_limit_archive_20261003/snapshot'
    CONTINUATION_CONTEXT = {'mode':'preview','bundle_root':str(ROOT),
                            'preview_source_root':str(archive),'preview_initial':str(archive/'outputs/training')}
ROOT=Path(CONTINUATION_CONTEXT['bundle_root'])
sys.path.insert(0,str(ROOT))
import scripts.aia72_continue as continuation
import scripts.train_aia72_gpu as original
from scripts.aia72_continue_contract import check_contract
from scripts.aia72_replay_contract import require
from scripts.train_aia72_gpu import atomic_save,rng_state,restore_rng,set_seed,loader,validation_predictions
from scripts.aia72_vm_data import write_json
from scripts.dataset_io import file_sha256
CONTRACT=check_contract(json.loads((ROOT/'configs/aia72_continuation_v1.json').read_text()))
display(pd.DataFrame([{'Mode':CONTINUATION_CONTEXT['mode'],'First seed':29,'Following seed':43,
 'Maximum slot minutes':65,'Maximum fitting minutes':60,'Seed17 repeated':False}]))''')
    md('''## Data
### 2. Verify the saved starting point
Verify metadata and checkpoint hashes without reading any image pixels. The snapshot remains immutable; the live continuation writes to a separate copy.''')
    code('''preview=CONTINUATION_CONTEXT['mode']=='preview'
source=CONTINUATION_CONTEXT.get('preview_source_root',CONTRACT['source_root'])
initial=CONTINUATION_CONTEXT.get('preview_initial',str(Path(CONTRACT['review_root'])/'initial_training'))
frame,records,normalization,CONFIG,state=continuation.inspect_inputs(CONTRACT,source,initial)
display(pd.DataFrame([{'Seed':29,'Next epoch':state['epoch'],'Processed cases in epoch':state['cursor'],
                      'Optimizer steps':state['steps'],'Completed epochs':len(state['history'])}]))
display(frame.groupby('role',sort=False).agg(Cases=('forecast_case_id','size'),Positive_windows=('label','sum')).reset_index())
print('Saved normalization reused. Image pixels read in this cell: 0.')''')
    md('''### 3. Original model architecture
This executable definition is unchanged from the frozen training source.''')
    code((root/'scripts/aia72_model.py').read_text())
    md('''### 4. Original resumable training loop
The definition below is the unchanged original function. The worker calls that same hash-verified function from its frozen module. Its only loader wrapper passes the shared lock into spawned workers; sample order, generators, preprocessing and optimizer settings are unchanged. The original six-hour scheduler is not called; the new outer controller enforces the separately recorded short allowance.''')
    code(inspect.getsource(train_seed))
    md('''## Results
### 5. Resume only through the reviewed controller
The supervised process group contains the notebook, model and loaders. Stop, failure, disk, ownership and deadline checks apply throughout. No automatic retry or second invocation is permitted.''')
    code('''if CONTINUATION_CONTEXT['mode']=='authorized_training_continuation':
    result=continuation.run_training(CONTRACT,CONTINUATION_CONTEXT)
else:
    require(CONTINUATION_CONTEXT['mode']=='preview','Unknown mode')
    result={'status':'prepared_only_no_training','additional_steps':0,'scientific_acceptance':False}
display(pd.DataFrame([{'Item':key,'Value':result[key]} for key in ['status','additional_steps','scientific_acceptance']]))
if result.get('checkpoints'):
    display(pd.DataFrame(result['checkpoints'])[['seed','epoch','cursor','steps','fit_complete']])''')
    md('''## Takeaways
### 6. Preserve the status and return the GPU
The notebook result remains subject to controller integrity and cleanup checks. Keep the executed notebook, training checkpoint, final controller receipt and explicit handback together. Completing seed 29/43 still requires their saved-model verification before later scientific evaluation.''')
    code("print('Notebook status:',result['status'])\nprint('Later-year evaluation, calibration and fusion remain pending.')")
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
    for i,cell in enumerate(nb.cells):cell.id=f'aia72-continuation-{i:02d}'
    nbf.validate(nb);nbf.write(nb,root/'notebooks/11_AIA_72h_Training_Continuation.ipynb')


def prepare(root,archive,output):
    root,archive,output=map(lambda p:Path(p).resolve(),[root,archive,output])
    require(not output.exists(),'Preserve previous prepared bundle')
    proposal=json.loads((root/'results/aia72_training_continuation_proposal_20261003/continuation_compute_proposal.json').read_text())
    for relative,record in proposal['pinned_parent_files'].items():require(sha(archive/relative)==record['sha256'],'Parent file changed: '+relative)
    old=json.loads((archive/'outputs/training/run_contract.json').read_text())
    for name,digest in old['source_sha256'].items():require(sha(root/'scripts'/name)==digest,'Original implementation changed: '+name)
    paths={'cases':'inputs/fit_cases.csv.gz','configuration':'configs/aia72_temporal_v1.json',
           'run_contract':'outputs/training/run_contract.json','normalization':'outputs/training/normalization.json',
           'sources':'outputs/training/bound_sources.json','training_loop':'scripts/train_aia72_gpu.py',
           'checkpoint29':'outputs/training/seed_29_resume.pt','complete17':'outputs/training/seed_17_complete.json',
           'best17':'outputs/training/seed_17_best.pt'}
    manifest=json.loads((archive/'snapshot_manifest.json').read_text())['files']
    initial={}
    for p in sorted((archive/'outputs/training').glob('*')):
        require(p.is_file() and sha(p)==manifest[str(p.relative_to(archive))]['sha256'],'Archived training artifact changed')
        initial[p.name]={'sha256':sha(p),'bytes':p.stat().st_size}
    c={'kind':'aia72_first_training_continuation_slot','review_root':ROOT,'consumed_root':CONSUMED,
       'source_root':proposal['source_root'],'owner_pointer':proposal['coordination']['current_owner_pointer'],
       'common_lock':LOCK,'common_lock_inode':LOCK_INODE,'parent_contract_sha256':PARENT,
       'initial_seed29_sha256':RESUME,'initial_seed29_state':proposal['seed29_resume']['state'],
       'prior_return_path':'/home/abmoses2000/graybox_aia72_seed17_replay_v1_20261003/handoff_release_to_aia_after_replay_20261003.json',
       'prior_return_sha256':RETURN,'seeds':[29,43],'maximum_slot_seconds':3900,'maximum_fitting_seconds':3600,
       'cleanup_seconds':60,'invocation_count':1,'automatic_retry':False,'minimum_free_bytes':20*1024**3,
       'runtime':old['runtime'],'original_cumulative_fitting_limit_seconds':21600,
       'external_allowance_policy':'One short extension only. The12hour proposal is not granted. Time is charged from fresh handoff UTC, including held idle/setup/checkpoint/cleanup. No automatic second invocation.',
       'inputs':{key:{'path':relative,'sha256':sha(archive/relative)} for key,relative in paths.items()},
       'original_source_sha256':old['source_sha256'],'initial_training_files':initial,
       'loader_change':'Pass duplicated persistent-lock descriptors to spawned workers; no random, sample-order, precision, preprocessing or model changes.',
       'scientific_acceptance':False}
    check_contract(c);write_json(root/'configs/aia72_continuation_v1.json',c);build_notebook(root)
    bundle=output/'bundle';bundle.mkdir(parents=True)
    code=['aia72_continue.py','aia72_continue_contract.py','run_aia72_continue.py','aia72_replay_contract.py','run_aia72_replay.py',
          'train_aia72_gpu.py','aia72_model.py','aia72_vm_data.py','aia72_data.py','aia_io.py','dataset_io.py']
    names=['scripts/'+name for name in code]+['configs/aia72_continuation_v1.json','notebooks/11_AIA_72h_Training_Continuation.ipynb']
    for name in names:
        dest=bundle/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest)
    write_json(bundle/'continuation_bundle_manifest.json',{'files':{name:sha(bundle/name) for name in names}})
    shutil.copytree(archive/'outputs/training',output/'initial_training')
    mirror=output/'source_mirror'
    for relative in paths.values():
        dest=mirror/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(archive/relative,dest)
    with tarfile.open(output/'source_bundle.tar.gz','x:gz') as tar:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():tar.add(path,arcname=str(path.relative_to(output)))
    record={'status':'prepared_pending_CPU_and_source_review','bundle':str(bundle),'contract_sha256':sha(bundle/'configs/aia72_continuation_v1.json'),
            'bundle_sha256':sha(bundle/'continuation_bundle_manifest.json'),'source_tar_sha256':sha(output/'source_bundle.tar.gz'),
            'source_tar_bytes':(output/'source_bundle.tar.gz').stat().st_size,'GPU_launch':False,'new_allowance':False,'seed17_repeated':False}
    write_json(output/'preparation.json',record);return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(Path(__file__).resolve().parents[1],args.archive,args.output),indent=2))
