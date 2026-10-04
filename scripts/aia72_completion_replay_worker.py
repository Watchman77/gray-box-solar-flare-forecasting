"""Execute the replay notebook inside one supervised process and inherited lease."""
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

import nbformat
import numpy as np
import pandas as pd
import torch

from scripts.aia72_continue import leased_loader
from scripts.aia72_completion_replay import run_completion_replay
from scripts.aia72_completion_replay_contract import check_contract, guard
from scripts.aia72_replay_contract import LOCK_INODE, require
import scripts.train_aia72_gpu as original


def run(c, context):
    check_contract(c)
    require(json.loads(os.environ['GRAYBOX_COMPLETION_REPLAY_CONTEXT']) == context, 'Controller context differs')
    fd = int(os.environ['GRAYBOX_COMPLETION_REPLAY_LEASE_FD'])
    require(os.fstat(fd).st_ino == LOCK_INODE, 'Inherited lease changed')
    def check():
        guard(c,context['output'],context['authorization'],context['authorization_sha256'],
              context['handoff'],context['handoff_sha256'],context['work_end']-20)
    check()
    runtime = {'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
               'python':sys.version,'device':torch.cuda.get_device_name(0)}
    require(runtime == c['runtime'], 'Exact training runtime required')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    # Reuse the original ordered inference loader, with the already tested lease
    # wrapper. No fitting function or optimizer is invoked.
    config={'batch_size':16,'num_workers':4}
    factory=lambda dataset,seed:leased_loader(original.loader,fd,dataset,config,seed)
    result=run_completion_replay(c,Path(context['output'])/'replay',context['work_end']-20,factory,check)
    check()
    return result


def execute_notebook(bundle, output, context):
    nb=nbformat.read(Path(bundle)/'notebooks/16_AIA_72h_Completed_Model_Replay.ipynb',as_version=4)
    target=Path(output)/'16_AIA_72h_Completed_Model_Replay_EXECUTED.ipynb'
    namespace={'__name__':'__completed_replay_notebook__','REPLAY_CONTEXT':context}
    for index,cell in enumerate(c for c in nb.cells if c.cell_type=='code'):
        cell.execution_count=index+1;cell.outputs=[]
        def display(value):
            data={'text/plain':str(value)}
            if isinstance(value,pd.DataFrame):data['text/html']=value.to_html(index=False)
            cell.outputs.append(nbformat.v4.new_output('display_data',data=data))
        namespace['display']=display
        capture=io.StringIO()
        try:
            with contextlib.redirect_stdout(capture):
                exec(compile(cell.source,str(target),'exec'),namespace)
        except BaseException as error:
            cell.outputs.append(nbformat.v4.new_output('error',ename=type(error).__name__,evalue=str(error),
                                                       traceback=traceback.format_exc().splitlines()))
            raise
        finally:
            if capture.getvalue():
                cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=capture.getvalue()))
                sys.__stdout__.write(capture.getvalue());sys.__stdout__.flush()
            nb.metadata['execution_engine']='supervised in-process plain Python; no detached Jupyter kernel'
            temporary=target.with_suffix('.tmp');nbformat.write(nb,temporary);temporary.replace(target)


if __name__=='__main__':
    fd=int(os.environ['GRAYBOX_COMPLETION_REPLAY_LEASE_FD'])
    require(os.fstat(fd).st_ino==LOCK_INODE,'Missing inherited persistent lease')
    os.set_inheritable(fd,False)
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    context=json.loads(os.environ['GRAYBOX_COMPLETION_REPLAY_CONTEXT'])
    require(context['mode']=='authorized_completed_model_replay','Unsupported worker context')
    execute_notebook(context['bundle_root'],context['output'],context)
