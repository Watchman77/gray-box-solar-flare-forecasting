"""Run the user-facing prediction notebook under the inherited exclusive lease."""
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import sys
import traceback

import nbformat
import numpy as np
import pandas as pd
import torch

from scripts.aia72_inference import run_inference
from scripts.aia72_inference_contract import check_contract,guard
from scripts.aia72_replay_contract import require,LOCK_INODE


def run(c,context):
    check_contract(c)
    require(json.loads(os.environ['GRAYBOX_INFERENCE_CONTEXT'])==context,'Context changed')
    fd=int(os.environ['GRAYBOX_INFERENCE_LEASE_FD'])
    require(os.fstat(fd).st_ino==LOCK_INODE,'Inherited lease changed')
    def check():
        guard(c,context['output'],context['authorization'],context['authorization_sha256'],
              context['handoff'],context['handoff_sha256'],context['work_end']-20)
    check()
    runtime={'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
             'python':sys.version,'device':torch.cuda.get_device_name(0)}
    require(runtime==c['runtime'],'Frozen runtime changed')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    return run_inference(c,Path(context['output'])/'predictions',fd,check)


def execute_notebook(context):
    source=Path(context['bundle_root'])/'notebooks/17_AIA_72h_Frozen_Model_Inference.ipynb'
    target=Path(context['output'])/'17_AIA_72h_Frozen_Model_Inference_EXECUTED.ipynb'
    nb=nbformat.read(source,as_version=4)
    namespace={'__name__':'__inference_notebook__','INFERENCE_CONTEXT':context}
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
            cell.outputs.append(nbformat.v4.new_output('error',ename=type(error).__name__,evalue=str(error),
                                                       traceback=traceback.format_exc().splitlines()))
            raise
        finally:
            if capture.getvalue():cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=capture.getvalue()))
            nb.metadata['execution_engine']='supervised in-process plain Python; no detached kernel'
            temporary=target.with_suffix('.tmp');nbformat.write(nb,temporary);temporary.replace(target)


if __name__=='__main__':
    fd=int(os.environ['GRAYBOX_INFERENCE_LEASE_FD']);require(os.fstat(fd).st_ino==LOCK_INODE,'Inherited lease missing')
    os.set_inheritable(fd,False);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    execute_notebook(json.loads(os.environ['GRAYBOX_INFERENCE_CONTEXT']))
