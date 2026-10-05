"""Visible notebook for saved-model inference; ordinary execution previews only."""
import inspect
from pathlib import Path
import nbformat
from scripts.aia72_inference import predict_block


def build():
    root=Path(__file__).resolve().parents[1]
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    nb=nbformat.v4.new_notebook(cells=[
        md('''# 17 · Apply the frozen AIA 72-hour models

## tl;dr
The three AIA models have finished training and saved-model verification. **This notebook prepares their application to the full frozen comparison population; preparation is not completed inference.** All 3,905 accepted earlier-selection predictions are reused without repeated GPU work. The remaining 92,691 cases retain their original roles and order.

## Context & Methods
Use the saved best models from seeds 17, 29 and 43, with original training-only normalization, six AIA channels, three histories and 16-case batches. Load each image batch once and evaluate all three fixed models. Average probabilities, not logits. Calibration, conformal methods and fusion are separate subsequent analyses.

### Key Assumptions
Outcomes are candidate labels and historical availability is unverified. Later-period results are retrospective. No model, threshold or preprocessing fitting occurs here. Missing/invalid input rows remain in outputs with no probability. Authentication or storage-service failures stop the job.

### 1. Configuration
Normal local execution is a metadata preview. The reviewed controller supplies the lease and exact runtime for GPU execution. The four-hour reservation is a safety ceiling, not an estimated runtime; completed work returns resources immediately.'''),
        code('''from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import torch
if 'INFERENCE_CONTEXT' not in globals():
    ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'scripts/aia72_inference.py').is_file())
    INFERENCE_CONTEXT={'mode':'preview','bundle_root':str(ROOT)}
ROOT=Path(INFERENCE_CONTEXT['bundle_root']);sys.path.insert(0,str(ROOT))
from scripts.aia72_replay_contract import require,sha
from scripts.aia72_inference_contract import check_contract
from scripts.aia72_inference import SEEDS
config=ROOT/'configs/aia72_inference_v1.json'
CONTRACT=check_contract(json.loads(config.read_text())) if config.exists() else None
display(pd.DataFrame([{'Mode':INFERENCE_CONTEXT.get('mode','reviewed_GPU_inference'),
                      'Saved models':'17,29,43','Fitting allowed':False,'Case population':96596}]))'''),
        md('''## Data
### 2. Frozen inputs and resource bounds
The input table, role blocks, cloud object generations/checksums, live cache inventory and saved models are hash-bound. Existing files remain intact. Restore missing images into a new owned block directory, at most 14 GiB, while keeping at least 20 GiB free on both disks. Restored images remain independently recoverable from their pinned cloud versions.'''),
        code('''if CONTRACT is None:
    print('Exact execution package not bound in this preview; no images read and no GPU work.')
else:
    source=Path(INFERENCE_CONTEXT.get('preview_source_root',CONTRACT['source_root']))
    for record in CONTRACT['inputs'].values():
        require(sha(source/record['path'])==record['sha256'],'Input hash mismatch')
    cases=pd.read_csv(source/'cases.csv.gz',low_memory=False)
    display(cases.groupby('role',sort=False).size().rename('Cases').reset_index())
    print('Frozen input hashes verified; no image bytes read.')'''),
        md('''### 3. Prediction method
The executable function below is also used by the worker module. Each row retains its identity, role, outcome, last observation, all three logits/probabilities and input status. Models remain in evaluation mode with fixed parameters and buffers.'''),
        code(inspect.getsource(predict_block)),
        md('''## Results
### 4. Run the reviewed prediction phase
The controller supervises the notebook and its loader processes using the original persistent lock. Results and source receipts are retained per block. Calibration/conformal/policy blocks run first, followed by later evaluation periods, then training-period diagnostic predictions.'''),
        code('''if INFERENCE_CONTEXT.get('mode')=='preview':
    result={'status':'metadata_preview_only','inferred_cases':0,'fitting_steps':0,'blocks':[]}
else:
    require(CONTRACT is not None,'Exact inference package missing')
    from scripts.aia72_inference_worker import run
    result=run(CONTRACT,INFERENCE_CONTEXT)
display(pd.DataFrame([{'Status':result['status'],'Cases inferred in this run':result['inferred_cases'],
                      'Fitting steps':result['fitting_steps']}]))
if result['blocks']:
    display(pd.DataFrame(result['blocks']).groupby('role',sort=False).agg(Blocks=('block_id','size'),Cases=('cases','sum')).reset_index())'''),
        md('''## Takeaways
### 5. Preserve the outcome and limits
The final controller receipt and collected block outputs establish execution status. Fitting calibration/conformal thresholds and evaluating SHARP/AIA fusion remain subsequent work. Inspect input failures on the full population before interpreting matched-case performance.'''),
        code("print('Recorded status:',result['status'])\nprint('Scientific acceptance requires the subsequent matched evaluation; GPU inference alone does not establish it.')")
    ],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
    nbformat.validate(nb)
    path=root/'notebooks/17_AIA_72h_Frozen_Model_Inference.ipynb';nbformat.write(nb,path)
    return path


if __name__=='__main__':print(build())
