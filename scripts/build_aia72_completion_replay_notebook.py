"""Build the visible replay notebook; ordinary execution is metadata preview only."""
from pathlib import Path
import inspect
import nbformat
from scripts.aia72_replay import infer_rows
from scripts.aia72_completion_replay import selection_ensemble


def build(path=None):
    root=Path(__file__).resolve().parents[1]
    path=Path(path or root/'notebooks/16_AIA_72h_Completed_Model_Replay.ipynb')
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    nb=nbformat.v4.new_notebook(cells=[
        md('''# 16 · Check the completed AIA 72-hour models

## tl;dr
**Preparation is not completed replay.** This notebook checks saved models 29 and 43 against their original model-selection predictions. Model 17's accepted check is reused. The three-seed ensemble uses every seed; no seed or epoch is selected from later data.

## Context & Methods
The comparison uses the original 3,905 January–June 2014 selection cases, original train-only normalization, six channels and three histories. Predictions must match within the predeclared `atol=1e-6, rtol=1e-5`. Batches contain 16 cases, with four loader workers retaining the persistent lease. A reviewed invocation is bounded to 30 minutes including setup, with 60 seconds reserved for cleanup. Actual work stops promptly after completion.

### Key assumptions and limits
This tests saved-model reproducibility using the same architecture/preprocessing implementation. It is not independent scientific validation, later-period evaluation, calibration or fusion. Labels and historical availability remain provisional.

### 1. Setup and exact input binding
Use the repository and its pinned artifacts. Dependencies are NumPy, pandas, PyTorch and nbformat; GPU execution additionally requires the exact recorded VM runtime. Normal execution previews metadata. The reviewed controller supplies GPU context after completed-training verification.'''),
        code('''from pathlib import Path
import json, sys, time
import numpy as np
import pandas as pd
import torch

if 'REPLAY_CONTEXT' not in globals():
    candidates=[Path.cwd(),*Path.cwd().parents]
    ROOT=next(p for p in candidates if (p/'scripts/aia72_completion_replay.py').is_file())
    REPLAY_CONTEXT={'mode':'preview','bundle_root':str(ROOT)}
ROOT=Path(REPLAY_CONTEXT['bundle_root'])
sys.path.insert(0,str(ROOT))
from scripts.aia72_replay_contract import require
from scripts.aia72_replay import before_deadline
from scripts.aia72_completion_replay import load_completed_support
from scripts.aia72_completion_replay_contract import check_contract
configuration=ROOT/'configs/aia72_completion_replay_v1.json'
CONTRACT=check_contract(json.loads(configuration.read_text())) if configuration.is_file() else None
display(pd.DataFrame([{'Mode':REPLAY_CONTEXT['mode'],'Inputs pinned':CONTRACT is not None,
                      'New model checks':'29,43','Prior check reused':17,'Maximum reservation minutes':30}]))'''),
        md('''## Data
### 2. Verify completed models and preserved support
The metadata check hashes inputs and verifies selected epochs, case identities, labels, normalization support and the first model's accepted receipt. It does not read image pixels. Missing final checkpoints remain a visible preparation dependency.'''),
        code('''if CONTRACT is None:
    print('Final input binding pending completed training and its verified archive. No image pixels or GPU model run.')
else:
    source=REPLAY_CONTEXT.get('preview_source_root',CONTRACT['source_root'])
    paths,selected,references,completions,records,normalization=load_completed_support(CONTRACT,source)
    display(pd.DataFrame([{'Seed':seed,'Selected epoch':completions[seed]['best_epoch'],
                          'Selection cases':len(references[seed]),'Replay needed':seed!=17}
                         for seed in [17,29,43]]))
    print('Completed-model inputs and earlier-selection support verified. Image pixels read: 0.')'''),
        md('''### 3. Read-only prediction loop
This is the executable inference function used by the worker's hash-verified module. It checks ordered identities and labels, unchanged parameters/buffers, the deadline and resource guards. It constructs no optimizer.'''),
        code(inspect.getsource(infer_rows)),
        md('''### 4. Keep every seed in the ensemble
After successful checks, average the three saved probability vectors on identical cases. Averaging logits or dropping an inconvenient seed would change the prespecified experiment.'''),
        code(inspect.getsource(selection_ensemble)),
        md('''## Results
### 5. Execute only under the reviewed controller
The model and loader processes share the inherited persistent lock. Any input change, prediction mismatch, stop or incomplete cleanup prevents an accepted result. A notebook preview is not GPU execution.'''),
        code('''if REPLAY_CONTEXT['mode']=='authorized_completed_model_replay':
    require(CONTRACT is not None,'Final inputs are not pinned')
    from scripts.aia72_completion_replay_worker import run
    result=run(CONTRACT,REPLAY_CONTEXT)
else:
    require(REPLAY_CONTEXT['mode']=='preview','Unknown notebook mode')
    result={'status':'metadata_preview_no_scientific_replay','fitting_steps':0,
            'scientific_acceptance':False,'comparisons':[]}
display(pd.DataFrame([{'Status':result['status'],'Fitting steps':result['fitting_steps'],
                      'Independent scientific validation':result['scientific_acceptance']}]))
if result['comparisons']:
    display(pd.DataFrame(result['comparisons'])[['seed','best_epoch','cases','mismatched_cases','max_absolute_logit_difference']])'''),
        md('''## Takeaways
### 6. Preserve the exact status
The final controller receipt establishes technical acceptance and cleanup. The executed notebook, per-case comparisons and three-seed ensemble stay together. Gray-Box scheduling priority continues through the remaining later-period and fusion analysis.'''),
        code('''print('Recorded notebook status:',result['status'])
print('Later AIA inference, probability calibration, uncertainty and SHARP/AIA fusion are separate phases.')''')
    ],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                'language_info':{'name':'python'}})
    nbformat.validate(nb)
    path.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(nb,path)
    return path


if __name__=='__main__':
    print(build())
