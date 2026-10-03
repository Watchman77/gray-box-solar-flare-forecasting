"""Build the inference-only notebook; it defaults to a CPU metadata preview."""
import inspect
from pathlib import Path
import nbformat as nbf
from scripts.aia72_replay import infer_rows


def build():
    root = Path(__file__).resolve().parents[1]
    cells = []
    def md(text): cells.append(nbf.v4.new_markdown_cell(text))
    def code(text): cells.append(nbf.v4.new_code_cell(text))
    md('''# 10 · AIA 72-hour saved-checkpoint replay

## tl;dr
**Prepared verification notebook. No scientific GPU replay has been run by preparing this file.**
The completed seed-17 model selected epoch 2. This review will restore that exact checkpoint and compare all 3,905 earlier validation predictions with the saved logits. It performs no fitting, tuning, new model selection, calibration or later-period evaluation.

## Context & Methods
Use the same frozen cases, saved training-only normalization, CNN–GRU architecture, float32 settings and batch size 16. The predeclared logit rule is `abs(replayed − saved) ≤ 1e-6 + 1e-5 × abs(saved)` for **every case**; changed or missing inputs stop the review.

### Key assumptions and limits
- This is a technical reproducibility check of serialization and inference on an already used selection block. It is not independent scientific validation or evidence of operational forecast skill.
- The model and preprocessing implementation are reused. Source-file hashes and cached image bytes are checked, but shared implementation errors may remain undetected.
- A fresh review-specific GPU handoff and separate allowance are required. The exhausted six-hour fitting ceiling remains unchanged.
- The proposed slot is at most 1,800 seconds, including loading, output and cleanup, with 60 seconds reserved for cleanup. Any failed or incomplete review stays unverified, with no automatic retry.
- No new images, environment changes, full-disk cache scans or additional GPU jobs are required.

### 1. Inspect the contract
Opening or running the notebook normally uses metadata-preview mode. Scientific execution is only through the reviewed launcher with its inherited GPU lock. The supervised runner executes plain Python cells in one process group, so it can terminate the model and loading workers together.''')
    code('''from pathlib import Path
import json, os, sys, time
import numpy as np
import pandas as pd
import torch

if 'REPLAY_CONTEXT' not in globals():
    candidates = [Path.cwd(), *Path.cwd().parents]
    root = next(p for p in candidates if (p / 'scripts/aia72_replay.py').is_file())
    REPLAY_CONTEXT = {'mode': 'preview', 'bundle_root': str(root)}
ROOT = Path(REPLAY_CONTEXT['bundle_root']).resolve()
sys.path.insert(0, str(ROOT))
import scripts.aia72_replay as replay
from scripts.aia72_replay_contract import check_contract, verify_inputs, require
from scripts.aia72_replay import before_deadline
CONTRACT = check_contract(json.loads((ROOT / 'configs/aia72_seed17_replay_v1.json').read_text()))
display(pd.DataFrame([{'Mode': REPLAY_CONTEXT['mode'], 'Seed': CONTRACT['seed'],
                      'Selected epoch': CONTRACT['selected_epoch'], 'Cases': CONTRACT['case_count'],
                      'Maximum slot (minutes)': CONTRACT['slot_seconds'] / 60,
                      'Fitting allowed': CONTRACT['fitting_allowed']}]))''')
    md('''## Data
### 2. Verify pinned metadata and exact support
This preview reads local metadata and checkpoint hashes only. It does not read cached images or initialize CUDA. Saved normalization is reused; no statistics are fitted.''')
    code('''source_root = REPLAY_CONTEXT.get('preview_source_root') if REPLAY_CONTEXT['mode'] == 'preview' else None
paths = verify_inputs(CONTRACT, source_root)
selected, reference, records, normalization = replay.load_support(CONTRACT, paths)
display(pd.DataFrame([{'Frozen validation cases': len(selected), 'Positive windows': int(selected.label.sum()),
                      'Distinct image objects': len(records), 'Saved normalization fit role': normalization['fit_role'],
                      'Image pixels read in this cell': 0}]))''')
    md('''### 3. Inspect the restored architecture
These classes are unchanged from the completed fit. The selected weight file is loaded only during authorized replay.''')
    code((root / 'scripts/aia72_model.py').read_text())
    md('''### 4. Inspect the inference loop
Evaluation mode disables dropout and running-statistic updates. The loop checks all IDs, labels, finite logits and unchanged model parameters/buffers. There is no optimizer or training call.''')
    code(inspect.getsource(infer_rows) + '\nreplay.infer_rows = infer_rows\nreplay.TemporalAIACNNGRU = TemporalAIACNNGRU')
    md('''## Results
### 5. Run only with the approved launcher
The launcher checks the fresh allowance and handoff before starting this process. Preview mode reports preparation status without generating predictions.''')
    code('''if REPLAY_CONTEXT['mode'] == 'authorized_gpu_replay':
    import fcntl
    from scripts.aia72_replay_contract import LOCK_INODE
    require(json.loads(os.environ['GRAYBOX_REPLAY_CONTEXT']) == REPLAY_CONTEXT, 'Launcher context missing')
    lease_fd = int(os.environ['GRAYBOX_REPLAY_LEASE_FD'])
    require(os.fstat(lease_fd).st_ino == LOCK_INODE, 'Shared lease missing')
    fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result = replay.run_replay(CONTRACT, REPLAY_CONTEXT['output'], REPLAY_CONTEXT['deadline'], REPLAY_CONTEXT['owner_sha256'])
else:
    require(REPLAY_CONTEXT['mode'] == 'preview', 'Unknown execution mode')
    result = {'status': 'prepared_only_no_GPU_replay', 'cases_replayed': 0,
              'scientific_acceptance': False, 'fitting_steps': 0}
shown = ['status', 'cases', 'cases_replayed', 'mismatched_cases', 'max_absolute_logit_difference',
         'saved_log_loss', 'replayed_log_loss', 'fitting_steps', 'scientific_acceptance']
display(pd.DataFrame([{'Check': name, 'Value': result[name]} for name in shown if name in result]))''')
    md('''## Takeaways
### 6. Read the execution status literally
A metadata preview or successful synthetic test does not verify the scientific model. A completed comparison is still pending the controller's source recheck, process cleanup and final receipt. Keep the prediction comparison, executed notebook and controller receipt together. Later-period validation and completion of seeds 29 and 43 remain separate work.''')
    code("print('Notebook status:', result['status'])\nprint('Scientific acceptance: False. No new fitting or later-period evaluation.')")
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}})
    for index, cell in enumerate(nb.cells):
        cell.id = f'aia72-replay-{index:02d}'
    nbf.validate(nb)
    path = root / 'notebooks/10_AIA_72h_Checkpoint_Replay.ipynb'
    nbf.write(nb, path)
    return path


if __name__ == '__main__':
    print(build())
