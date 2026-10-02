"""Create an inspectable, executable notebook from the bounded real-data check."""

import inspect
from pathlib import Path
import textwrap

import nbformat as nbf
from scripts.run_aia72_canary import run


def build():
    root = Path(__file__).resolve().parents[1]
    cells = []
    def md(value):
        cells.append(nbf.v4.new_markdown_cell(value))
    def code(value):
        cells.append(nbf.v4.new_code_cell(value))

    md('''# 08 · AIA 72-hour loader, training-step and checkpoint check

## tl;dr
This is a **technical check, not the full AIA experiment**. It loads the first and last eligible cases in each of the seven frozen roles: 14 cases and 42 images. It tests both image metadata formats, trains fresh weights for two steps on the two training-role cases, and verifies exact replay after saving and reloading. Observed results appear below.

The checkpoint and normalization from this notebook must **not** initialize the scientific experiment. No forecast-skill, fusion improvement or uncertainty coverage is measured here.''')
    md('''## Context & Methods
We reuse Bamidele's temporal CNN–GRU architecture with a fresh 72-hour target and the roles from Notebook 07. Images contain upstream processed pixel values; they are not raw physical radiances. The manifest supplies the 72-hour outcome. Embedded 48-hour labels are ignored.

### Key assumptions
- Selection uses issue times, not outcome classes. Two boundary cases per role are deliberately a software check, not a representative evaluation sample.
- Each frame's pinned generation, SHA-256, embedded identity, channel order and finite values are checked. Nominal history times precede issue time. Actual channel observation times and historical delivery are not certified.
- Six channels are transformed with training-only asinh scaling and standardization, then resized from 512 to 256 pixels using bilinear interpolation. Only the two technical training cases fit these temporary statistics.
- All other roles are forward-pass checks only. No label-based accuracy is computed.
- This local CPU notebook requires the already staged small image sample. It does not start cloud resources or change other jobs.

### 1. Configuration and pinned inputs
Run inside the repository with `requirements-training-lock.txt`. The bounded acquisition helper is `scripts/stage_aia72_canary.py`; its saved plan and receipt identify the input objects. Reruns use a new output directory.''')
    code('''from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import display, Markdown

root = Path.cwd().resolve()
if not (root / 'scripts/aia72_data.py').is_file():
    root = root.parent
assert (root / 'scripts/aia72_data.py').is_file(), 'Run inside the Gray-Box repository'
sys.path.insert(0, str(root))
from scripts.aia72_data import AIA72Dataset, LAGS, fit_normalization, transform_image, validate_frame_identity
from scripts.aia_io import CHANNELS, load_aia_frame
from scripts.dataset_io import file_sha256
from scripts.prepare_multimodal72 import write_json
from scripts.run_aia72_canary import forward, state_digest
stage = root / 'data/raw/aia72_canary_v1_20261002'
output = root / 'outputs' / ('aia72_canary_notebook_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
print('Output:', output)
pd.set_option('display.max_colwidth', 65)''')
    md('''### 2. Fresh temporal CNN–GRU
The executable architecture is reproduced here for inspection. Its classes are unchanged from the recorded upstream notebook; source provenance is in `results/aia72_canary_20261002/source_provenance.json`. It takes three frames × six channels and returns one 72-hour logit.''')
    code((root / 'scripts/aia72_model.py').read_text())

    body = textwrap.dedent('\n'.join(inspect.getsource(run).splitlines()[1:]))
    body = body[:body.rindex('return result, cases, checks, normalization, training_rows')]
    sections = [
        ('## Data\n### 3. Check the selected cases and image bytes', 'root, stage, output ='),
        ('### 4. Fit preprocessing on the two training cases only', 'normalization = fit_normalization'),
        ('## Results\n### 5. Two technical optimization steps\nThese steps test finite loss, gradients and parameter updates. They are not an estimate of forecast quality.', 'for step in range(2):'),
        ('### 6. Save and replay every selected case\nEvaluation must leave weights and batch-normalization buffers unchanged. Reloaded logits must agree exactly with identical batching.', 'logits = forward(model, loader)'),
        ('### 7. Independently express the channel transformation\nCompare the broadcast implementation against a channel-by-channel calculation on every checked object.', '# Channelwise independent expression'),
        ('## Takeaways\n### 8. Save and summarize the executed technical evidence', 'result = {'),
    ]
    starts = [body.index(marker) for _, marker in sections] + [len(body)]
    for i, (heading, _) in enumerate(sections):
        md(heading)
        source = body[starts[i]:starts[i+1]].strip()
        extras = {
            0: "\ndisplay(cases.groupby('role', sort=False).size().rename('Checked cases').to_frame())\ndisplay(checks.groupby('metadata_schema').size().rename('Verified images').to_frame())",
            1: "\ndisplay(pd.DataFrame({'Channel': CHANNELS, 'Scale': normalization['channel_scale'], 'Mean': normalization['channel_mean'], 'Std': normalization['channel_std']}))",
            2: "\ndisplay(pd.DataFrame(training_rows).drop(columns='case_ids'))",
            3: "\nprint('Saved-model replay: exact for all 14 cases; evaluation changed no model state.')",
            4: "\nprint('Maximum independent transformation difference:', maximum_difference)",
            5: "\ndisplay(pd.DataFrame([{'Cases': result['cases'], 'Verified images': result['objects_verified'], 'Parameters': result['model_parameters'], 'Technical steps': result['technical_optimization_steps'], 'Seconds': round(result['seconds'], 2), 'Full training complete': result['full_72h_training_complete']}]))\ndisplay(Markdown('**Passed: loader, two training steps and exact checkpoint replay. Full 72-hour training remains pending.**'))\nprint('Receipt:', output / 'summary.json')",
        }
        code(source + extras[i])
    md('''The next experiment must use all frozen training cases, fresh full-training preprocessing and fresh weights. Benchmark the selected GPU before estimating time; save resumable checkpoints and preserve the reserved calibration and evaluation roles. Reusing the existing VM requires coordination with its active AIA work. No additional GPU resource was started by this notebook.''')
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
    })
    nbf.validate(nb)
    path = root / 'notebooks/08_AIA_72h_Loader_and_Model_Check.ipynb'
    nbf.write(nb, path)
    print(path)


if __name__ == '__main__':
    build()
