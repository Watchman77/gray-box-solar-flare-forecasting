"""Build the shared-VM AIA experiment notebook with visible fitting code."""
import inspect
from pathlib import Path
import nbformat as nbf
from scripts.train_aia72_gpu import train_seed


def build():
    root = Path(__file__).resolve().parents[1]; cells = []
    def md(text): cells.append(nbf.v4.new_markdown_cell(text))
    def code(text): cells.append(nbf.v4.new_code_cell(text))
    md('''# 09 · AIA 72-hour training on the shared L4

## tl;dr
**Prepared training notebook; consult the saved execution receipt for actual progress.** This fits fresh temporal image models on 25,586 frozen 2010–2013 cases and selects checkpoints using 3,905 cases from the first half of 2014. It uses three fixed seeds and reads existing VM images. No 48-hour model weights, later calibration labels or Cycle-25 evaluation data enter training.

This is an exploratory candidate-label experiment. A fitted model is not evidence of operational readiness. Checkpoint replay, probability calibration, uncertainty assessment, matched SHARP–AIA fusion, independent AR/event sensitivities and later-period inference remain subsequent stages.''')
    md('''## Context & Methods
Use the previously checked CNN–GRU architecture, 256-pixel float32 images and training-only asinh normalization. The optimizer is AdamW, learning rate 0.0003, weight decay 0.0001, batch size 16. Minimize unweighted binary cross-entropy, selecting the lowest earlier validation log loss separately for seeds 17, 29 and 43. Stop after four non-improving epochs or twenty epochs. All seeds are retained.

### Key assumptions
- The frozen split and 72-hour manifest outcomes are authoritative; NPZ outcome fields are ignored. Missing required files cause a stop rather than a silent reduction in cases.
- Original NPZ images are read directly, checking pinned cloud generation metadata, size, MD5 and image identity/schema. A small per-worker RAM cache avoids repeated work without creating a large second image archive.
- The source values are upstream processed AIA images, not physical radiances. Nominal history times are past-only; actual channel availability and complete negative-event coverage remain unverified.
- Each invocation is bounded to one hour of fitting, with a six-hour cumulative fitting ceiling. Checkpoints include model, optimizer, sample cursor, RNG state and code/data/configuration identity. A stopped invocation is incomplete, not a completed experiment.
- Use the agreed existing GPU lock, require explicit handoff release, and preserve at least 20 GiB free on both disks. Never change the shared environment, delete its cached data, or run a second GPU job.

### 1. Locate the immutable execution bundle
Launch through `scripts/run_aia72_training_notebook.py` after the AIA chat releases the VM. The runner pins the bundle and holds the common GPU lock. Do not run this notebook directly outside that runner.''')
    code('''from pathlib import Path
from datetime import datetime, timezone
import os, sys, json, time, math, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from IPython.display import display, Markdown

ROOT = Path.cwd().resolve()
assert os.environ.get('GRAYBOX_SHARED_GPU_LEASE') == str(ROOT), 'Use the reviewed shared-VM runner'
sys.path.insert(0, str(ROOT))
from scripts.dataset_io import file_sha256
from scripts.aia72_vm_data import bind_inputs, prepare_normalization, write_json
from scripts.train_aia72_gpu import (atomic_save, rng_state, restore_rng, set_seed, loader,
                                    validation_predictions, run_training)
CONFIG = json.loads((ROOT / 'configs/aia72_temporal_v1.json').read_text())
OUTPUT = ROOT / 'outputs/training'
OUTPUT.mkdir(parents=True, exist_ok=True)
INPUTS = ROOT / 'inputs'
INVENTORY = Path('/home/abmoses2000/aia19_cycle24_cache_prep/cycle24_unique_aia_objects_with_local_paths.csv.gz')
torch.set_num_threads(CONFIG['cpu_threads'])
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
assert torch.cuda.is_available(), 'The shared L4 is required for this execution'
display(pd.DataFrame([{'GPU': torch.cuda.get_device_name(0), 'Torch': torch.__version__,
                       'Batch size': CONFIG['batch_size'], 'Seeds': str(CONFIG['seeds']),
                       'Maximum epochs per seed': CONFIG['max_epochs']}]))''')
    md('''## Data
### 2. Reconcile exact cases and existing cached images
This checks the fixed input hashes, role counts, URI mappings, generation and file sizes before fitting. Each image's actual bytes and identity are checked when read. There is no new image acquisition.''')
    code('''frame, records, preparation = bind_inputs(INPUTS, INVENTORY)
write_json(OUTPUT / 'bound_sources.json', records)
display(frame.groupby('role', sort=False).agg(Cases=('forecast_case_id', 'size'), Positive_windows=('label', 'sum')))
print(f'{len(records):,} existing image files; 0 image downloads')''')
    md('''### 3. Fit preprocessing using training images only
Sample at most 1,000 unique training images with seed 17 and 1,024 pixels per image. Save the sample IDs and hashes alongside the training case IDs. The held-out 2014 data does not fit this transform.''')
    code('''normalization = prepare_normalization(frame, records, OUTPUT)
display(pd.DataFrame({'Channel': normalization['channel_order'],
                      'Scale': normalization['channel_scale'], 'Mean': normalization['channel_mean'],
                      'Std': normalization['channel_std']}))
contract = {'configuration': CONFIG, 'configuration_sha256': file_sha256(ROOT / 'configs/aia72_temporal_v1.json'),
            'input_preparation_sha256': file_sha256(INPUTS / 'preparation.json'),
            'bound_sources_sha256': file_sha256(OUTPUT / 'bound_sources.json'),
            'normalization_sha256': file_sha256(OUTPUT / 'normalization.json'),
            'source_sha256': {p.name: file_sha256(p) for p in sorted((ROOT / 'scripts').glob('*.py'))},
            'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
                        'python': sys.version, 'device': torch.cuda.get_device_name(0)},
            'training_cases': int(frame.role.eq('train').sum()), 'selection_cases': int(frame.role.eq('model_validation').sum()),
            'historical_availability': 'unverified_retrospective', 'source_images_downloaded': 0}
contract_path = OUTPUT / 'run_contract.json'
if contract_path.exists():
    assert json.loads(contract_path.read_text()) == contract, 'Changed contract; do not resume existing weights'
else:
    write_json(contract_path, contract)
CONTRACT_SHA256 = file_sha256(contract_path)
print('Frozen run contract:', CONTRACT_SHA256)''')
    md('''### 4. Model architecture
Fresh weights for every seed. The temporary weights from Notebook 08 and the following GPU benchmark are discarded.''')
    code((root / 'scripts/aia72_model.py').read_text())
    md('''### 5. Resumable training loop
The executable function below is the same source used by the tested runner. It preserves the epoch permutation, next-case cursor, stochastic state and optimizer. Only the earlier model-validation block selects checkpoints.''')
    code(inspect.getsource(train_seed) + "\n# Bind this visible function to the experiment driver.\nimport scripts.train_aia72_gpu as training_module\ntraining_module.train_seed = train_seed\ntraining_module.TemporalAIACNNGRU = TemporalAIACNNGRU")
    md('''## Results
### 6. Benchmark, then fit the three seeds
Two warm-up and four timed batches provide an initial speed estimate, not a completion guarantee. Actual per-epoch logs are the better estimate. A 60-second checkpoint interval preserves progress. If the time limit is reached, rerun the same unchanged bundle after coordinating GPU access; it resumes the saved cursor.

No evaluation-period score or fusion result is produced here.''')
    code('''result = run_training(frame, records, normalization, CONFIG, OUTPUT, CONTRACT_SHA256)
display(pd.DataFrame([json.loads((OUTPUT / 'benchmark.json').read_text())]).T)
display(pd.DataFrame(result.get('seeds', [])))
print('Invocation status:', result['status'])''')
    md('''## Takeaways
### 7. Preserve the actual progress
Read the status literally: a bounded stop is not a completed model; all-seed fitting still needs independent checkpoint replay before downstream use. The runner saves an executed notebook and logs even when a cell fails. No fitted scientific results should be inferred from unexecuted cells.''')
    code('''history = []
for path in sorted(OUTPUT.glob('seed_*_history.csv')):
    history.append(pd.read_csv(path))
if history:
    history_frame = pd.concat(history, ignore_index=True)
    display(history_frame.groupby('seed', sort=False).tail(3))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True, layout='constrained')
    for ax, seed in zip(axes, CONFIG['seeds']):
        rows = history_frame[history_frame.seed.eq(seed)]
        if len(rows):
            ax.plot(rows.epoch, rows.training_log_loss, label='Training 2010–2013', color='#245A81')
            ax.plot(rows.epoch, rows.selection_log_loss, label='Selection Jan–Jun 2014', color='#B55D23', linestyle='--')
        else:
            ax.text(.5, .5, 'No completed epoch', transform=ax.transAxes, ha='center')
        ax.set(title=f'Seed {seed}', xlabel='Completed epoch')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Mean binary log loss'); axes[0].legend(fontsize=8)
    fig.suptitle('Earlier-block fitting progress · no later-period evaluation')
    fig.savefig(OUTPUT / 'learning_curves.png', dpi=160)
    plt.show()
else:
    print('No complete epoch yet; consult progress.json for the saved batch cursor.')
display(Markdown('**Status: ' + result['status'] + '.** Later-period inference, calibration and fusion have not been performed.'))
write_json(OUTPUT / 'notebook_status.json', {'utc': datetime.now(timezone.utc).isoformat(),
    'status': result['status'], 'run_contract_sha256': CONTRACT_SHA256,
    'fit_complete': result['full_training_complete'], 'independent_checkpoint_replay': 'pending'})''')
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    })
    nbf.validate(nb); path = root / 'notebooks/09_AIA_72h_Shared_GPU_Training.ipynb'
    nbf.write(nb, path); print(path)


if __name__ == '__main__':
    build()
