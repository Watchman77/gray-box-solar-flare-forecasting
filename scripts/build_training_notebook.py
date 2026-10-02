"""Build the self-contained notebook from the tested training functions."""

import ast
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def definitions(path):
    source = path.read_text()
    return {node.name: ast.get_source_segment(source, node)
            for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def build():
    training = definitions(ROOT / "scripts/train_sharp_temporal.py")
    loader = definitions(ROOT / "scripts/dataset_io.py")
    reference = definitions(ROOT / "scripts/run_exploratory_sharp.py")
    config = json.loads((ROOT / "configs/sharp72_temporal_v1.json").read_text())
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text))

    md("""# 72-hour SHARP training — Gray-Box project

**Bamidele Akinwumi · exploratory experiment · no AIA pixels required**

Train three compact GRU models and a matched logistic reference, preserve their checkpoints, and evaluate frozen raw scores across solar cycles. This notebook includes the executable model and training functions; it does not require the repository's Python modules.

The first local run found similar Brier scores for both models on Cycle 25, with slightly higher average precision for the GRU ensemble. These candidate-label results do not establish a significant improvement. Calibration, conformal sets and an operational decision policy are subsequent stages.

## Context and methods

- Inputs: 16 SHARP parameters at issue minus 288, 192 and 96 native minutes; physical UTC observation times are preserved.
- Outcome: candidate science-class M/X event start in `(issue, issue + 72 h]`, associated with the primary NOAA region. Unknown labels never become negatives.
- Model fitting uses 2010–2013; early stopping uses January–June 2014. Later Cycle-24 blocks are reserved for probability calibration, conformal calibration and policy development.
- Cycle 25 (2021–2025) is retrospective evaluation; 2026 through 17 August is supplementary. Previously inspected years are not untouched prospective confirmation.
- A 24-hour reporting delay is an explicit assumption. Earlier windows crossing each block's boundary are purged.
- Three short histories differ from the published seven-daily-state model. This is a new compact baseline, not its reproduction.

### Run locally or in Colab

Locally, use the `.venv-training` kernel and run all cells. In Colab, upload `gray_box_aligned_v1.zip` to `/content`, set `WORKSPACE = Path('/content')`, and run the optional dependency cell if required. The archive is approximately 46 MB; no AIA image download is needed. You may also point `DATASET_DIR` to an already extracted, verified package.

Every full run writes a new timestamped result directory. Existing results are preserved. The saved notebook includes actual local execution outputs; Colab execution is not claimed here.""")
    md("## 1. Environment")
    code("""# Set True only when the current kernel lacks the required libraries.
INSTALL_DEPENDENCIES = False
if INSTALL_DEPENDENCIES:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
        'numpy==2.3.5', 'pandas==2.2.3', 'torch==2.8.0', 'matplotlib==3.10.7'])
    print('Restart the kernel after installation, set this flag to False, then run all cells.')""")
    code("""import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from torch import nn
from IPython.display import display, Markdown

print({'numpy': np.__version__, 'pandas': pd.__version__, 'torch': torch.__version__})""")
    md("## 2. Paths and frozen experiment configuration\n\nChange paths here. Keep the target and split definitions fixed when comparing models. Updating the configuration creates a different experiment.")
    code("""# In Colab, set WORKSPACE = Path('/content'). Locally the repository is detected.
WORKSPACE = Path.cwd()
if WORKSPACE.name == 'notebooks':
    WORKSPACE = WORKSPACE.parent
DATASET_DIR = WORKSPACE / 'data/processed/training_snapshot_20261002/gray_box_aligned_v1'
DATASET_ARCHIVE = WORKSPACE / 'data/processed/gray_box_aligned_v1.zip'
if not DATASET_ARCHIVE.exists() and (WORKSPACE / 'gray_box_aligned_v1.zip').exists():
    DATASET_ARCHIVE = WORKSPACE / 'gray_box_aligned_v1.zip'
RUN_NAME = 'sharp72_notebook_v1_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT_DIR = WORKSPACE / 'outputs' / RUN_NAME
CONFIG = json.loads(r'''""" + json.dumps(config, indent=2) + """''')
display(pd.DataFrame(CONFIG['blocks']))
print('Dataset:', DATASET_DIR)
print('New run:', OUTPUT_DIR)""")
    md("## 3. Load and verify the data\n\nThe archive and manifest are pinned. A changed file stops execution. Blank histories remain unavailable; unknown outcomes are excluded from training and metric calculations. This cohort was assembled for matched modalities, so removing AIA does not make it a complete SHARP census.")
    code("""if not DATASET_DIR.exists():
    if not DATASET_ARCHIVE.is_file():
        raise FileNotFoundError('Upload gray_box_aligned_v1.zip or set DATASET_DIR to its extracted folder.')
    with DATASET_ARCHIVE.open('rb') as stream:
        archive_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert archive_hash == '3bb0a7fc10d0192ed8bdabb4744079ac8ed7bcc8845d4bd21fefa45068b3df18', 'Archive changed'
    extraction_root = DATASET_DIR.parent
    extraction_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATASET_ARCHIVE) as archive:
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in archive.namelist())
        archive.extractall(extraction_root)
print('Data package located; full checks run before fitting.')""")
    code("\n\n".join(loader[name] for name in ("file_sha256", "load_dataset", "preflight_dataset")))
    md("## 4. Temporal splits and training-only scaling\n\nSigned `log1p` compression is followed by feature-wise scaling estimated from the training block only. No evaluation data are used for preprocessing or early stopping.")
    code("\n\n".join(training[name] for name in ("write_json", "make_roles", "fit_transform", "transform")))
    md("## 5. GRU architecture and inference\n\nOne GRU layer with 32 hidden units, 0.2 head dropout and a linear binary output. All three random seeds are retained; no seed is selected using Cycle-25 results.")
    code("\n\n".join(training[name] for name in ("SharpGRU", "predict")) + "\n\ndisplay(SharpGRU(16, CONFIG['hidden_size'], CONFIG['head_dropout']))")
    md("## 6. Metrics and logistic reference\n\nBrier skill uses the training prevalence as climatology. Average precision handles tied scores; AUC is undefined when only one outcome class is present. The logistic reference uses the same cases and transformed SHARP values.")
    code(training["metrics"] + "\n\n" + "\n\n".join(reference[name] for name in ("sigmoid", "fit_logistic")))
    md("## 7. Train each seed and save its best checkpoint\n\nUse natural class prevalence and unweighted binary cross-entropy. Select the epoch with minimum earlier validation log loss; stop after five epochs without improvement. The saved transform and checkpoints must reproduce every supported prediction.")
    code(training["train_seed"])
    md("## 8. Execute the complete training run\n\nThis cell trains the GRUs and logistic reference, then scores the reserved and evaluation cases. It does not fit calibrators or policies. Detailed epoch logs are written to a file; the notebook displays compact results below.")
    runner = training["run"]
    old = '''"code_sha256": {name: file_sha256(Path(__file__).with_name(name))
                                for name in (Path(__file__).name, "dataset_io.py", "run_exploratory_sharp.py")},'''
    new = '''"executed_cells_sha256": hashlib.sha256("\\n\\n".join(get_ipython().history_manager.input_hist_raw).encode()).hexdigest(),'''
    assert old in runner
    runner = runner.replace(old, new)
    runner = runner.replace('write_json(staging / "run_contract.json", contract)',
        'write_json(staging / "run_contract.json", contract)\n    (staging / "executed_training_cells.py").write_text("\\n\\n".join(get_ipython().history_manager.input_hist_raw))')
    code(runner)
    code("""OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
config_path = OUTPUT_DIR.parent / (RUN_NAME + '_config.json')
write_json(config_path, CONFIG)
log_path = OUTPUT_DIR.parent / (RUN_NAME + '.log')
with log_path.open('w') as log_stream, contextlib.redirect_stdout(log_stream):
    run(DATASET_DIR, config_path, OUTPUT_DIR)
summary = json.loads((OUTPUT_DIR / 'summary.json').read_text())
print(summary['status'])
print('Saved models, transform, split and predictions:', OUTPUT_DIR)
display(pd.DataFrame(summary['seeds']).round(6))""")
    md("## 9. Results and learning curves\n\nThese are uncalibrated, exploratory results. The calibration and policy blocks retain their labels for later stages; their performance is not inspected here.")
    code("""support = pd.DataFrame(summary['roles'])
display(support)
results = pd.DataFrame(summary['metrics'])
evaluation = results[results.role.isin(['retrospective_cycle25', 'supplementary_2026'])].copy()
readable = evaluation[['role', 'model', 'cases', 'positive', 'brier',
                       'average_precision', 'roc_auc', 'brier_skill_train_climatology']].copy()
readable['role'] = readable['role'].replace({'retrospective_cycle25': 'Cycle 25: 2021–25', 'supplementary_2026': '2026: supplementary'})
readable['model'] = readable['model'].replace({'probability_gru_mean': 'GRU ensemble', 'probability_logistic': 'Logistic'})
readable = readable.rename(columns={'role': 'Period', 'model': 'Model', 'cases': 'Cases', 'positive': 'Positive windows',
    'brier': 'Brier', 'average_precision': 'AP', 'roc_auc': 'ROC AUC', 'brier_skill_train_climatology': 'Brier skill'})
display(readable.reset_index(drop=True).round(6))""")
    code("""os.environ.setdefault('MPLCONFIGDIR', str(WORKSPACE / 'outputs' / '.matplotlib'))
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white'})
colors = {17: '#245B84', 29: '#B96A24', 43: '#6D5B91'}
markers = {17: 'o', 29: 's', 43: '^'}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for seed in CONFIG['seeds']:
    history = pd.read_csv(OUTPUT_DIR / f'seed_{seed}_history.csv')
    for axis, column in zip(axes, ['train_loss', 'validation_log_loss']):
        axis.plot(history.epoch, history[column], marker=markers[seed], markersize=4,
                  color=colors[seed], label=f'Seed {seed}', linewidth=1.7)
for axis, title in zip(axes, ['Training loss: 2010–2013', 'Validation log loss: Jan–Jun 2014']):
    axis.set(title=title, xlabel='Epoch', ylabel='Binary log loss')
    axis.grid(axis='y', color='#E3E5E8', linewidth=0.7)
    axis.legend(frameon=False)
fig.suptitle('72-hour SHARP GRU · natural prevalence · early stopping', fontsize=13)
fig.savefig(OUTPUT_DIR / 'learning_curves.png', dpi=160, bbox_inches='tight')
plt.show()""")
    md("## 10. Verify saved outputs and interpret cautiously")
    code("""for name, expected in summary['output_sha256'].items():
    assert file_sha256(OUTPUT_DIR / name) == expected, f'Changed output: {name}'
assert summary['saved_model_replay'] == 'identical_all_supported_cases_each_seed'
cycle25 = evaluation[evaluation.role == 'retrospective_cycle25'].set_index('model')
gru = cycle25.loc['probability_gru_mean']
logistic = cycle25.loc['probability_logistic']
display(Markdown(
    f"**Executed:** three GRU seeds and the matched logistic reference. "
    f"Cycle-25 Brier: GRU **{gru.brier:.6f}**, logistic **{logistic.brier:.6f}**; "
    f"average precision: **{gru.average_precision:.4f}** and **{logistic.average_precision:.4f}**, respectively. "
    "Small descriptive differences are not evidence of statistical significance."
))
print('All saved output hashes verified; checkpoint replay identical.')
print('Full population:', summary['population_rows'], '| Unscored missing-input cases:', summary['unscored_rows'])
display(Markdown('### Remaining limits\\n' + '\\n'.join('- ' + item for item in summary['limitations'])))""")
    md("""## Next steps

Fit and compare probability calibration on the reserved earlier block, then develop conformal classification and the decision policy using their separate blocks. Keep this run frozen. Do not choose methods from the already inspected Cycle-25 scores.

**Data source:** `gray_box_aligned_v1.zip`, archive SHA-256 `3bb0a7fc10d0192ed8bdabb4744079ac8ed7bcc8845d4bd21fefa45068b3df18`; manifest SHA-256 is recorded in `CONFIG`. GOES provides event outcomes here, not continuous XRS predictors. Per-run contracts retain the executed notebook code, configuration, runtime versions and output hashes. Candidate labels and historical availability remain scientifically provisional.""")
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3 (SHARP training)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "01_SHARP_72h_Training.ipynb", "provenance": []}})
    nbf.validate(notebook)
    destination = ROOT / "notebooks/01_SHARP_72h_Training.ipynb"
    destination.parent.mkdir(exist_ok=True)
    nbf.write(notebook, destination)
    print(destination)


if __name__ == "__main__":
    build()
