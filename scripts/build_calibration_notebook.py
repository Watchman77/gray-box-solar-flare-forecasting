"""Create notebook 02 with the tested calibration functions embedded."""

import ast
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    source = (ROOT / "scripts/calibrate_sharp.py").read_text()
    functions = {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    config = json.loads((ROOT / "configs/sharp72_calibration_v1.json").read_text())
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text))
    md("""# 72-hour SHARP probability calibration

**Notebook 02 · Bamidele Akinwumi · exploratory candidate-label experiment**

Keep the trained GRU ensemble and logistic reference frozen. Fit and compare raw, logistic/Platt and isotonic probabilities using earlier data; then evaluate the locked methods on Cycle 25 and supplementary 2026. Actual results and reliability figures are saved below.

## Context and methods

1. Fit candidate calibrators on eligible July–November 2014 cases.
2. Select separately for each backbone on December 2014, using Brier score, then log loss, then the fixed method order to break ties. The raw identity map is a candidate.
3. Refit the prespecified methods on the full eligible July–December calibration block. Save the selected method before evaluation.
4. Evaluate all locked candidates on the existing retrospective/supplementary blocks. Never change the choice using these results.

### Key assumptions

- Target: the same science-class M/X start within 72 hours for the primary NOAA region. Labels remain provisional, and counts below are forecast windows, not independent flare events.
- Purge outcome windows plus an assumed 24-hour reporting delay at the internal selection boundary.
- Platt-style mapping: unpenalized logistic regression of outcome on the raw probability's logit, clipped at `1e-6`. Isotonic uses increasing regression and endpoint clipping outside fitted score support.
- Reliability uses ten fixed equal-width bins; ECE depends on these bins. Empty bins remain undefined.
- Calibration intercept/slope are jointly estimated descriptive diagnostics on each evaluation block. They never alter the issued probabilities.
- Paired Brier intervals use 1,000 resamples of mapped region components, plus an epoch-anchored seven-day UTC block sensitivity. These are conditional, per-comparison percentile intervals, not simultaneous guarantees or independent validation.
- Conformal-calibration and policy-development blocks remain reserved.

**Required input:** the frozen result directory from notebook 01, and the matching dataset package. This notebook does not retrain the backbones or download AIA. A different training result requires its own pinned configuration, rather than bypassing the integrity checks.""")
    md("## 1. Environment and paths\n\nUse the project's training kernel with `requirements-notebook.txt`. For another environment, install the listed packages and set the source/data paths below. Execution was verified locally; cloud execution is not claimed.")
    code("""# Optional setup for a fresh notebook environment. Restart afterwards if packages were already loaded.
INSTALL_DEPENDENCIES = False
if INSTALL_DEPENDENCIES:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy==2.3.5', 'pandas==2.2.3',
                           'scikit-learn==1.7.2', 'scipy==1.16.3', 'matplotlib==3.10.7'])

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from IPython.display import display, Markdown

WORKSPACE = Path.cwd()
if WORKSPACE.name == 'notebooks':
    WORKSPACE = WORKSPACE.parent
print({'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__})""")
    code("CONFIG = json.loads(r'''" + json.dumps(config, indent=2) + "''')\n" + """TRAINING_DIR = WORKSPACE / CONFIG['training_dir']
DATASET_DIR = WORKSPACE / 'data/processed/training_snapshot_20261002/gray_box_aligned_v1'
OUTPUT_DIR = WORKSPACE / 'outputs' / ('sharp72_calibration_v1_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
print('Frozen training:', TRAINING_DIR)
print('New calibration run:', OUTPUT_DIR)
display(pd.DataFrame([
    {'Stage': 'Inner fit', 'Start': '2014-07-01', 'End (exclusive)': '2014-12-01'},
    {'Stage': 'Method selection', 'Start': '2014-12-01', 'End (exclusive)': '2015-01-01'},
    {'Stage': 'Final calibration refit', 'Start': '2014-07-01', 'End (exclusive)': '2015-01-01'}]))""")
    md("## 2. Verify the frozen model outputs and target data")
    code("\n\n".join(functions[name] for name in ("sha256", "save_json", "check_probabilities", "load_sources")))
    md("## 3. Fit and serialize calibration maps\n\nCalibrators are saved as plain JSON parameters. Their restored predictions are checked against the fitted scikit-learn estimator.")
    code("\n\n".join(functions[name] for name in ("apply_calibrator", "fit_calibrator")))
    md("## 4. Reliability, probability scores and earlier-data selection")
    code("\n\n".join(functions[name] for name in ("reliability_table", "score_probabilities", "calibration_masks", "select_and_refit")))
    md("## 5. Paired uncertainty and execution\n\nA negative Brier difference favours the calibrated candidate over its own raw backbone. Components/blocks are resampled with all their included windows; methods stay paired on identical cases.")
    code(functions["paired_brier_bootstrap"] + "\n\n" + functions["run_calibration"])
    code("""executed_code = '\\n\\n'.join(get_ipython().history_manager.input_hist_raw)
summary = run_calibration(TRAINING_DIR, DATASET_DIR, CONFIG, OUTPUT_DIR, executed_code)
selection = json.loads((OUTPUT_DIR / 'selection.json').read_text())
print(summary['status'])
print('Backbone refitted:', summary['backbone_refitted'])
display(pd.DataFrame(selection['support']).T.rename(columns={'cases': 'Windows', 'positive': 'Positive windows'}))
display(Markdown('**Frozen choices:** ' + '; '.join(f"{model}: **{method}**" for model, method in selection['selected_methods'].items())))""")
    md("## 6. Selection evidence from December 2014\n\nThis table controls the chosen method. Later evaluation tables are descriptive and do not change the selection.")
    code("""inner = pd.DataFrame(selection['inner_scores'])
inner['Selected'] = [method == selection['selected_methods'][model] for model, method in zip(inner.model, inner.method)]
display(inner[['model', 'method', 'cases', 'positive', 'brier', 'log_loss', 'Selected']]
        .rename(columns={'model': 'Model', 'method': 'Method', 'cases': 'Windows', 'positive': 'Positive windows',
                         'brier': 'Brier', 'log_loss': 'Log loss'}).round(6))""")
    md("## 7. Frozen evaluation results\n\nBrier skill compares against the training-fitted climatology. A lower Brier score alone does not establish improved calibration; inspect reliability and support alongside it.")
    code("""metrics = pd.read_csv(OUTPUT_DIR / 'metrics.csv')
period_names = {'retrospective_cycle25': 'Cycle 25: 2021–25', 'supplementary_2026': '2026: supplementary'}
readable = metrics.copy()
readable['role'] = readable.role.replace(period_names)
readable = readable.rename(columns={'role': 'Period', 'model': 'Model', 'method': 'Method',
    'selected_on_earlier_data': 'Selected', 'brier': 'Brier', 'brier_skill_train_climatology': 'Brier skill',
    'log_loss': 'Log loss', 'ece_equal_width_10': 'ECE', 'calibration_intercept': 'Intercept', 'calibration_slope': 'Slope'})
display(readable[['Period', 'Model', 'Method', 'Selected', 'Brier', 'Brier skill', 'Log loss']].round(6))
display(readable[['Period', 'Model', 'Method', 'ECE', 'Intercept', 'Slope']].round(4))""")
    md("## 8. Reliability and bin support\n\nThe diagonal is ideal reliability. Curves show mean forecast probability versus observed candidate-event fraction within each bin. The lower panels show the number of windows behind those points; sparse bins are not precise estimates. No independent-bin confidence intervals are asserted.")
    code("""os.environ.setdefault('MPLCONFIGDIR', str(WORKSPACE / 'outputs' / '.matplotlib'))
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white'})
reliability = pd.read_csv(OUTPUT_DIR / 'reliability.csv')
styles = {'raw': ('#245B84', 'o', '-'), 'platt': ('#B96A24', 's', '--'), 'isotonic': ('#6D5B91', '^', ':')}
for model in ['gru', 'logistic']:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex='col',
                             gridspec_kw={'height_ratios': [2, 1]}, constrained_layout=True)
    for column, role in enumerate(CONFIG['evaluation_roles']):
        evidence = metrics[(metrics.model == model) & (metrics.role == role)].iloc[0]
        axes[0, column].plot([0, 1], [0, 1], color='#666666', linewidth=1, label='Ideal')
        for method, (color, marker, line) in styles.items():
            subset = reliability[(reliability.model == model) & (reliability.role == role) & (reliability.method == method)]
            axes[0, column].plot(subset.mean_probability, subset.observed_fraction, color=color, marker=marker,
                                 linestyle=line, linewidth=1.6, markersize=4, label=method.capitalize())
            axes[1, column].step((subset.left + subset.right)/2, subset['count'], where='mid',
                                 color=color, linestyle=line, linewidth=1.6)
        axes[0, column].set(title=f"{period_names[role]}\\n{int(evidence.cases):,} windows; {int(evidence.positive):,} positive",
                            xlim=(0, 1), ylim=(0, 1), ylabel='Observed event fraction')
        axes[0, column].legend(frameon=False, loc='upper left')
        axes[1, column].set(yscale='symlog', xlabel='Forecast probability', ylabel='Windows (symlog)')
        axes[1, column].set_ylim(bottom=0)
        axes[0, column].grid(color='#E6E8EA', linewidth=0.6)
        axes[1, column].grid(axis='y', color='#E6E8EA', linewidth=0.6)
    fig.suptitle(f"72-hour {model.upper()} · reliability and evidence per bin", fontsize=13)
    fig.savefig(OUTPUT_DIR / f'reliability_{model}.png', dpi=150, bbox_inches='tight')
    plt.show()""")
    md("## 9. Paired Brier differences and checks\n\nIntervals compare each prespecified candidate with its own raw probabilities. They are 95% percentile intervals for each contrast, with no correction for multiple comparisons. They do not include uncertainty in the labels, calibration-method selection or training process.")
    code("""intervals = pd.read_csv(OUTPUT_DIR / 'paired_brier_intervals.csv')
for scheme in CONFIG['bootstrap_groups']:
    display(Markdown(f'**Resampling unit: {scheme}**'))
    table = intervals[intervals.grouping == scheme].copy()
    table['role'] = table.role.replace(period_names)
    display(table[['role', 'model', 'method', 'difference', 'low', 'high', 'groups']]
            .rename(columns={'role': 'Period', 'model': 'Model', 'method': 'Method', 'difference': 'Brier difference',
                             'low': '95% lower', 'high': '95% upper', 'groups': 'Groups'}).round(6))
for name, expected in summary['output_sha256'].items():
    assert sha256(OUTPUT_DIR / name) == expected, f'Changed calibration output: {name}'
parent = json.loads((TRAINING_DIR / 'summary.json').read_text())
for name, expected in parent['output_sha256'].items():
    assert sha256(TRAINING_DIR / name) == expected, f'Backbone artifact changed: {name}'
print('Calibration outputs verified; all frozen backbone artifacts remain unchanged.')
print('Full population:', summary['population_rows'], '| Missing-input cases without scores:', summary['unscored_rows'])""")
    md("## Takeaways and next step")
    code("""messages = []
for model in ['gru', 'logistic']:
    method = summary['selected_methods'][model]
    messages.append(f'**{model.upper()}** retained **{method}** using the earlier selection block.')
    for role in CONFIG['evaluation_roles']:
        selected_score = metrics[(metrics.model == model) & (metrics.method == method) & (metrics.role == role)].iloc[0]
        raw_score = metrics[(metrics.model == model) & (metrics.method == 'raw') & (metrics.role == role)].iloc[0]
        messages.append(f"{period_names[role]}: selected Brier {selected_score.brier:.6f}, raw {raw_score.brier:.6f}; difference {selected_score.brier-raw_score.brier:+.6f}.")
display(Markdown('\\n\\n'.join(messages)))
display(Markdown('### Interpretation limits\\n' + '\\n'.join('- ' + item for item in summary['limitations'])))""")
    md("""Next, use the frozen chosen probabilities for the separate conformal-calibration stage and assess class-specific coverage before developing the operational policy. Retain the raw and alternative methods for the prespecified comparisons; do not replace the chosen method based on these evaluation results.

**Sources and implementation:** frozen prediction/summary SHA-256 values appear in `CONFIG`; the run contract records runtime versions and executed code. Method definitions follow the [scikit-learn probability-calibration guide](https://scikit-learn.org/1.7/modules/calibration.html) and [IsotonicRegression reference](https://scikit-learn.org/1.7/modules/generated/sklearn.isotonic.IsotonicRegression.html). The chronological selection, reporting-delay assumption and group-resampling scheme are choices of this experiment, not guarantees from those references.""")
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3 (SHARP calibration)", "name": "python3", "language": "python"},
        "language_info": {"name": "python"}, "colab": {"name": "02_SHARP_72h_Calibration.ipynb", "provenance": []}})
    nbf.validate(notebook)
    destination = ROOT / "notebooks/02_SHARP_72h_Calibration.ipynb"
    nbf.write(notebook, destination)
    print(destination)


if __name__ == "__main__":
    build()
