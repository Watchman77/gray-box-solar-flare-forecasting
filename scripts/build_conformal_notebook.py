"""Build notebook 03 with executable conformal methods embedded for portable review."""
import ast
import json
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    source = (ROOT / 'scripts/conformal_sharp.py').read_text()
    functions = {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    config = json.loads((ROOT / 'configs/sharp72_conformal_v1.json').read_text())
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text))
    md('''# 72-hour SHARP conformal uncertainty evaluation
**Notebook 03 · Bamidele Akinwumi · candidate-label research experiment**

## tl;dr
The saved execution supplies empirical coverage, class-specific coverage, prediction-set sizes and uncertainty estimates below. This notebook evaluates frozen predictions; it does not retrain the backbones or choose a policy from later outcomes.''')
    md(r'''## Context & Methods

**Question:** do uncertainty sets retain coverage for rare M/X flare outcomes when transferred from earlier data into Cycle 25, and how much ambiguity does that coverage require?

The target is a science-class M/X flare **starting within the next 72 hours for the primary NOAA region**. Class 0 means no qualifying M/X start in this window, not an absence of all solar activity. Both the GRU ensemble and logistic reference use the earlier-selected raw probabilities from Notebook 02.

Compare **pooled split conformal** with **class-conditional (Mondrian) split conformal**, using the same score $s(x,y)=1-\hat p_y(x)$. At each declared miscoverage level $\alpha$, take the $\lceil(n+1)(1-\alpha)\rceil$-th smallest calibration score. If that rank exceeds the available count, use infinity. Include a candidate class when its score is less than or equal to its threshold; retain ties and empty sets. Class-conditional fitting uses a separate score sample for each true class. No label is needed when issuing a set.

**Fixed comparisons:** 90% target coverage is primary; 95% is a sensitivity. All methods and levels are retained. No method, level or threshold is chosen from evaluation performance.

### Key Assumptions

- Fit only the reserved January–June 2015 conformal block. The frozen split purges cases whose 72-hour outcome plus assumed 24-hour reporting delay crosses 1 July 2015.
- Evaluate the existing 2021–2025 retrospective block and partial 2026 supplementary block. Counts are overlapping forecast windows, not independent flare events.
- Standard coverage theory needs exchangeability; cross-cycle drift and dependence make that assumption questionable here. Nominal 90%/95% levels are targets to check, not operational guarantees.
- Coverage is the fraction of known outcomes included in their issued sets. Report it overall and separately by class. Set size is 0, 1 or 2; a singleton is not proof that a forecast is safe.
- Intervals use 1,000 region-component bootstrap resamples, with seven-day UTC block sensitivity for aggregate periods. They condition on fixed thresholds/models/labels, omit their estimation uncertainty, and are per-comparison rather than simultaneous intervals. Annual plots use region resampling only.
- Unknown outcomes never become negatives. Unsupported SHARP input rows remain unscored; rows before the conformal fitting cutoff receive no retrospectively backdated sets.
- This is binary classification. Conformalized quantile regression for a continuous target is not implemented, and these sets are not intervals for a latent flare probability.
- Policy-development labels remain unused for fitting or evaluation. No normal/degraded/abstention safety policy is validated here.

Method references: [Angelopoulos & Bates, §§1 and 4.2](https://arxiv.org/html/2107.07511v6) and [Barber et al., Conformal prediction beyond exchangeability](https://arxiv.org/abs/2202.13415). The date splits, alpha levels and resampling design are experimental choices.''')
    md('## Data\n### 1. Environment\nUse the project training kernel and `requirements-notebook.txt`. For a fresh environment, optionally install the packages below. Set the paths to your local frozen artifacts. No cloud execution or remote data download is required.')
    code('''INSTALL_DEPENDENCIES = False
if INSTALL_DEPENDENCIES:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy==2.3.5', 'pandas==2.2.3', 'matplotlib==3.10.7'])

from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from IPython.display import display, Markdown

WORKSPACE = Path.cwd()
if WORKSPACE.name == 'notebooks':
    WORKSPACE = WORKSPACE.parent
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 115, 'axes.titleweight': 'bold'})
METHOD_COLORS = {'pooled': '#245A81', 'class_conditional': '#B36B19'}
METHOD_NAMES = {'pooled': 'Pooled', 'class_conditional': 'Class-conditional'}
PERIOD_NAMES = {'retrospective_cycle25': '2021–2025', 'supplementary_2026': '2026 (partial)'}
print({'numpy': np.__version__, 'pandas': pd.__version__})''')
    md('### 2. Pinned inputs and declared protocol')
    code("CONFIG = json.loads(r'''" + json.dumps(config, indent=2) + "''')\n" + '''PARENT_DIR = WORKSPACE / CONFIG['parent_dir']
DATASET_DIR = WORKSPACE / 'data/processed/training_snapshot_20261002/gray_box_aligned_v1'
OUTPUT_DIR = WORKSPACE / 'outputs' / ('sharp72_conformal_v1_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
print('Frozen probabilities:', PARENT_DIR)
print('New run:', OUTPUT_DIR)
display(pd.DataFrame([
    {'Use': 'Conformal fit', 'Start': '2015-01-01', 'End': '2015-07-01 exclusive'},
    {'Use': 'Policy development — reserved', 'Start': '2015-07-01', 'End': '2020-01-01 exclusive'},
    {'Use': 'Retrospective evaluation', 'Start': '2021-01-01', 'End': '2026-01-01 exclusive'},
    {'Use': 'Supplementary evaluation', 'Start': '2026-01-01', 'End': 'latest eligible case in frozen source'}]))''')
    md('### 3. Source integrity and identity checks\nAll parent artifacts are verified against pinned hashes before any fitting. The master table supplies outcome boundaries and confirms the target labels.')
    code('\n\n'.join(functions[n] for n in ['sha256', 'save_json', 'validate_probabilities', 'load_sources']))
    md('### 4. Fit exact order statistics and produce sets\nEach method reads only the conformal-calibration role. JSON null plus `include_all=true` represents an infinite threshold when support is insufficient. Ties are included without randomization.')
    code('\n\n'.join(functions[n] for n in ['finite_sample_threshold', 'fit_thresholds', 'predict_sets', 'conformal_mask', 'freeze_thresholds']))
    md('### 5. Define coverage, ambiguity and singleton errors\nThe singleton error is conditional on the subset receiving exactly one label. It is a descriptive diagnostic, not a controlled selective-risk guarantee. Flare cases assigned only class 0 are also reported explicitly.')
    code(functions['set_metrics'])
    md('### 6. Resample groups, preserving all windows within each group\nIntervals are suppressed if a class has fewer than two supported groups or over 5% of resamples have no such class. They do not account for shared events across different groups or establish independence.')
    code(functions['coverage_bootstrap'])
    md('### 7. Execute the frozen protocol\nSave thresholds before evaluating later outcomes. Preserve every candidate case in the output. Set codes are 0=empty, 1={0}, 2={1}, 3={0,1}; a blank code means no set issued. Sets after the cutoff can be issued without knowing the outcome.')
    code(functions['run_conformal'])
    # Capture the actual executed definitions, including edits made in the notebook.
    imports = source[:source.index('\ndef sha256')]
    code("SOURCE_IMPORTS = " + repr(imports) + "\n" + '''history = get_ipython().history_manager.input_hist_raw
definition_starts = ['def sha256', 'def finite_sample_threshold', 'def set_metrics', 'def coverage_bootstrap', 'def run_conformal']
executed_definitions = [next(cell for cell in reversed(history) if cell.startswith(start)) for start in definition_starts]
EXECUTED_SOURCE = SOURCE_IMPORTS + '\\n\\n'.join(executed_definitions)
summary = run_conformal(PARENT_DIR, DATASET_DIR, CONFIG, OUTPUT_DIR, EXECUTED_SOURCE)
metrics = pd.read_csv(OUTPUT_DIR / 'metrics.csv')
yearly = pd.read_csv(OUTPUT_DIR / 'yearly_metrics.csv')
intervals = pd.read_csv(OUTPUT_DIR / 'coverage_intervals.csv', dtype={'year': str})
thresholds = json.loads((OUTPUT_DIR / 'thresholds.json').read_text())
print(summary['status'])
display(pd.DataFrame([summary['support']]))
threshold_rows = []
for model, methods in thresholds['models'].items():
    for method, levels in methods.items():
        for alpha, parameters in levels.items():
            for label, q in parameters['thresholds'].items():
                threshold_rows.append({'Model': model, 'Method': METHOD_NAMES[method], 'Target coverage': 1-float(alpha),
                                      'Class': label, 'Calibration scores': q['n'], 'Rank': q['rank'], 'Threshold': q['threshold']})
display(pd.DataFrame(threshold_rows).round(5))''')
    md('## Results\n### 8. Coverage and efficiency on matched known-outcome cases\nA larger prediction set can improve coverage by admitting both classes, at the cost of an ambiguous forecast. Review both columns together. The 95% level remains a sensitivity rather than a post-hoc choice.')
    code('''for alpha in CONFIG['alphas']:
    display(Markdown(f"**Target coverage: {1-alpha:.0%}**"))
    table = metrics[metrics.alpha.eq(alpha)].copy()
    table['Period'] = table.role.map(PERIOD_NAMES)
    table['Method'] = table.method.map(METHOD_NAMES)
    display(table[['Period', 'model', 'Method', 'cases', 'flare_cases', 'coverage', 'nonflare_coverage',
                   'flare_coverage', 'mean_set_size', 'both_fraction', 'empty_fraction', 'singleton_error']].round(4))''')
    md('### 9. Overall coverage can conceal rare-class failure\nDots show empirical coverage; bars show conditional 95% region-component bootstrap intervals. The dashed line is the 90% target. The interval is uncertainty about the observed coverage estimate, distinct from the target coverage of the prediction set.')
    code('''fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharey=True, layout='constrained')
for row, model in enumerate(CONFIG['models']):
    for col, role in enumerate(CONFIG['evaluation_roles']):
        ax = axes[row, col]
        data = intervals[(intervals.model == model) & (intervals.role == role) & intervals.alpha.eq(.1)
                         & intervals.year.eq('all') & intervals.grouping.eq('region_component')]
        for method, offset, marker in [('pooled', -.10, 'o'), ('class_conditional', .10, 's')]:
            values = data[data.method.eq(method)].set_index('population').loc[['all', 'nonflare', 'flare']]
            x = np.arange(3) + offset
            ax.vlines(x, values.low, values.high, color=METHOD_COLORS[method], lw=2)
            ax.plot(x, values.coverage, marker, color=METHOD_COLORS[method], label=METHOD_NAMES[method])
        ax.axhline(.9, color='#444444', ls='--', lw=1.2)
        ax.set(xticks=range(3), xticklabels=['All outcomes', 'No M/X', 'M/X flare'], ylim=(0, 1.04),
               title=f'{model.upper()} · {PERIOD_NAMES[role]}')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', alpha=.18)
        if col == 0: ax.set_ylabel('Empirical set coverage')
axes[0, 0].legend(loc='lower left', frameon=False)
fig.suptitle('72-hour prediction-set coverage · 90% target', fontsize=15)
fig.savefig(OUTPUT_DIR / 'coverage_by_class.png', bbox_inches='tight')
plt.show()''')
    md('### 10. What the prediction sets contain\nEach bar covers the same known-outcome evaluation population used above. Missing-input rows are reported separately below, not absorbed into the empty-set category. These four mathematical set types are not operational safety states.')
    code('''fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, layout='constrained')
state_style = [('nonflare_only', 'Only no M/X', '#245A81', ''), ('flare_only', 'Only M/X', '#B36B19', '//'),
               ('both', 'Both labels', '#D8B95B', '..'), ('empty', 'Empty', '#92979C', 'xx')]
for ax, model in zip(axes, CONFIG['models']):
    table = metrics[(metrics.model == model) & metrics.alpha.eq(.1)].copy()
    table['order'] = table.role.map({'retrospective_cycle25': 0, 'supplementary_2026': 2}) + table.method.map({'pooled': 0, 'class_conditional': 1})
    table = table.sort_values('order')
    left = np.zeros(len(table))
    for state, label, color, hatch in state_style:
        values = table[state + '_fraction'].to_numpy()
        ax.barh(np.arange(len(table)), values, left=left, color=color, hatch=hatch, edgecolor='white', label=label)
        for i, width in enumerate(values):
            if width >= .07:
                ax.text(left[i]+width/2, i, f'{width:.0%}', ha='center', va='center', color='white' if state == 'nonflare_only' else '#202020')
        left += values
    ax.set(yticks=np.arange(len(table)), yticklabels=[f'{PERIOD_NAMES[r.role]}\\n{METHOD_NAMES[r.method]}' for r in table.itertuples()],
           xlim=(0, 1), title=model.upper(), xlabel='Share of known-outcome evaluation windows')
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=4, frameon=False)
fig.suptitle('Prediction-set composition · 90% target', fontsize=15)
fig.savefig(OUTPUT_DIR / 'set_composition.png', bbox_inches='tight')
plt.show()''')
    md('### 11. Transfer across years\nAnnual estimates retain the fixed 2015 thresholds. Bands show conditional region-component bootstrap intervals at the primary 90% target. The final year is partial through the available August 2026 cases; calendar periods differ in duration and event counts.')
    code('''fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharex=True, sharey=True, layout='constrained')
for row, model in enumerate(CONFIG['models']):
    for col, population_name in enumerate(['nonflare', 'flare']):
        ax = axes[row, col]
        for method, marker, linestyle in [('pooled', 'o', '-'), ('class_conditional', 's', '--')]:
            values = intervals[(intervals.model == model) & intervals.method.eq(method) & intervals.alpha.eq(.1)
                               & intervals.population.eq(population_name) & intervals.year.ne('all')
                               & intervals.grouping.eq('region_component')].copy()
            values['year'] = values.year.astype(int)
            values = values.sort_values('year')
            ax.plot(values.year, values.coverage, marker=marker, ls=linestyle, color=METHOD_COLORS[method], label=METHOD_NAMES[method])
            ax.fill_between(values.year, values.low, values.high, color=METHOD_COLORS[method], alpha=.12)
        ax.axhline(.9, color='#444444', ls=':', lw=1.2)
        ax.set(title=f'{model.upper()} · {"No M/X" if population_name == "nonflare" else "M/X flare"}', ylim=(0, 1.04),
               xticks=range(2021, 2027), xticklabels=['2021', '2022', '2023', '2024', '2025', '2026*'])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', alpha=.18)
        if col == 0: ax.set_ylabel('Empirical set coverage')
axes[0, 0].legend(loc='lower left', frameon=False)
fig.suptitle('Yearly coverage · 90% target · *2026 partial', fontsize=15)
fig.savefig(OUTPUT_DIR / 'yearly_coverage.png', bbox_inches='tight')
plt.show()
support = yearly[(yearly.model == 'gru') & yearly.method.eq('pooled') & yearly.alpha.eq(.1)]
display(support[['year', 'cases', 'nonflare_cases', 'flare_cases']].reset_index(drop=True))''')
    md('### 12. Dependence sensitivity and population availability\nThe seven-day resampling provides a sensitivity to the grouping choice; neither scheme establishes independent events. Complete population outputs retain missing inputs and unknown outcomes. The two backbones use matched input cases.')
    code('''table = intervals[intervals.alpha.eq(.1) & intervals.year.eq('all') & intervals.population.eq('flare')]
display(table[['role', 'model', 'method', 'grouping', 'cases', 'groups_with_class', 'coverage', 'low', 'high']].round(4))
availability = pd.read_csv(OUTPUT_DIR / 'availability.csv')
display(availability[(availability.model == 'gru') & (availability.year >= 2021)].drop(columns='model').reset_index(drop=True))
print('All candidate cases:', summary['population_rows'])
print('All-year missing-input cases:', summary['missing_input_rows'])
print('Mapped region overlap with conformal fit:', summary['calibration_evaluation_region_overlap_counts'])
for name, expected in summary['output_sha256'].items():
    assert sha256(OUTPUT_DIR / name) == expected, name
parent = json.loads((PARENT_DIR / 'summary.json').read_text())
for name, expected in parent['output_sha256'].items():
    assert sha256(PARENT_DIR / name) == expected, name
print('Frozen inputs and saved outputs verified unchanged.')''')
    md('## Takeaways')
    code('''messages = []
for role in CONFIG['evaluation_roles']:
    for model in CONFIG['models']:
        rows = metrics[(metrics.role == role) & metrics.model.eq(model) & metrics.alpha.eq(.1)].set_index('method')
        pooled, conditional = rows.loc['pooled'], rows.loc['class_conditional']
        messages.append(f"**{model.upper()}, {PERIOD_NAMES[role]}:** pooled overall coverage {pooled.coverage:.1%}, "
                        f"flare coverage {pooled.flare_coverage:.1%}. Class-conditional flare coverage {conditional.flare_coverage:.1%}; "
                        f"both-label sets {conditional.both_fraction:.1%}. Target: 90%.")
display(Markdown('\\n\\n'.join(messages)))
display(Markdown('### Interpretation limits\\n' + '\\n'.join('- ' + item for item in summary['limitations'])))''')
    md('''**Next research step:** use the untouched policy-development outcomes to develop a prespecified decision policy, with explicit coverage and error targets, while retaining all failures observed here. Any rolling/adaptive conformal extension must use only outcomes whose forecast windows and declared reporting delay have elapsed. Neither a favorable aggregate coverage number nor this completed notebook establishes operational readiness.

**Reusable outputs:** frozen thresholds, full-population sets, exact numerators/denominators, annual metrics, coverage intervals, availability counts, executable code, and a configuration/runtime/hash receipt in the timestamped output directory. Large per-case files stay local; compact receipts and this executed notebook are versioned in the repository.''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3 (SHARP conformal)', 'name': 'python3', 'language': 'python'},
        'language_info': {'name': 'python'}, 'colab': {'name': '03_SHARP_72h_Conformal_Uncertainty.ipynb', 'provenance': []}})
    nbf.validate(notebook)
    path = ROOT / 'notebooks/03_SHARP_72h_Conformal_Uncertainty.ipynb'
    nbf.write(notebook, path)
    print(path)


if __name__ == '__main__':
    build()
