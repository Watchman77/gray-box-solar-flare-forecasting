"""Build Notebook 07: the executable SHARP/AIA/GOES integration contract."""

from pathlib import Path
import nbformat as nbf


def build():
    nb = nbf.v4.new_notebook()
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text))

    md('''# 07 · SHARP, AIA and GOES: the shared 72-hour experiment

## tl;dr
This notebook exports the exact cases, target and chronological roles needed to compare SHARP-only, AIA-only, GOES-only, SHARP+GOES, SHARP+AIA and SHARP+AIA+GOES. It checks agreement with the executed SHARP experiment, lists reusable image objects and creates GOES case requests.

**This is executed preparation, not AIA/GOES training or a fusion result.** Image references do not prove pixel integrity; a GOES source inventory is not a predictor matrix. The numerical summary below is generated from the pinned data. Full-population records retain missing inputs and unknown labels.''')
    md('''## Context & Methods
Use the original 72-hour primary-region M/X candidate target. Keep all fitting, model selection, probability calibration, uncertainty calibration and policy development in their reserved earlier blocks. Purge outcomes plus the existing assumed 24-hour reporting delay at role boundaries. The same native forecast cases are retained; this is not a new daily forecast schedule.

### Key assumptions
- Labels remain exploratory science-class/start-time candidates. Continuous negative-label coverage and historical delivery are not certified.
- Preserve native TAI issue times and the pinned UTC outcome endpoints: the 72-hour elapsed interval crosses leap seconds correctly, while naive UTC subtraction can differ by one second.
- Reuse raw AIA objects and compatible upstream code. A 48-hour score is not a 72-hour forecast, and a model fitted on later years cannot be used to generate leakage-free earlier calibration predictions.
- GOES future events supply labels; past continuous XRS measurements may supply global-context inputs. Satellite/detector choice, flags, history length and availability assumptions still need an accepted predictor specification.
- Freeze combination choices on earlier data; calibrate the combined output on the separate probability-calibration block. Later periods remain retrospective because they have already been inspected.
- Chronological roles do not establish AR/event-disjoint or prospective validation. Those require separately named evaluations.

### 1. Configuration
Run from the repository or its `notebooks/` directory using `requirements-training-lock.txt`. This preparation uses only local files, NumPy and pandas; importing the established SHARP role helper also requires PyTorch. The frozen dataset archive and parent SHARP outputs must be present. It does not download images, start GPU work or send messages to another chat.''')
    code('''from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from IPython.display import display, Markdown

REPO = Path.cwd().resolve()
if not (REPO / 'scripts' / 'prepare_multimodal72.py').exists():
    REPO = REPO.parent
assert (REPO / 'scripts' / 'prepare_multimodal72.py').is_file(), 'Run inside the Gray-Box repository'
sys.path.insert(0, str(REPO))
from scripts.prepare_multimodal72 import prepare, write_json, validate_prediction_import
from scripts.dataset_io import file_sha256
DATASET = REPO / 'data/processed/training_snapshot_20261002/gray_box_aligned_v1'
PARENT = REPO / 'outputs/sharp72_notebook_v1_20261002T080801721986Z'
TEMPORAL_CONFIG = REPO / 'configs/sharp72_temporal_v1.json'
OUTPUT = REPO / 'outputs' / ('multimodal72_preparation_v1_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
UPSTREAM = REPO.parent.parent / 'AIA Solar flare Project'
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 120)
print('New output:', OUTPUT)''')
    md('''## Data
### 2. Export cases and reconcile the SHARP baseline
All package checksums, identity joins, label masks and history timestamps are checked. Every parent SHARP case must retain its exact role and label. Cases without assembled histories stay in the full-population file. The smaller comparison file is a **candidate** cohort pending actual AIA and GOES acceptance.''')
    code('''summary, cases, support, annual, images, contract = prepare(DATASET, PARENT, TEMPORAL_CONFIG, OUTPUT)
display(pd.DataFrame([{'Full population': summary['population_cases'],
                       'Candidate comparison cases': summary['candidate_comparison_cases'],
                       'SHARP rows reconciled': summary['parent_sharp_rows_reconciled'],
                       'Unique requested images': summary['unique_requested_aia_objects'],
                       'Model fits in this notebook': summary['model_fits']}]))
display(support.rename(columns={'candidate_sharp_aia': 'Candidate cases', 'positive_candidates': 'Positive candidate windows'}))''')
    md('''### 3. Input roles and reusable upstream evidence
Read a bounded status snapshot from the existing AIA project, if available. These counts describe source processing at the saved timestamp, not complete XRS coverage or accepted predictors. Only derived status and source hashes are exported; no correspondence is copied.''')
    code('''upstream = {'snapshot_utc': datetime.now(timezone.utc).isoformat(), 'sources': {},
            'scope': 'Read-only source status; no upstream jobs launched or modified'}
legacy_path = UPSTREAM / 'run_records/19D_GOES_LEGACY_UNION_ARCHIVE_20261002.json'
if legacy_path.is_file():
    raw = legacy_path.read_bytes(); legacy = json.loads(raw)
    upstream['sources']['legacy_xrs'] = {'record': legacy_path.name, 'sha256': hashlib.sha256(raw).hexdigest(),
        'status': legacy['status'], 'predictors_created': legacy['predictors_created'],
        'historical_delivery_proven': legacy['historical_delivery_proven']}
state_path = UPSTREAM / 'inputs/goes_modern_native_inventory_20261002/state.json'
if state_path.is_file():
    raw = state_path.read_bytes(); state = json.loads(raw)
    upstream['sources']['modern_xrs'] = {'record': 'goes_modern_native_inventory_20261002/state.json',
        'sha256': hashlib.sha256(raw).hexdigest(), 'source_state_utc': state['updated_utc'],
        'files_in_inventory': len(state['records']), 'source_plan_sha256': state['plan_sha256'],
        'status': 'inventory snapshot; complete source review and predictor acceptance not inferred'}
write_json(OUTPUT / 'upstream_source_snapshot.json', upstream)
status_rows = [
    {'Component': 'SHARP', 'Here': '72-hour model fitted; frozen predictions available'},
    {'Component': 'AIA', 'Here': 'Past image references aligned; full pixel/model integration pending'},
    {'Component': 'GOES labels', 'Here': '72-hour candidate event outcomes in the frozen dataset'},
    {'Component': 'GOES predictors', 'Here': 'Past continuous XRS case requests exported; no accepted matrix here'},
]
display(pd.DataFrame(status_rows))
if 'modern_xrs' in upstream['sources']:
    r = upstream['sources']['modern_xrs']
    print(f"Upstream XRS inventory: {r['files_in_inventory']:,} file records at {r['source_state_utc']}; not model readiness.")
else:
    print('Upstream snapshot unavailable here; the portable preparation remains valid.')''')
    md('''## Results
### 4. Population and candidate support by year
Candidate support requires known labels, finite SHARP histories, past ordered AIA references and an eligible chronological role. It does not require a GOES measurement that has not yet been imported. The full population retains 2020 issues, but there are no eligible 2020 comparison cases under the frozen input/split contract.''')
    code('''import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
plot = annual.set_index('year').reindex(range(int(annual.year.min()), int(annual.year.max()) + 1), fill_value=0)
fig, ax = plt.subplots(figsize=(12, 4.8), layout='constrained')
x = np.arange(len(plot))
ax.bar(x-.18, plot.population, .36, label='Full issue population', color='#B9C1C7', edgecolor='#56616B')
ax.bar(x+.18, plot.candidate_sharp_aia, .36, label='Candidate SHARP–AIA comparison', color='#245A81', hatch='//', edgecolor='white')
ax.set(xticks=x, xticklabels=[str(y) if y != 2026 else '2026*' for y in plot.index], ylabel='Forecast cases',
       title='Case support for the 72-hour experiment · image/GOES acceptance still pending')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True); ax.legend(frameon=False, loc='upper left')
fig.text(.01, -.035, '*2026 is partial through 17 August. Unknown labels and missing histories remain in the full-population manifest.', fontsize=10)
fig.savefig(OUTPUT / 'candidate_support_by_year.png', dpi=160, bbox_inches='tight')
plt.show()
display(annual)
print(f"Candidate cases: {summary['candidate_comparison_cases']:,}; this is not the final three-input evaluation denominator.")''')
    md('''### 5. Reuse images and request past GOES measurements
The image inventory deduplicates objects across overlapping forecast histories. Blank generation/hash fields are requests for exact source receipts, not verification claims. The GOES request file specifies case IDs and observation cutoffs; it contains no invented flux values. Input availability times remain a separate requirement for an operational replay.''')
    code('''display(pd.DataFrame([
    {'Item': 'AIA history references', 'Count': summary['aia_history_references']},
    {'Item': 'Unique requested AIA objects', 'Count': summary['unique_requested_aia_objects']},
    {'Item': 'GOES case requests', 'Count': summary['candidate_comparison_cases']},
    {'Item': 'New remote downloads', 'Count': summary['downloads']},
]))
preview = cases[cases.candidate_sharp_aia_comparison].groupby('role', sort=False).head(1)
display(preview[['source_sample_id', 'role', 'issue_utc', 'label']].reset_index(drop=True))
print('Full tables:', OUTPUT / 'candidate_comparison_cases.csv.gz')''')
    md('''### 6. Reconcile saved SHARP predictions on the requested cases
This checks the integration interface using already-executed raw SHARP probabilities. It does not refit, select thresholds or inspect new performance scores. For future branches, the interface rejects changed targets/splits, silent case loss, future observations and probabilities assigned to failed inputs. Declarations and hashes alone do not prove the actual model fit; checkpoint/provenance review remains necessary.''')
    code('''parent_predictions = pd.read_csv(PARENT / 'predictions.csv.gz', float_precision='round_trip')
requested = cases[cases.candidate_sharp_aia_comparison]
anchor = requested[['forecast_case_id', 'history_96_UTC']].merge(
    parent_predictions[['forecast_case_id', 'probability_gru_mean']], on='forecast_case_id', validate='one_to_one')
anchor = anchor.rename(columns={'history_96_UTC': 'last_observation_utc', 'probability_gru_mean': 'probability'})
anchor['input_status'] = 'ok'
metadata = {k: contract[k] for k in ['horizon_hours','target_scope','target_version','dataset_manifest_sha256',
    'split_manifest_sha256','training_case_ids_sha256','probability_kind','fit_roles','selection_roles']}
metadata.update(branch='sharp', historical_availability='unverified_retrospective', combiner_fit_roles=[])
anchor_check = validate_prediction_import(anchor, metadata, contract, cases)
write_json(OUTPUT / 'sharp_import_check.json', anchor_check)
display(pd.DataFrame([anchor_check]))
rejections = []
for name, changes in [('Old 48-hour target', {'horizon_hours': 48}),
                      ('Different chronological split', {'split_manifest_sha256': 'different'}),
                      ('Model fitted on later evaluation data', {'fit_roles': ['train','retrospective_cycle25']})]:
    try:
        validate_prediction_import(anchor, {**metadata, **changes}, contract, cases)
    except ValueError as error:
        rejections.append({'Incompatible import': name, 'Result': str(error)})
    else:
        raise AssertionError('Incompatible import was accepted')
display(pd.DataFrame(rejections))''')
    md('''### 7. Model comparison and information boundaries
The added SHARP+GOES control helps distinguish GOES's contribution without AIA. For the main GOES comparison, evaluate SHARP+AIA again on exactly the triple-input cases; otherwise changing case coverage can masquerade as a modelling improvement. Whole-population availability is reported separately.''')
    code('''display(pd.DataFrame([
    {'Model': 'SHARP-only', 'Purpose': 'Existing magnetic baseline'},
    {'Model': 'AIA-only', 'Purpose': 'Image information alone'},
    {'Model': 'GOES-only', 'Purpose': 'Past full-disk XRS context baseline'},
    {'Model': 'SHARP+GOES', 'Purpose': 'GOES addition without the image branch'},
    {'Model': 'SHARP+AIA', 'Purpose': 'Two-source fusion reference'},
    {'Model': 'SHARP+AIA+GOES', 'Purpose': 'GOES added to the same SHARP+AIA cases'},
]))
display(pd.DataFrame(contract['blocks']))
print(contract['training_case_policy'])
print(contract['late_fusion'])''')
    md('''## Takeaways
### 8. Save the preparation receipt
The next compute phase is a **72-hour AIA training run with these exact roles**, after source bytes, preprocessing, cache capacity and a bounded real-data loader/model check are established. Upstream XRS source collection can continue independently. Its accepted raw source work can be reused, but the final GOES predictor specification and aligned tensor are still needed before GOES or three-source training.

No new AIA/GOES model, fusion improvement, operational readiness or independent future validation is claimed by this notebook.''')
    code('''receipt = {'status': 'completed_preparation_not_training',
           'completed_utc': datetime.now(timezone.utc).isoformat(),
           'population_cases': summary['population_cases'],
           'candidate_cases': summary['candidate_comparison_cases'],
           'sharp_import': anchor_check, 'expected_rejections': len(rejections),
           'model_fits': 0, 'remote_downloads': 0,
           'artifacts': {p.name: file_sha256(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()}}
write_json(OUTPUT / 'notebook_receipt.json', receipt)
display(Markdown(f"**Prepared {summary['candidate_comparison_cases']:,} candidate cases from {summary['population_cases']:,} total cases.** "
                 f"Reconciled {summary['parent_sharp_rows_reconciled']:,} parent SHARP rows. "
                 'AIA/GOES fitting and full pixel/predictor acceptance remain pending.'))
print('Saved receipt:', OUTPUT / 'notebook_receipt.json')''')
    nb.cells = cells
    nb.metadata = {'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.12'}}
    for i, cell in enumerate(nb.cells):
        cell['id'] = f'multimodal72-{i:02d}'
    path = Path(__file__).resolve().parents[1] / 'notebooks/07_SHARP_AIA_GOES_72h_Integration.ipynb'
    nbf.validate(nb)
    nbf.write(nb, path)
    print(path)


if __name__ == '__main__':
    build()
