"""Prepare a continuous fitting invocation; never grant compute or launch a job."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tarfile

import nbformat
import torch

from scripts.aia72_replay_contract import require, sha, write_json

NEXT_ROOT = '/home/abmoses2000/graybox_aia72_completion_training_v5_20261004'
PREVIOUS_ROOT = '/home/abmoses2000/graybox_aia72_continuation_v4_20261004'
CHECKPOINT = '455e7be13bec0d831f827f8a4b2f7aa64c19781bab480c8d3ef076466ba7dc95'
RETURN = 'e07b42187607c3966c149918892c72b028a76af65591943872fe3b0872cf5381'
MANIFEST = '01dd935c24889c896973628124592fccca5568807f4059c67c63f1ec939644cd'
SOURCE_MANIFEST = '4deb443244cb6abe4295abb4bab8f2f1de885aa07343f8466628fbd939cabd3b'
TIMING = 'bdd06cdcbc2910bfa013ccf55232a5abded01e70df61df9dd58167e9e2e1cf0d'
STATE = {'epoch': 7, 'cursor': 20416, 'loss_sum': 3458.8709086179733,
         'best_loss': 0.3539059323749676, 'best_epoch': 3, 'stale': 3,
         'elapsed_seconds': 18927.444897180772, 'steps': 10876}


def replace_once(text, old, new):
    require(text.count(old) == 1, 'Source replacement is not unique: ' + old)
    return text.replace(old, new, 1)


def prepare(root, predecessor, original, output):
    root, predecessor, original, output = map(lambda p: Path(p).resolve(), (root, predecessor, original, output))
    require(not output.exists(), 'Preserve every previously prepared package')
    require(sha(predecessor/'snapshot_manifest.json') == MANIFEST, 'Predecessor manifest changed')
    require(sha(predecessor/'bundle/continuation_bundle_manifest.json') == SOURCE_MANIFEST, 'Accepted source changed')
    manifest = json.loads((predecessor/'snapshot_manifest.json').read_text())
    source_files = json.loads((predecessor/'bundle/continuation_bundle_manifest.json').read_text())['files']
    for name, digest in source_files.items():
        require(sha(predecessor/'bundle'/name) == digest, 'Source changed: ' + name)
    for name, entry in manifest.items():
        require(sha(predecessor/name) == entry['sha256'], 'Predecessor artifact changed: ' + name)
    require(sha(predecessor/'training/seed_29_resume.pt') == CHECKPOINT, 'Wrong starting checkpoint')
    require(sha(predecessor/'handoff_return_to_aia.json') == RETURN, 'Wrong predecessor return')
    saved = torch.load(predecessor/'training/seed_29_resume.pt', map_location='cpu', weights_only=True)
    require({k: saved['state'][k] for k in STATE} == STATE and len(saved['state']['history']) == 6, 'Saved state differs')
    for name in ['seed_29_epoch_07_selection.csv.gz', 'seed_29_complete.json', 'seed_43_resume.pt']:
        require(not (predecessor/'training'/name).exists(), 'Unexpected completed work: ' + name)
    c = json.loads((predecessor/'bundle/configs/aia72_continuation_v1.json').read_text())
    previous = copy.deepcopy(c)
    c.update(kind='aia72_continuous_completion_training', review_root=NEXT_ROOT,
             initial_seed29_sha256=CHECKPOINT, initial_seed29_state=STATE,
             prior_return_path=PREVIOUS_ROOT+'/handoff_return_to_aia.json', prior_return_sha256=RETURN,
             maximum_slot_seconds=129600, maximum_fitting_seconds=126000)
    c['initial_training_files'] = {name.removeprefix('training/'): entry for name, entry in manifest.items()
                                   if name.startswith('training/')}
    c['external_allowance_policy'] = ('Prepared ceiling: one continuous invocation, at most36h held and35h fitting, '
        '60s cleanup. Stop promptly at original stopping rules; do not pad to ceiling. '
        'Fresh exact allowance/handoff required; all earlier allowances remain consumed. No automatic retry.')
    c['scheduling_priority'] = ('Gray-Box retains scheduling priority through its agreed remaining72h analysis. '
        'The unchanged controller returns an idle resource receipt at this technical boundary; '
        'AIA must not resume research until Gray-Box explicitly returns total-analysis priority.')
    c['scope'] = ('Finish seeds29 and43 only. Unchanged fitting/model/loader source. Saved-model replay, '
                  'later inference and SHARP/AIA fusion analysis follow separate verified phases.')
    c['predecessor'] = {'review_root': PREVIOUS_ROOT, 'snapshot_manifest_sha256': MANIFEST,
        'execution_receipt_sha256': manifest['execution/execution_receipt.json']['sha256'],
        'slot_ledger_sha256': manifest['slot_ledger.json']['sha256'],
        'next_action': 'Resume epoch7 at cursor20416 of25586; validate after5170 remaining cases. Preserve RNG, optimizer, running loss, histories and all original stopping rules.'}
    require(saved['contract_sha256'] == c['parent_contract_sha256'], 'Scientific parent changed')
    for name, entry in c['initial_training_files'].items():
        if name.startswith('seed_17_'):
            require(entry == previous['initial_training_files'][name], 'Seed17 changed')
    for name, entry in c['inputs'].items():
        require(sha(original/entry['path']) == entry['sha256'], 'Science input changed: ' + name)
    for name, digest in c['original_source_sha256'].items():
        path = predecessor/'bundle/scripts'/name
        if not path.is_file():
            path = original/'scripts'/name
        require(sha(path) == digest, 'Scientific implementation changed: ' + name)

    bundle = output/'bundle'
    bundle.mkdir(parents=True)
    for name in source_files:
        target = bundle/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(predecessor/'bundle'/name, target)
    gates = bundle/'scripts/aia72_continue_contract.py'
    text = gates.read_text()
    for old, new in [(PREVIOUS_ROOT, NEXT_ROOT), (previous['initial_seed29_sha256'], CHECKPOINT),
                     (previous['prior_return_sha256'], RETURN),
                     ('aia72_next_checkpoint_training_continuation_slot', c['kind']),
                     ("c['maximum_slot_seconds'] == 3900", "c['maximum_slot_seconds'] == 129600"),
                     ("c['maximum_fitting_seconds'] == 3600", "c['maximum_fitting_seconds'] == 126000"),
                     ('60 < slot <= 3900', '60 < slot <= 129600'),
                     ('0 < fit <= 3600', '0 < fit <= 126000'),
                     ('Short slot bound exceeded', 'Continuous completion bound exceeded')]:
        text = replace_once(text, old, new)
    gates.write_text(text)
    write_json(bundle/'configs/aia72_continuation_v1.json', c)

    notebook = bundle/'notebooks/11_AIA_72h_Training_Continuation.ipynb'
    nb = nbformat.read(notebook, as_version=4)
    nb.cells[0].source = '''# 15 · Complete the remaining AIA 72-hour training

## tl;dr
**Prepared, not yet launched.** Seed17 completed and its saved predictions reproduced. Resume seed29 at epoch7, cursor20416/25586 and10876 optimizer steps; seed43 follows after seed29 reaches its original stopping rule. This package changes the outer time ceiling, not the experiment.

## Context & Methods
One continuous reviewed invocation may use at most35h fitting within a36h reservation, with60s reserved for cleanup. It stops as soon as fitting finishes. The measured seed17 rate suggests about5h remaining if seed29 stops at epoch7 and seed43 takes6epochs; convergence is uncertain. The worst20+20epoch scenario is about26.2h, or32.7h with25% time contingency. These are planning estimates, not a request to spend the ceiling.

The CNN–GRU, batch16, float32, AdamW, four loader workers, original train-only normalization, patience4 and20epoch maximum remain unchanged. Preserve all previous time counters, RNG, optimizer state, case order, accumulated loss and histories. No repeat fitting of seed17.

### Key assumptions and limits
Candidate labels and historical availability remain provisional. Fitting uses2010–2013; model selection uses January–June2014. Saved-model replay and later-period inference, calibration, uncertainty and SHARP/AIA fusion remain separate phases. A time-capped checkpoint is incomplete. A technical idle receipt at this phase boundary does not return Gray-Box scheduling priority to AIA.

### 1. Read the frozen configuration
Ordinary execution performs a CPU metadata preview. Only the reviewed controller can execute fitting with the persistent shared lock.'''
    nb.cells[1].source = replace_once(nb.cells[1].source,
        'outputs/aia72_continuation_epoch6_20261004/bundle', 'outputs/aia72_completion_training_v5_20261004/bundle')
    nb.cells[1].source = replace_once(nb.cells[1].source, previous['initial_seed29_sha256'], CHECKPOINT)
    nb.cells[1].source = replace_once(nb.cells[1].source,
        "'Maximum slot minutes':65,'Maximum fitting minutes':60", "'Maximum slot hours':36,'Maximum fitting hours':35")
    nb.cells[6].source = nb.cells[6].source.replace('separately recorded short allowance', 'separately recorded continuous fitting allowance')
    nb.cells[10].source = '''## Takeaways
### 6. Preserve the actual result
Successful fitting still requires saved-model replay and later analysis. Preserve the executed notebook, checkpoint, controller receipt and idle resource receipt together. Gray-Box retains scheduling priority across the technical boundary; AIA does not resume until the agreed analysis is complete and explicitly returned.'''
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.execution_count, cell.outputs = None, []
    nbformat.validate(nb)
    nbformat.write(nb, notebook)
    nbformat.write(nb, root/'notebooks/15_AIA_72h_Completion_Training.ipynb')
    shutil.copyfile(bundle/'configs/aia72_continuation_v1.json', root/'configs/aia72_completion_training_v5.json')
    shutil.copytree(predecessor/'training', output/'initial_training')
    for entry in c['inputs'].values():
        target = output/'source_mirror'/entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original/entry['path'], target)
    files = {name: sha(bundle/name) for name in source_files}
    write_json(bundle/'continuation_bundle_manifest.json', {'files': files})
    with tarfile.open(output/'source_bundle.tar.gz', 'x:gz') as tf:
        for name in sorted([*files, 'continuation_bundle_manifest.json']):
            tf.add(bundle/name, arcname='bundle/'+name)
    timing = root/'results/aia72_continuation_v4_completed_20261004/cumulative_timing_history.json'
    require(sha(timing) == TIMING, 'Cumulative timing history changed')
    shutil.copyfile(timing, output/'cumulative_timing_history.json')
    rate = json.loads((predecessor/'training/seed_17_complete.json').read_text())['elapsed_seconds']/6
    proposal = {'status': 'PREPARATION_ONLY_NOT_AN_ALLOWANCE_OR_HANDOFF', 'review_root': NEXT_ROOT,
        'maximum_slot_seconds': 129600, 'maximum_fitting_seconds': 126000, 'cleanup_seconds': 60,
        'invocation_count': 1, 'automatic_retry': False, 'GPU_launch_performed': False,
        'starting_checkpoint_sha256': CHECKPOINT, 'predecessor_return_sha256': RETURN,
        'original_science_configuration_unchanged': True, 'seed17_repeated': False,
        'cumulative_timing_history_sha256': TIMING,
        'planning_scenarios': [{'seed29_epochs': a, 'seed43_epochs': b,
            'remaining_fitting_hours': ((a+b)*rate-STATE['elapsed_seconds'])/3600,
            'with_25_percent_contingency_hours': ((a+b)*rate-STATE['elapsed_seconds'])*1.25/3600}
            for a,b in [(7,6),(10,8),(20,20)]],
        'bound_basis': 'Measured seed17 mean time per completed epoch; subtract all seed29 accumulated fitting. Original patience/max_epochs determine actual stop.35h fitting covers32.7h maximum-epoch scenario with25% timing contingency; stop earlier whenever done.',
        'phase1': 'Continuous completion fitting29 then43; seed17 preserved. All source except contract pins/bounds and notebook metadata unchanged.',
        'phase2': 'Separate saved-best replay29/43 on3905 earlier-selection cases; retain accepted seed17 replay; freeze three-seed arithmetic-mean probability ensemble. Proposed replay bound30min from225.5s measured seed17 replay, subject to review.',
        'phase3': 'Frozen inference on96596 candidate comparison cases. Exact URI/generation/cache join and bounded streaming preparation required; no image downloads here. Estimate3x96596/3905x225.5s=4.65h compute proxy plus I/O; not a granted bound.',
        'phase4': 'Matched SHARP/AIA/fixed equal-probability fusion; earlier-only calibration/conformal/policy analysis and retrospective evaluation, with full support/missingness and paired uncertainty. Mac CPU where feasible.',
        'outside_current_completion_claim': 'Unreleased GOES predictor fusion, other forecast horizons, confirmatory labels/grouped validation and future prospective study.',
        'priority': c['scheduling_priority'], 'scientific_acceptance': False}
    write_json(output/'compute_proposal.json', proposal)
    record = {'status': 'prepared_pending_changed_path_CPU_and_independent_review',
        'contract_sha256': sha(bundle/'configs/aia72_continuation_v1.json'),
        'bundle_sha256': sha(bundle/'continuation_bundle_manifest.json'),
        'source_tar_sha256': sha(output/'source_bundle.tar.gz'), 'source_tar_bytes': (output/'source_bundle.tar.gz').stat().st_size,
        'changed_bundle_files': [n for n in files if files[n] != source_files[n]],
        'unchanged_bundle_files': [n for n in files if files[n] == source_files[n]],
        'GPU_launch': False, 'new_allowance': False, 'seed17_repeated': False}
    write_json(output/'preparation.json', record)
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['predecessor', 'original', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1], args.predecessor, args.original, args.output), indent=2))
