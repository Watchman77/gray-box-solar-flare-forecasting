"""Freeze one next-checkpoint proposal locally; never issue an allowance or launch."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tarfile

import nbformat
import torch

from scripts.aia72_replay_contract import require, sha, write_json


NEXT_ROOT = '/home/abmoses2000/graybox_aia72_continuation_v3_20261003'
PREVIOUS_ROOT = '/home/abmoses2000/graybox_aia72_continuation_v2_20261003'
CHECKPOINT = '08cc00bac09719b1d21ed8fe637f229bb0b46255eec1c51231895d54c7b13948'
RETURN = 'c91bc482a36e585680b64601307ee3a71f53eb1d30da1526aef6a2a52ab73d42'
MANIFEST = '64d9acaba3f777279255dc9b8b86bd9bc7d130cc081b94fa5251099ff6c28c9f'
SOURCE_MANIFEST = 'aa4b902a879ee0b06643e82f14b0efecce127472c6be0e2e442da02c3b8b3d08'
STATE = {'epoch': 5, 'cursor': 5184, 'loss_sum': 926.9427725076675,
         'best_loss': 0.3539059323749676, 'best_epoch': 3, 'stale': 1,
         'elapsed_seconds': 11777.969757828861, 'steps': 6724}


def replace_once(text, old, new):
    require(text.count(old) == 1, 'Frozen source replacement is not unique: ' + old)
    return text.replace(old, new, 1)


def prepare(root, predecessor, original, output):
    root, predecessor, original, output = [Path(p).resolve() for p in (root, predecessor, original, output)]
    require(not output.exists(), 'Preserve every previously prepared proposal')
    require(sha(predecessor/'snapshot_manifest.json') == MANIFEST, 'Predecessor manifest changed')
    require(sha(predecessor/'bundle/continuation_bundle_manifest.json') == SOURCE_MANIFEST, 'Accepted source changed')
    manifest = json.loads((predecessor/'snapshot_manifest.json').read_text())
    source_manifest = json.loads((predecessor/'bundle/continuation_bundle_manifest.json').read_text())['files']
    for name, digest in source_manifest.items():
        require(sha(predecessor/'bundle'/name) == digest, 'Accepted source changed: ' + name)
    for path in (predecessor/'training').iterdir():
        entry = manifest['training/' + path.name]
        require(path.is_file() and sha(path) == entry['sha256'], 'Predecessor training changed: ' + path.name)
    require(sha(predecessor/'training/seed_29_resume.pt') == CHECKPOINT, 'Wrong next checkpoint')
    require(sha(predecessor/'handoff_return_to_aia.json') == RETURN, 'Wrong predecessor return')
    saved = torch.load(predecessor/'training/seed_29_resume.pt', map_location='cpu', weights_only=True)
    require({k: saved['state'][k] for k in STATE} == STATE, 'Epoch-boundary state differs')
    require(len(saved['state']['history']) == 4, 'Completed selection history differs')
    require(not (predecessor/'training/seed_29_epoch_05_selection.csv.gz').exists(), 'Epoch5 validation already exists')
    require(not (predecessor/'training/seed_29_complete.json').exists(), 'Seed29 already complete')
    require(not (predecessor/'training/seed_43_resume.pt').exists(), 'Seed43 already started')

    c = json.loads((predecessor/'bundle/configs/aia72_continuation_v1.json').read_text())
    previous = copy.deepcopy(c)
    require(saved['contract_sha256'] == c['parent_contract_sha256'], 'Original scientific parent changed')
    c.update(kind='aia72_next_checkpoint_training_continuation_slot', review_root=NEXT_ROOT,
             initial_seed29_sha256=CHECKPOINT, initial_seed29_state=STATE,
             prior_return_path=PREVIOUS_ROOT+'/handoff_return_to_aia.json', prior_return_sha256=RETURN)
    # The old source directory still contains the original pre-continuation checkpoint.
    # The new checkpoint is pinned by the complete immutable initial_training manifest.
    # All scientific support/configuration inputs continue to resolve at the original root.
    require('checkpoint29' not in c['inputs'], 'Obsolete checkpoint input reappeared')
    c['initial_training_files'] = {name.removeprefix('training/'): entry for name, entry in manifest.items()
                                   if name.startswith('training/')}
    c['external_allowance_policy'] = ('Proposal only for one new invocation. At most3900 total seconds, '
        'at most3600 fitting and60 cleanup. Charge all held time from a new handoff. '
        'Previous allowance consumed; no automatic retry or12hour allowance.')
    c['predecessor'] = {'review_root': PREVIOUS_ROOT, 'snapshot_manifest_sha256': MANIFEST,
        'execution_receipt_sha256': manifest['execution/execution_receipt.json']['sha256'],
        'slot_ledger_sha256': manifest['slot_ledger.json']['sha256'],
        'next_action': 'Resume epoch5 at cursor5184 of25586 in the saved permutation; preserve running loss, RNG and optimizer. Validate only after the remaining20402 cases.'}
    for name, entry in c['initial_training_files'].items():
        if name.startswith('seed_17_'):
            require(entry == previous['initial_training_files'][name], 'Completed seed17 changed')
    for name, entry in c['inputs'].items():
        require(sha(original/entry['path']) == entry['sha256'], 'Original science input changed: ' + name)
    for name, digest in c['original_source_sha256'].items():
        path = predecessor/'bundle/scripts'/name
        if not path.is_file():
            path = original/'scripts'/name  # Retired original scheduler is archived, not part of the continuation bundle.
        require(sha(path) == digest, 'Scientific implementation changed: ' + name)

    bundle = output/'bundle'
    bundle.mkdir(parents=True)
    for name in source_manifest:
        target = bundle/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(predecessor/'bundle'/name, target)
    gates = bundle/'scripts/aia72_continue_contract.py'
    text = gates.read_text()
    for old, new in [(PREVIOUS_ROOT, NEXT_ROOT),
                     (previous['initial_seed29_sha256'], CHECKPOINT),
                     (previous['prior_return_sha256'], RETURN)]:
        text = replace_once(text, old, new)
    gates.write_text(text)
    write_json(bundle/'configs/aia72_continuation_v1.json', c)

    notebook = bundle/'notebooks/11_AIA_72h_Training_Continuation.ipynb'
    nb = nbformat.read(notebook, as_version=4)
    nb.cells[0].source = replace_once(nb.cells[0].source,
        'Seed 29 resumes at epoch 3 with all 25,586 training cases processed and validation pending;',
        'Seed 29 resumes epoch 5 after 5,184 of 25,586 training cases, with 6,724 optimizer steps saved;')
    nb.cells[0].source = replace_once(nb.cells[0].source,
        'The previous invocation added 2,059 steps and returned the GPU. '
        'This proposal grants no additional compute. Resume from the saved epoch-3 validation boundary; '
        'preserve the original schedule, early stopping, optimizer and RNG state. '
        'The saved best epoch remains 1; later scientific validation is pending.',
        'The previous invocation added 1,924 steps and returned the GPU. '
        'This proposal grants no additional compute. Resume the remaining 20,402 cases of epoch 5; '
        'preserve the original schedule, accumulated loss, optimizer and RNG state. '
        'Four selection epochs are complete; the best is epoch 3. With patience 4 and stale count 1, '
        'the earliest patience-based completion is epoch 7 if there is no further improvement. '
        'Later scientific validation remains pending.')
    nb.cells[1].source = replace_once(nb.cells[1].source,
        'outputs/aia72_continuation_next_checkpoint_20261003/bundle',
        'outputs/aia72_continuation_epoch5_20261003/bundle')
    nb.cells[1].source = replace_once(nb.cells[1].source, previous['initial_seed29_sha256'], CHECKPOINT)
    nbformat.validate(nb)
    nbformat.write(nb, notebook)
    nbformat.write(nb, root/'notebooks/13_AIA_72h_Epoch5_Continuation.ipynb')
    shutil.copytree(predecessor/'training', output/'initial_training')
    for entry in c['inputs'].values():
        target = output/'source_mirror'/entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original/entry['path'], target)
    files = {name: sha(bundle/name) for name in source_manifest}
    write_json(bundle/'continuation_bundle_manifest.json', {'files': files})
    with tarfile.open(output/'source_bundle.tar.gz', 'x:gz') as tf:
        for name in sorted([*files, 'continuation_bundle_manifest.json']):
            tf.add(bundle/name, arcname='bundle/'+name)

    prior_verification = json.loads((predecessor/'independent_verification.json').read_text())
    unused = json.loads((root/'results/aia72_continuation_preparation_20261003/expired_unused_slot_return_20261003.json').read_text())
    earlier = json.loads((root/'results/aia72_continuation_completed_slot2_20261003/independent_verification.json').read_text())
    require(sha(root/'results/aia72_continuation_preparation_20261003/expired_unused_slot_return_20261003.json') == '5076fc84a6b9211e8cb00d4ef1203003ac0b6c2f9e1b8df749abb6197221433f', 'Unused reservation history changed')
    require(sha(root/'results/aia72_continuation_completed_slot2_20261003/independent_verification.json') == '3adb4657f5d77eda2bb860b0b8fd007f0c2c12ed15dbb05b2e9d5e42af5bf746', 'Earlier fitting history changed')
    seed17 = json.loads((predecessor/'training/seed_17_complete.json').read_text())
    rate = seed17['elapsed_seconds']/seed17['epochs_completed']
    proposal = {'status': 'PREPARATION_ONLY_NOT_AN_ALLOWANCE_OR_HANDOFF',
        'review_root': NEXT_ROOT, 'predecessor_return_sha256': RETURN, 'starting_checkpoint_sha256': CHECKPOINT,
        'starting_seed29_state': STATE, 'next_action': c['predecessor']['next_action'],
        'maximum_slot_seconds': 3900, 'maximum_fitting_seconds': 3600, 'cleanup_seconds': 60,
        'invocation_count': 1, 'automatic_retry': False, 'charge_from': 'fresh_handoff_utc_including_idle',
        'seed_order': [29, 43], 'seed17_repeated': False, 'original_science_configuration_unchanged': True,
        'completed_slot_accounting': {
            'original_fitting_seconds': 21604.531843159,
            'previous_continuation_fitting_seconds': earlier['additional_fitting_counter_seconds'],
            'additional_fitting_seconds': prior_verification['additional_fitting_counter_seconds'],
            'total_fitting_counter_seconds': seed17['elapsed_seconds'] + STATE['elapsed_seconds'],
            'completed_slot_reserved_seconds': prior_verification['reserved_seconds'],
            'completed_slot_overrun_seconds': prior_verification['reservation_overrun_seconds'],
            'previous_continuation_reserved_seconds': earlier['reserved_seconds'],
            'all_completed_continuation_reserved_seconds': earlier['reserved_seconds'] + prior_verification['reserved_seconds'],
            'expired_unused_reservation_seconds': unused['actual_reserved_elapsed_seconds'],
            'expired_unused_reservation_overrun_seconds': unused['reserved_overrun_seconds'],
            'expired_unused_fitting_seconds': unused['GPU_fitting_seconds'],
            'remaining_previous_allowance_seconds': 0,
            'billing_note': 'Fitting counters and reservation intervals are different quantities; neither is a monetary invoice. Preserve the unused reservation separately.'},
        'earliest_seed29_patience_stop_epoch_without_further_improvement': 7,
        'planning_scenarios': [
            {'assumed_seed29_completed_epochs': n29, 'assumed_seed43_completed_epochs': n43,
             'additional_fitting_hours': max(0, (n29+n43)*rate-STATE['elapsed_seconds'])/3600,
             'with_25_percent_contingency_hours': max(0, (n29+n43)*rate-STATE['elapsed_seconds'])*1.25/3600}
            for n29, n43 in [(7, 6), (8, 6), (10, 8), (20, 20)]],
        'estimate_basis': 'Same measured seed17 mean seconds per completed epoch, subtracting all seed29 elapsed work. Seed29 best epoch3 and stale1 after epoch4 imply earliest patience stop at epoch7 if no improvement. Scenarios are not convergence predictions or allowances; seed43 starts only if naturally reached within the single bounded slot.',
        'remaining_gates': ['Independent source/changed-path review', 'Exact VM CPU changed-path preflight',
            'Fresh explicit bounded allowance and AIA handoff; recheck current owner under original common lock'],
        'GPU_launch_performed': False, 'new_allowance_recorded': False, 'scientific_acceptance': False}
    write_json(output/'compute_proposal.json', proposal)
    record = {'status': 'prepared_pending_changed_path_CPU_and_independent_review',
        'contract_sha256': sha(bundle/'configs/aia72_continuation_v1.json'),
        'bundle_sha256': sha(bundle/'continuation_bundle_manifest.json'),
        'source_tar_sha256': sha(output/'source_bundle.tar.gz'),
        'source_tar_bytes': (output/'source_bundle.tar.gz').stat().st_size,
        'changed_bundle_files': [name for name in files if files[name] != source_manifest[name]],
        'unchanged_bundle_files': [name for name in files if files[name] == source_manifest[name]],
        'checkpoint_input_note': 'New mid-epoch seed29 is pinned in full immutable initial_training manifest and checked state. Original science inputs and all runtime source except three literal bindings retain their hashes.',
        'GPU_launch': False, 'new_allowance': False, 'seed17_repeated': False}
    write_json(output/'preparation.json', record)
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predecessor', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1], a.predecessor, a.original, a.output), indent=2))
