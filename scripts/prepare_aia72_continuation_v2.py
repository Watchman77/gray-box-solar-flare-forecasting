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


NEXT_ROOT = '/home/abmoses2000/graybox_aia72_continuation_v2_20261003'
PREVIOUS_ROOT = '/home/abmoses2000/graybox_aia72_continuation_v1_20261003'
CHECKPOINT = 'b8d3c08a119f6b90b11ca2cfa4253b17647d172043fa32948e2b927df40ca6bb'
RETURN = '42f657719703913bb720bbb36ee9ac58b1d9454769fa01266837e1e5104a437c'
MANIFEST = 'd1a61714c8e949c5fff85a2047431fbafe6a801e5f542080db2e60f33df90ad5'
SOURCE_MANIFEST = '1fe682668a86480ab5a3d9c3c67c5b0f9efa700c668bcda5e9b2245dc1f3209a'
STATE = {'epoch': 3, 'cursor': 25586, 'loss_sum': 4894.803735494614,
         'best_loss': 0.36538815186923185, 'best_epoch': 1, 'stale': 1,
         'elapsed_seconds': 8201.251223390922, 'steps': 4800}


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
    require(len(saved['state']['history']) == 2, 'Pending validation history differs')
    require(not (predecessor/'training/seed_29_epoch_03_selection.csv.gz').exists(), 'Epoch3 validation already exists')
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
    del c['inputs']['checkpoint29']
    c['initial_training_files'] = {name.removeprefix('training/'): entry for name, entry in manifest.items()
                                   if name.startswith('training/')}
    c['external_allowance_policy'] = ('Proposal only for one new invocation. At most3900 total seconds, '
        'at most3600 fitting and60 cleanup. Charge all held time from a new handoff. '
        'Previous allowance consumed; no automatic retry or12hour allowance.')
    c['predecessor'] = {'review_root': PREVIOUS_ROOT, 'snapshot_manifest_sha256': MANIFEST,
        'execution_receipt_sha256': manifest['execution/execution_receipt.json']['sha256'],
        'slot_ledger_sha256': manifest['slot_ledger.json']['sha256'],
        'next_action': 'Validate epoch3 first: all25586 training cases are already processed; do not repeat them.'}
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
                     (previous['prior_return_sha256'], RETURN),
                     ('aia72_first_training_continuation_slot', c['kind'])]:
        text = replace_once(text, old, new)
    gates.write_text(text)
    write_json(bundle/'configs/aia72_continuation_v1.json', c)

    notebook = bundle/'notebooks/11_AIA_72h_Training_Continuation.ipynb'
    nb = nbformat.read(notebook, as_version=4)
    nb.cells[0].source = nb.cells[0].source.replace(
        'Seed 29 resumes epoch 2 after 18,256 cases;',
        'Seed 29 resumes at epoch 3 with all 25,586 training cases processed and validation pending;')
    nb.cells[0].source = nb.cells[0].source.replace('The first separate allowance may grant', 'A fresh separate allowance may grant')
    nb.cells[0].source = nb.cells[0].source.replace('The first short slot cannot', 'This short slot cannot')
    nb.cells[0].source += ('\n\nThe previous invocation added 2,059 steps and returned the GPU. '
        'This proposal grants no additional compute. Resume from the saved epoch-3 validation boundary; '
        'preserve the original schedule, early stopping, optimizer and RNG state. '
        'The saved best epoch remains 1; later scientific validation is pending.')
    old_preview = """    ROOT = next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'scripts/aia72_continue.py').is_file())
    archive = ROOT/'outputs/compute_limit_archive_20261003/snapshot'
    CONTINUATION_CONTEXT = {'mode':'preview','bundle_root':str(ROOT),
                            'preview_source_root':str(archive),'preview_initial':str(archive/'outputs/training')}"""
    new_preview = """    candidates = [Path.cwd(), *Path.cwd().parents]
    candidates += [p/'outputs/aia72_continuation_next_checkpoint_20261003/bundle' for p in candidates]
    ROOT = next(p for p in candidates if (p/'configs/aia72_continuation_v1.json').is_file()
                and json.loads((p/'configs/aia72_continuation_v1.json').read_text()).get('initial_seed29_sha256')
                == 'CHECKPOINT_DIGEST')
    CONTINUATION_CONTEXT = {'mode':'preview','bundle_root':str(ROOT),
                            'preview_source_root':str(ROOT.parent/'source_mirror'),
                            'preview_initial':str(ROOT.parent/'initial_training')}""".replace('CHECKPOINT_DIGEST', CHECKPOINT)
    nb.cells[1].source = replace_once(nb.cells[1].source, old_preview, new_preview)
    nbformat.validate(nb)
    nbformat.write(nb, notebook)
    nbformat.write(nb, root/'notebooks/12_AIA_72h_Next_Checkpoint_Continuation.ipynb')
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
    unused = json.loads((predecessor/'expired_unused_slot_return_20261003.json').read_text())
    seed17 = json.loads((predecessor/'training/seed_17_complete.json').read_text())
    rate = seed17['elapsed_seconds']/seed17['epochs_completed']
    proposal = {'status': 'PREPARATION_ONLY_NOT_AN_ALLOWANCE_OR_HANDOFF',
        'review_root': NEXT_ROOT, 'predecessor_return_sha256': RETURN, 'starting_checkpoint_sha256': CHECKPOINT,
        'starting_seed29_state': STATE, 'next_action': c['predecessor']['next_action'],
        'maximum_slot_seconds': 3900, 'maximum_fitting_seconds': 3600, 'cleanup_seconds': 60,
        'invocation_count': 1, 'automatic_retry': False, 'charge_from': 'fresh_handoff_utc_including_idle',
        'seed_order': [29, 43], 'seed17_repeated': False, 'original_science_configuration_unchanged': True,
        'completed_slot_accounting': {
            'original_fitting_seconds': seed17['elapsed_seconds'] + previous['initial_seed29_state']['elapsed_seconds'],
            'additional_fitting_seconds': prior_verification['additional_fitting_counter_seconds'],
            'total_fitting_counter_seconds': seed17['elapsed_seconds'] + STATE['elapsed_seconds'],
            'completed_slot_reserved_seconds': prior_verification['reserved_seconds'],
            'completed_slot_overrun_seconds': prior_verification['reservation_overrun_seconds'],
            'expired_unused_reservation_seconds': unused['actual_reserved_elapsed_seconds'],
            'expired_unused_reservation_overrun_seconds': unused['reserved_overrun_seconds'],
            'expired_unused_fitting_seconds': unused['GPU_fitting_seconds'],
            'remaining_previous_allowance_seconds': 0,
            'billing_note': 'Fitting counters and reservation intervals are different quantities; neither is a monetary invoice. Preserve the unused reservation separately.'},
        'planning_scenarios': [
            {'assumed_completed_epochs_per_remaining_seed': n,
             'additional_fitting_hours': max(0, 2*n*rate-STATE['elapsed_seconds'])/3600,
             'with_25_percent_contingency_hours': max(0, 2*n*rate-STATE['elapsed_seconds'])*1.25/3600}
            for n in [5, 6, 8, 20]],
        'estimate_basis': 'Same measured seed17 mean seconds per completed epoch, subtracting all seed29 elapsed work. Scenarios are not convergence predictions or allowances; one short slot need not complete seed29.',
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
        'checkpoint_input_note': 'Obsolete checkpoint29 entry removed from original-source inputs; new seed29 is pinned in full immutable initial_training manifest and checked state. Original source data/configuration retain their hashes.',
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
