"""Prepare an immutable local replay bundle. Does not contact the VM or launch CUDA."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import pandas as pd

from scripts.aia72_data import LAGS
from scripts.aia72_replay_contract import LOCK, LOCK_INODE, REVIEW_ROOT, CONSUMED_ROOT, check_contract, json_sha, require, sha, write_json
from scripts.build_aia72_replay_notebook import build


INPUTS = {
    'checkpoint': 'outputs/training/seed_17_best.pt',
    'completion': 'outputs/training/seed_17_complete.json',
    'reference': 'outputs/training/seed_17_epoch_02_selection.csv.gz',
    'normalization': 'outputs/training/normalization.json',
    'sources': 'outputs/training/bound_sources.json',
    'cases': 'inputs/fit_cases.csv.gz',
    'objects': 'inputs/objects.json',
    'training_contract': 'outputs/training/run_contract.json',
    'training_config': 'configs/aia72_temporal_v1.json',
    'prior_return': 'handoff_release_to_aia_20261003.json',
}
CODE = ['scripts/aia72_replay.py', 'scripts/aia72_replay_contract.py', 'scripts/run_aia72_replay.py',
        'scripts/aia72_model.py', 'scripts/aia72_vm_data.py', 'scripts/aia72_data.py',
        'scripts/aia_io.py', 'scripts/dataset_io.py', 'notebooks/10_AIA_72h_Checkpoint_Replay.ipynb',
        'configs/aia72_seed17_replay_v1.json']


def prepare(root, archive, output):
    root, archive, output = Path(root).resolve(), Path(archive).resolve(), Path(output).resolve()
    require(not output.exists(), 'Preparation output already exists; preserve the reviewed bundle')
    manifest = json.loads((archive / 'snapshot_manifest.json').read_text())['files']
    source_paths = {}
    for name, relative in INPUTS.items():
        path = archive / relative if name != 'prior_return' else archive.parent / relative
        expected = manifest[relative]['sha256'] if name != 'prior_return' else '59d27415b5acff1f11ffb67c9ec77609d18ce0ac95e26882178b5a407c22f3bf'
        require(sha(path) == expected, 'Archived source changed: ' + name)
        source_paths[name] = path
    training = json.loads(source_paths['training_contract'].read_text())
    for name in ['aia72_model.py', 'aia72_vm_data.py', 'aia72_data.py', 'aia_io.py', 'dataset_io.py']:
        require(sha(root / 'scripts' / name) == training['source_sha256'][name], 'Reused implementation changed')
    frame = pd.read_csv(source_paths['cases'])
    selected = frame.loc[frame.role.eq('model_validation')]
    contract = {
        'kind': 'aia72_seed17_saved_checkpoint_replay', 'seed': 17, 'selected_epoch': 2,
        'source_root': '/home/abmoses2000/graybox_aia72_workers4_20261002',
        'case_count': len(selected), 'role': 'model_validation',
        'ordered_support_sha256': json_sha(list(zip(selected.forecast_case_id, selected.label.astype(int)))),
        'unique_images': len(set(selected[[f'history_uri_tminus{x}' for x in LAGS]].to_numpy().ravel())),
        'batch_size': 16, 'num_workers': 4, 'max_cached_images_per_worker': 96,
        'runtime': training['runtime'], 'atol': 1e-6, 'rtol': 1e-5,
        'slot_seconds': 1800, 'cleanup_reserve_seconds': 60,
        'fitting_allowed': False, 'automatic_retry': False,
        'existing_fitting_limit_seconds': 21600, 'common_lock': LOCK, 'common_lock_inode': LOCK_INODE,
        'review_root': REVIEW_ROOT, 'consumed_root': CONSUMED_ROOT,
        'owner_pointer': '/home/abmoses2000/graybox_aia72_workers4_20261002/resource_reservation_status.json',
        'minimum_free_bytes': 20 * 1024**3,
        'inputs': {name: {'path': INPUTS[name], 'sha256': sha(path)} for name, path in source_paths.items()},
        'expected_artifacts': ['execution_receipt.json', 'execution.log', 'progress.json',
                               'prediction_comparison.csv.gz', 'comparison_result.json',
                               '10_AIA_72h_Checkpoint_Replay_EXECUTED.ipynb'],
        'scientific_scope': 'Technical saved-checkpoint reproducibility on the earlier selection block; shared implementation remains; no new fitting, selection, later-period inference, calibration or fusion.',
        'deadline_behavior': 'Stop work before reserved cleanup; terminate worker process group if needed; never accept partial results or auto-retry. OS-level unkillable processes are a failed cleanup, not permission to release the reservation.'}
    check_contract(contract)
    write_json(root / 'configs/aia72_seed17_replay_v1.json', contract)
    build()
    output.mkdir(parents=True)
    bundle, mirror = output / 'bundle', output / 'source_mirror'
    for name in CODE:
        target = bundle / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    for name, path in source_paths.items():
        target = mirror / INPUTS[name]; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    bundle_files = {name: sha(bundle / name) for name in CODE}
    write_json(bundle / 'replay_bundle_manifest.json', {'files': bundle_files})
    receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'prepared_pending_local_checks_and_compute_review',
               'bundle': str(bundle), 'preview_source_root': str(mirror),
               'contract_sha256': sha(bundle / 'configs/aia72_seed17_replay_v1.json'),
               'bundle_sha256': sha(bundle / 'replay_bundle_manifest.json'),
               'preparation_did_not_launch_GPU_or_change_fitting_allowance': True,
               'requested_additional_GPU_review_seconds': 1800,
               'authorization_received': False, 'fresh_handoff_received': False, 'scientific_replay_executed': False}
    write_json(output / 'preparation.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1], args.archive, args.output), indent=2))
