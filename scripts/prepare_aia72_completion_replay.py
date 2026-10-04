"""Freeze a replay package from a verified completed training archive; never launch."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from scripts.aia72_replay_contract import LOCK, LOCK_INODE, require, sha, write_json
from scripts.aia72_completion_replay_contract import ROOT, TRAINING, ORIGINAL, CONSUMED, KIND, PARENT, check_contract
from scripts.run_aia72_completion_replay import check_finished_training

TRAINING_CONFIG_SHA = 'e312fbbc89a5cb803ab14945c001bb9203a6d819c40f5f10c0d03052fe5a5563'
TRAINING_BUNDLE_SHA = 'a1440d43ba382c4afa6fb3ffd44b08c0d10dfd173d2c37c47359f6a237e10128'
OBJECTS_SHA = 'ba5b2c12dfac49a7ce78d3b168ba98eb364b99a00b4818798eb4204c4d1aa515'
SEED17_REPLAY_ROOT = '/home/abmoses2000/graybox_aia72_seed17_replay_v1_20261003/executions/20261003T123828Z'


def verified_snapshot(root, manifest_sha, verification_sha):
    """Verify a downloaded snapshot before any package output is created."""
    root = Path(root).resolve()
    manifest_path, verification_path = root/'snapshot_manifest.json', root/'independent_verification.json'
    require(sha(manifest_path) == manifest_sha, 'Snapshot manifest differs from verified download')
    require(sha(verification_path) == verification_sha, 'Independent verification differs from verified download')
    manifest = json.loads(manifest_path.read_text())
    require(isinstance(manifest, dict) and bool(manifest), 'Empty snapshot')
    for name, entry in manifest.items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts and relative.as_posix() == name,
                'Unsafe snapshot member')
        path = root/name
        require(not path.is_symlink() and path.resolve().is_relative_to(root) and path.is_file(),
                'Snapshot member unavailable: '+name)
        require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                'Snapshot member changed: '+name)
    verification = json.loads(verification_path.read_text())
    require(verification['status'] == 'bounded_invocation_independently_verified_and_explicit_return_confirmed',
            'Training archive is not independently verified')
    require(verification['remaining_seeds_fit_complete'] is True, 'Training remains incomplete')
    require(verification['CUDA_initialized_by_verification'] is False and verification['seed17_unchanged'] is True,
            'CPU verification or preserved model evidence differs')
    require(verification['reservation_overrun_seconds'] == 0, 'Training reservation requires review')
    for member_key, sha_key in [('collector_source_member','collector_source_sha256'),
                                ('collector_launcher_member','collector_launcher_sha256')]:
        name=verification[member_key]
        require(name in manifest and manifest[name]['sha256']==verification[sha_key],
                'Executed collection source is not preserved in the archive')
    return manifest, verification


def prepare(repo, completed, original, output, manifest_sha, verification_sha):
    repo, completed, original, output = [Path(p).resolve() for p in [repo, completed, original, output]]
    require(not output.exists(), 'Preserve every previously prepared package')
    manifest, verification = verified_snapshot(completed, manifest_sha, verification_sha)
    def member(name):
        require(name in manifest, 'Required artifact missing from snapshot manifest: '+name)
        return completed/name
    config_path = member('bundle/configs/aia72_continuation_v1.json')
    require(sha(config_path) == TRAINING_CONFIG_SHA, 'Wrong completed training contract')
    training = json.loads(config_path.read_text())
    source_manifest = member('bundle/continuation_bundle_manifest.json')
    require(sha(source_manifest) == TRAINING_BUNDLE_SHA, 'Wrong completed training source')
    for name, digest in json.loads(source_manifest.read_text())['files'].items():
        require(sha(member('bundle/'+name)) == digest, 'Training source changed: '+name)
    boundary = {key: member(name) for key, name in {
        'training_execution':'execution/execution_receipt.json',
        'training_result':'execution/training_result.json',
        'prior_return':'handoff_return_to_aia.json'}.items()}
    check_finished_training(boundary)
    ledger = json.loads(member('slot_ledger.json').read_text())
    require(ledger['status'] == 'reservation_returned' and ledger['remaining_allowance_seconds'] == 0
            and ledger['overrun_seconds'] == 0, 'Training resource return requires review')
    require(ledger['return_receipt']['sha256'] == sha(boundary['prior_return'])
            == verification['cleanup']['owner_and_return_sha256'], 'Training return evidence differs')
    require(verification['reserved_seconds'] == ledger['charged_total_reserved_seconds'], 'Reservation counter differs')

    # No live VM path is opened here. These source locations tell the coordinator
    # which already present immutable files can be copied after its own checks.
    inputs, local, vm_sources = {}, {}, {}
    def bind(key, path, name, vm_path, expected=None):
        digest = sha(path)
        require(expected is None or digest == expected, 'Pinned input changed: '+key)
        require(name not in [v['path'] for v in inputs.values()], 'Duplicate input destination')
        inputs[key] = {'path':name, 'sha256':digest}
        local[key], vm_sources[key] = Path(path), vm_path
    for key, original_key, name in [
        ('cases','cases','fit_cases.csv.gz'), ('normalization','normalization','normalization.json'),
        ('sources','sources','bound_sources.json'), ('training_contract','run_contract','training_contract.json')]:
        record = training['inputs'][original_key]
        bind(key, original/record['path'], name, ORIGINAL+'/'+record['path'], record['sha256'])
    bind('objects', original/'inputs/objects.json', 'objects.json', ORIGINAL+'/inputs/objects.json', OBJECTS_SHA)
    selected_epochs = {}
    for seed in [17,29,43]:
        completion = member(f'training/seed_{seed}_complete.json')
        epoch = json.loads(completion.read_text())['best_epoch']
        require(type(epoch) is int and 1 <= epoch <= 20, 'Unresolved selected epoch')
        selected_epochs[str(seed)] = epoch
        for key, name in [(f'complete{seed}', f'seed_{seed}_complete.json'),
                          (f'best{seed}', f'seed_{seed}_best.pt'),
                          (f'reference{seed}', f'seed_{seed}_epoch_{epoch:02d}_selection.csv.gz')]:
            bind(key, member('training/'+name), name, TRAINING+'/training/'+name)
    accepted = repo/'results/aia72_replay_seed17_completed_20261003'
    for key, name in [('seed17_replay_execution','execution_receipt.json'),
                      ('seed17_replay_comparison','comparison_result.json')]:
        bind(key, accepted/name, key+'.json', SEED17_REPLAY_ROOT+'/'+name)
    for key, path in boundary.items():
        name = key+'.json'
        bind(key, path, name, TRAINING+'/'+str(path.relative_to(completed)))

    c = {'kind':KIND, 'review_root':ROOT, 'source_root':ROOT+'/inputs', 'training_root':TRAINING,
         'consumed_root':CONSUMED, 'owner_pointer':ORIGINAL+'/resource_reservation_status.json',
         'common_lock':LOCK, 'common_lock_inode':LOCK_INODE, 'parent_contract_sha256':PARENT,
         'seeds':[29,43], 'ensemble_seeds':[17,29,43], 'selected_epochs':selected_epochs,
         'fitting_allowed':False, 'automatic_retry':False, 'invocation_count':1,
         'maximum_slot_seconds':1800, 'cleanup_seconds':60, 'minimum_free_bytes':20*1024**3,
         'runtime':training['runtime'], 'role':'model_validation', 'case_count':3905,
         'ordered_support_sha256':'f8dab27335622ef3220ebe86872d4b3c9fff3c7f6f7c6206de6f8484b5b982b3',
         'unique_images':4661, 'batch_size':16, 'num_workers':4, 'max_cached_images_per_worker':96,
         'atol':1e-6, 'rtol':1e-5, 'inputs':inputs, 'input_sources_on_VM':vm_sources,
         'completed_training_snapshot_sha256':manifest_sha,
         'independent_training_verification_sha256':verification_sha,
         'scientific_acceptance':False, 'total_analysis_complete':False,
         'scope':'Replay saved models 29 and 43 on original selection cases; reuse accepted model 17 replay. Average all three saved probability vectors. No fitting, later-period inference, calibration or fusion.',
         'priority':'Gray-Box retains scheduling priority through the agreed remaining 72-hour analysis. A technical GPU return at this boundary does not return total-analysis priority to AIA.'}
    check_contract(c)
    # Import the scientific reader only after the completed, verified boundary.
    from scripts.aia72_completion_replay import check_design, load_completed_support
    check_design(c)

    sources = {}
    for name in ['aia72_completion_replay.py','aia72_completion_replay_contract.py',
                 'aia72_completion_replay_worker.py','run_aia72_completion_replay.py','aia72_replay.py',
                 'aia72_replay_supervisor.py']:
        sources['scripts/'+name] = repo/'scripts'/name
    for name in ['aia72_continue.py','aia72_continue_contract.py','aia72_replay_contract.py','run_aia72_replay.py']:
        sources['scripts/'+name] = member('bundle/scripts/'+name)
    for name in ['aia72_data.py','aia72_model.py','aia72_vm_data.py','aia_io.py','dataset_io.py','train_aia72_gpu.py']:
        path = original/'scripts'/name
        require(sha(path) == training['original_source_sha256'][name], 'Original scientific code changed: '+name)
        sources['scripts/'+name] = path
    sources['notebooks/16_AIA_72h_Completed_Model_Replay.ipynb'] = repo/'notebooks/16_AIA_72h_Completed_Model_Replay.ipynb'
    for path in sources.values():
        require(path.is_file() and not path.is_symlink(), 'Source unavailable')
    output.mkdir(parents=True)
    mirror, bundle = output/'inputs', output/'bundle'
    mirror.mkdir(); bundle.mkdir()
    for key, path in local.items():
        shutil.copyfile(path, mirror/inputs[key]['path'])
    load_completed_support(c, mirror)  # Metadata, support and logits only; no image pixels.
    for name, path in sources.items():
        target = bundle/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (bundle/'configs').mkdir()
    write_json(bundle/'configs/aia72_completion_replay_v1.json', c)
    names = sorted([*sources, 'configs/aia72_completion_replay_v1.json'])
    write_json(bundle/'replay_bundle_manifest.json', {'files':{name:sha(bundle/name) for name in names}})
    with tarfile.open(output/'source_bundle.tar.gz', 'x:gz') as archive:
        for name in [*names, 'replay_bundle_manifest.json']:
            archive.add(bundle/name, arcname='bundle/'+name, recursive=False)
    record = {'status':'prepared_pending_CPU_preview_and_independent_review',
              'contract_sha256':sha(bundle/'configs/aia72_completion_replay_v1.json'),
              'bundle_sha256':sha(bundle/'replay_bundle_manifest.json'),
              'source_tar_sha256':sha(output/'source_bundle.tar.gz'),
              'source_tar_bytes':(output/'source_bundle.tar.gz').stat().st_size,
              'selected_epochs':selected_epochs, 'input_files':len(inputs),
              'preparation_image_reads':0, 'GPU_launches':0, 'new_allowance':False,
              'scientific_acceptance':False, 'total_analysis_complete':False}
    write_json(output/'preparation.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['completed','original','output']:
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ['manifest-sha256','verification-sha256']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1], args.completed, args.original,
                             args.output, args.manifest_sha256, args.verification_sha256), indent=2))
