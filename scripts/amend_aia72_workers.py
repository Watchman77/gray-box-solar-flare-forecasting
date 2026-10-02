"""Version a stopped run for 2→4 loader workers, preserving all learned state."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def amend(source, target, test_receipt):
    source, target, test_receipt = map(Path, [source, target, test_receipt])
    if target.exists():
        raise ValueError('Preserve any previous amendment destination')
    test = json.loads(test_receipt.read_text())
    if test.get('status') != 'worker_count_resume_equivalence_passed':
        raise ValueError('Worker-count restart test must pass first')
    lock = Path('/home/abmoses2000/.aia19b2_run.lock')
    with lock.open('r+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise ValueError('GPU still occupied')
        for path in [source, Path('/mnt/disks/aia-cache')]:
            if shutil.disk_usage(path).free < 20 * 1024**3 + 200 * 1024**2:
                raise ValueError('Insufficient shared disk reserve')
        if json.loads((source / 'queue_status.json').read_text()).get('returncode') != 0:
            raise ValueError('Previous notebook has not exited normally')
        progress = json.loads((source / 'outputs/training/progress.json').read_text())
        if progress['status'] != 'checkpointed_incomplete':
            raise ValueError('A controlled incomplete checkpoint is required')
        old_contract_path = source / 'outputs/training/run_contract.json'
        old = json.loads(old_contract_path.read_text())
        config_name = 'configs/aia72_temporal_v1.json'
        old_config = json.loads((source / config_name).read_text())
        if old_config['num_workers'] != 2 or old['configuration'] != old_config:
            raise ValueError('Unexpected parent configuration')
        manifest = json.loads((source / 'bundle_manifest.json').read_text())
        target.mkdir()
        for name, digest in manifest['files'].items():
            path = source / name
            if sha(path) != digest:
                raise ValueError('Parent bundle changed')
            dest = target / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
        new_config = {**old_config, 'num_workers': 4}
        write(target / config_name, new_config)
        new = copy.deepcopy(old)
        new['configuration'] = new_config
        new['configuration_sha256'] = sha(target / config_name)
        output = target / 'outputs/training'; output.mkdir(parents=True)
        for name in ['normalization.json', 'bound_sources.json']:
            shutil.copy2(source / 'outputs/training' / name, output / name)
        if sha(output / 'normalization.json') != old['normalization_sha256'] or sha(output / 'bound_sources.json') != old['bound_sources_sha256']:
            raise ValueError('Parent fitted preprocessing or source bindings changed')
        write(output / 'run_contract.json', new)
        new_sha = sha(output / 'run_contract.json'); old_sha = sha(old_contract_path)
        migrations = []
        for path in sorted((source / 'outputs/training').glob('seed_*_resume.pt')):
            saved = torch.load(path, weights_only=True, map_location='cpu')
            if saved['contract_sha256'] != old_sha:
                raise ValueError('Parent checkpoint does not match parent contract')
            migrated = {**saved, 'contract_sha256': new_sha}
            dest = output / path.name; torch.save(migrated, dest)
            reread = torch.load(dest, weights_only=True, map_location='cpu')
            if not all(equal(saved[k], reread[k]) for k in saved if k != 'contract_sha256'):
                raise ValueError('A scientific checkpoint value changed during runtime amendment')
            migrations.append({'name': path.name, 'parent_sha256': sha(path), 'new_sha256': sha(dest),
                               'all_model_optimizer_rng_cursor_values_identical': True,
                               'seed_state': {k: saved['state'][k] for k in ['epoch', 'cursor', 'steps', 'elapsed_seconds']}})
        # The current amendment is only for the first incomplete epoch; refuse
        # migration of later selection artifacts without an explicit extension.
        if len(migrations) != 1 or list((source / 'outputs/training').glob('seed_*_best.pt')) or progress['epoch'] != 1:
            raise ValueError('This amendment only supports the first seed before first selection')
        shutil.copy2(source / 'handoff_release.json', target / 'handoff_release.json')
        receipt = {'status': 'worker_count_only_amendment_verified', 'utc': datetime.now(timezone.utc).isoformat(),
                   'reason': 'CPU-bound NPZ decoding; two busy workers on four-core VM with GPU waiting',
                   'configuration_changes': {'num_workers': {'before': 2, 'after': 4}},
                   'parent_contract_sha256': old_sha, 'new_contract_sha256': new_sha,
                   'parent_bundle_manifest_sha256': sha(source / 'bundle_manifest.json'),
                   'resume_test_receipt_sha256': sha(test_receipt), 'checkpoints': migrations,
                   'source_run_preserved': str(source), 'new_run': str(target),
                   'scientific_configuration_changed': False, 'new_images_downloaded': 0}
        write(target / 'worker_amendment.json', receipt)
        manifest['files'][config_name] = sha(target / config_name)
        manifest['files']['worker_amendment.json'] = sha(target / 'worker_amendment.json')
        write(target / 'bundle_manifest.json', manifest)
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True); parser.add_argument('--test-receipt', type=Path, required=True)
    args = parser.parse_args(); amend(args.source, args.target, args.test_receipt)
