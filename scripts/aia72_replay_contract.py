"""Read-only contract and handoff checks; no Torch import or GPU initialization."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time


LOCK = '/home/abmoses2000/.aia19b2_run.lock'
LOCK_INODE = 519274
REVIEW_ROOT = '/home/abmoses2000/graybox_aia72_seed17_replay_v1_20261003'
CONSUMED_ROOT = '/home/abmoses2000/.graybox_gpu_review_authorizations'
CHECKPOINT = '0038ef7ea4a27f5394da99cc4410b07e7cff0e867c145b322dd31474fec2869d'
TRAINING_CONTRACT = '70fbf61bd9596e19556412121d5677f1cad083e95bd4d1c3e7679602598232db'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_hashed_json(path):
    payload = Path(path).read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def utc(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'Timezone required')
    return parsed.astimezone(timezone.utc)


def check_contract(contract):
    require(contract['kind'] == 'aia72_seed17_saved_checkpoint_replay', 'Wrong review task')
    require(contract['seed'] == 17 and contract['selected_epoch'] == 2, 'Checkpoint selection changed')
    require(contract['case_count'] == 3905 and contract['role'] == 'model_validation', 'Replay support changed')
    require(contract['fitting_allowed'] is False and contract['automatic_retry'] is False, 'Fitting/retry prohibited')
    require(contract['slot_seconds'] == 1800 and contract['cleanup_reserve_seconds'] == 60, 'Slot bound changed')
    require(contract['atol'] == 1e-6 and contract['rtol'] == 1e-5, 'Predeclared tolerance changed')
    require(contract['batch_size'] == 16 and contract['num_workers'] == 4, 'Replay batching changed')
    require(contract['inputs']['checkpoint']['sha256'] == CHECKPOINT, 'Selected checkpoint changed')
    require(contract['inputs']['training_contract']['sha256'] == TRAINING_CONTRACT, 'Training contract changed')
    require(contract['common_lock'] == LOCK and contract['common_lock_inode'] == LOCK_INODE, 'Shared lock changed')
    require(contract['minimum_free_bytes'] == 20 * 1024**3, 'Disk reserve changed')
    require(contract['review_root'] == REVIEW_ROOT and contract['consumed_root'] == CONSUMED_ROOT,
            'Canonical review/authorization root changed')
    require(contract['owner_pointer'] == contract['source_root'] + '/resource_reservation_status.json',
            'Current ownership pointer changed')
    return contract


def verify_inputs(contract, source_root=None):
    root = Path(source_root or contract['source_root']).resolve()
    resolved = {}
    for name, record in contract['inputs'].items():
        path = (root / record['path']).resolve()
        require(path.is_relative_to(root) and path.is_file(), 'Input path unavailable: ' + name)
        require(sha(path) == record['sha256'], 'Input hash changed: ' + name)
        resolved[name] = path
    return resolved


def verify_bundle(root, expected_sha):
    root = Path(root).resolve()
    manifest = root / 'replay_bundle_manifest.json'
    require(sha(manifest) == expected_sha, 'Replay bundle manifest changed')
    files = json.loads(manifest.read_text())['files']
    for name, digest in files.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and path.is_file(), 'Replay source path changed')
        require(sha(path) == digest, 'Replay source changed: ' + name)
    return files


def check_authorization(authorization, handoff, contract_sha, bundle_sha,
                        authorization_sha, prior_return, now=None):
    now = now or datetime.now(timezone.utc)
    require(authorization['status'] == 'AUTHORIZED_GPU_REVIEW', 'Separate GPU-review allowance required')
    require(authorization['kind'] == 'aia72_seed17_saved_checkpoint_replay', 'Wrong authorized work')
    require(authorization['contract_sha256'] == contract_sha, 'Authorization contract differs')
    require(authorization['bundle_sha256'] == bundle_sha, 'Authorization source differs')
    require(authorization['max_slot_seconds'] == 1800, 'Review allowance must be exactly the bounded proposal')
    require(authorization['fitting_allowed'] is False, 'Fitting is not allowed')
    require(authorization['review_root'] == REVIEW_ROOT and handoff['review_root'] == REVIEW_ROOT,
            'Allowance/handoff is for a different canonical review root')
    require(bool(authorization['user_approval_reference']), 'User allowance reference required')
    require(utc(authorization['not_before_utc']) <= now < utc(authorization['expires_utc']), 'Allowance not current')
    require(handoff['status'] == 'GPU_RELEASED_TO_GRAYBOX_REPLAY', 'A fresh replay-specific handoff is required')
    require(handoff['authorization_sha256'] == authorization_sha, 'Handoff allowance differs')
    require(handoff['contract_sha256'] == contract_sha and handoff['bundle_sha256'] == bundle_sha,
            'Handoff work differs')
    require(handoff['common_lock'] == LOCK and handoff['common_lock_inode'] == LOCK_INODE, 'Wrong handoff lock')
    issued = utc(handoff['utc'])
    require(utc(prior_return['utc']) < issued <= now, 'Old release cannot be reused')
    require((now - issued).total_seconds() <= 900, 'Handoff too old; coordinate a fresh slot')
    require(handoff['fitting_allowed'] is False, 'Handoff must exclude fitting')


def check_guards(contract, output, owner_sha, deadline):
    """Recheck control files and both reserves without scanning image-cache contents."""
    require(time.monotonic() < deadline, 'Review deadline reached')
    owner = Path(contract['owner_pointer'])
    require(owner.is_file() and sha(owner) == owner_sha, 'Current GPU owner differs from supplied handoff')
    source = Path(contract['source_root'])
    roots = [source, source / 'outputs/training', Path(contract['review_root']),
             Path(contract['review_root']) / 'bundle', Path(output)]
    for root in roots:
        if not root.is_dir():
            continue
        for marker in root.iterdir():
            if marker.name.casefold().startswith(('stop', '.stop', 'cancel', '.cancel', 'failure', 'failed')):
                raise ValueError('STOP/failure marker present: ' + str(marker))
        for name in ['notebook_status.json', 'queue_status.json', 'execution_receipt.json']:
            path = root / name
            if path.is_file():
                status = str(json.loads(path.read_text()).get('status', '')).lower()
                require('fail' not in status and 'cancel' not in status, 'Recorded failure/cancellation: ' + str(path))
    for volume in ['/home/abmoses2000', '/mnt/disks/aia-cache']:
        require(shutil.disk_usage(volume).free >= contract['minimum_free_bytes'], 'Shared disk reserve violated')


def check_fixed_review_guards(contract, output, authorization_path, authorization_sha,
                             handoff_path, handoff_sha, deadline):
    require(sha(authorization_path) == authorization_sha, 'Authorization bytes changed during review')
    require(sha(handoff_path) == handoff_sha, 'Handoff bytes changed during review')
    check_guards(contract, output, handoff_sha, deadline)
