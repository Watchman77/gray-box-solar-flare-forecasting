"""Single-slot continuation gates; no Torch imports or GPU initialization."""
from datetime import datetime, timezone
import math
import json
from pathlib import Path
from scripts.aia72_replay_contract import (LOCK, LOCK_INODE, require, sha, utc,
    verify_inputs, check_fixed_review_guards)

ROOT = '/home/abmoses2000/graybox_aia72_continuation_v1_20261003'
CONSUMED = '/home/abmoses2000/.graybox_gpu_continuation_authorizations'
PARENT = '70fbf61bd9596e19556412121d5677f1cad083e95bd4d1c3e7679602598232db'
RESUME = 'cd81fc957daa86c9022744dbd21e78e7912953c6ca14a5660e7fe77df92facc8'
RETURN = 'bcd96bf8bb9f1ac9c0d2ab871eea09d3f001a4aaac2746a3b0acd6cf17737e4e'


def verify_bundle(directory, expected):
    root = Path(directory).resolve()
    manifest = root/'continuation_bundle_manifest.json'
    require(sha(manifest) == expected, 'Continuation manifest changed')
    files = json.loads(manifest.read_text())['files']
    for name, digest in files.items():
        path = (root/name).resolve()
        require(path.is_relative_to(root) and path.is_file() and sha(path) == digest,
                'Continuation source changed: '+name)
    return files


def check_contract(c):
    require(c['kind'] == 'aia72_first_training_continuation_slot', 'Wrong continuation')
    require(c['review_root'] == ROOT and c['consumed_root'] == CONSUMED, 'Canonical root changed')
    require(c['common_lock'] == LOCK and c['common_lock_inode'] == LOCK_INODE, 'Persistent lock changed')
    require(c['parent_contract_sha256'] == PARENT and c['initial_seed29_sha256'] == RESUME, 'Parent/checkpoint changed')
    require(c['prior_return_sha256'] == RETURN, 'Prior AIA return changed')
    require(c['seeds'] == [29, 43] and c['maximum_slot_seconds'] == 3900
            and c['maximum_fitting_seconds'] == 3600 and c['cleanup_seconds'] == 60, 'Slot/seeds changed')
    require(c['invocation_count'] == 1 and c['automatic_retry'] is False, 'Only one invocation is supported')
    require(c['minimum_free_bytes'] == 20 * 1024**3, 'Disk reserve changed')
    require(c['owner_pointer'] == c['source_root'] + '/resource_reservation_status.json', 'Owner pointer changed')
    return c


def check_allowance(a, h, c, contract_sha, bundle_sha, auth_sha, now=None):
    now = now or datetime.now(timezone.utc)
    require(a['status'] == 'AUTHORIZED_GPU_TRAINING_CONTINUATION', 'A new training allowance is required')
    require(h['status'] == 'GPU_RELEASED_TO_GRAYBOX_CONTINUATION', 'Fresh continuation handoff required')
    for value in [a, h]:
        require(value['contract_sha256'] == contract_sha and value['bundle_sha256'] == bundle_sha, 'Work digest changed')
        require(value['review_root'] == ROOT and value['parent_contract_sha256'] == PARENT, 'Scope changed')
        require(value['initial_seed29_sha256'] == RESUME, 'Initial checkpoint changed')
        require(value['seeds'] == [29, 43] and value['fitting_allowed'] is True, 'Training scope changed')
    require(a['invocation_count'] == 1 and a['automatic_retry'] is False, 'Repeated invocation prohibited')
    require(bool(a['user_approval_reference']), 'Authority reference required')
    slot, fit = a['max_slot_seconds'], a['max_fitting_seconds']
    require(type(slot) is int and 60 < slot <= 3900 and type(fit) is int and 0 < fit <= 3600, 'Short slot bound exceeded')
    require(a['cleanup_seconds'] == 60 and a['charge_from'] == 'handoff_utc', 'All held time must be charged')
    require(h['authorization_sha256'] == auth_sha and h['prior_return_sha256'] == RETURN, 'Allowance/return digest changed')
    require(h['common_lock'] == LOCK and h['common_lock_inode'] == LOCK_INODE, 'Wrong handoff lock')
    issued = utc(h['utc'])
    require(utc(a['not_before_utc']) <= issued <= now < utc(a['expires_utc']), 'Allowance not current')
    require((now-issued).total_seconds() <= 900, 'Handoff stale')
    return budget(a, h, now)


def budget(a, h, now=None):
    now = now or datetime.now(timezone.utc)
    charged = (now-utc(h['utc'])).total_seconds()
    require(math.isfinite(charged) and charged >= 0, 'Clock precedes handoff')
    # The allowance expiry can shorten a slot; it can never extend it.
    remaining = min(a['max_slot_seconds']-charged, (utc(a['expires_utc'])-now).total_seconds())
    require(remaining > a['cleanup_seconds'] + 30, 'Held idle/setup exhausted slot')
    return {'charged_before_launch_seconds': charged, 'remaining_total_seconds': remaining,
            'remaining_work_seconds': remaining-a['cleanup_seconds'],
            'maximum_fitting_seconds': min(a['max_fitting_seconds'], remaining-a['cleanup_seconds']-20)}


def verify_initial_training(c, directory):
    root = Path(directory).resolve()
    expected = c['initial_training_files']
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(actual == set(expected), 'Initial training snapshot members changed')
    for name, record in expected.items():
        path = (root/name).resolve()
        require(path.is_relative_to(root) and sha(path) == record['sha256'], 'Initial snapshot changed: '+name)
    return expected


def verify_completed17(c, directory):
    for name, record in c['initial_training_files'].items():
        if name.startswith('seed_17_'):
            require(sha(Path(directory)/name) == record['sha256'], 'Completed seed17 changed')


def guard(c, output, training, authorization, auth_sha, handoff, handoff_sha, deadline):
    for path in [output, training]:
        check_fixed_review_guards(c, path, authorization, auth_sha, handoff, handoff_sha, deadline)
