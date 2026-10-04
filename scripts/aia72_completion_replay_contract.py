"""CPU-only resource gates for one replay of the two newly completed seeds."""
from datetime import datetime, timezone
from pathlib import Path

from scripts.aia72_replay_contract import (
    LOCK, LOCK_INODE, require, utc, check_fixed_review_guards)

ROOT = '/home/abmoses2000/graybox_aia72_completion_replay_v1_20261004'
CONSUMED = '/home/abmoses2000/.graybox_gpu_review_authorizations'
ORIGINAL = '/home/abmoses2000/graybox_aia72_workers4_20261002'
TRAINING = '/home/abmoses2000/graybox_aia72_completion_training_v5_20261004'
KIND = 'aia72_completed_seeds29_43_checkpoint_replay'
PARENT = '70fbf61bd9596e19556412121d5677f1cad083e95bd4d1c3e7679602598232db'


def check_contract(c):
    require(c['kind'] == KIND and c['review_root'] == ROOT, 'Wrong replay task/root')
    require(c['source_root'] == ROOT+'/inputs' and c['consumed_root'] == CONSUMED, 'Input/claim root changed')
    require(c['training_root'] == TRAINING and c['parent_contract_sha256'] == PARENT, 'Training parent changed')
    require(c['owner_pointer'] == ORIGINAL+'/resource_reservation_status.json', 'Owner pointer changed')
    require(c['common_lock'] == LOCK and c['common_lock_inode'] == LOCK_INODE, 'Persistent lock changed')
    require(c['seeds'] == [29, 43] and c['ensemble_seeds'] == [17, 29, 43], 'Replay seed scope changed')
    require(c['fitting_allowed'] is False and c['automatic_retry'] is False and c['invocation_count'] == 1,
            'Only one inference-only invocation is allowed')
    require(c['maximum_slot_seconds'] == 1800 and c['cleanup_seconds'] == 60, 'Replay time bound changed')
    require(c['minimum_free_bytes'] == 20*1024**3, 'Disk reserve changed')
    require(c['inputs']['prior_return']['path'] == 'prior_return.json', 'Prior return binding changed')
    digest = c['inputs']['prior_return']['sha256']
    require(isinstance(digest,str) and len(digest) == 64 and set(digest) <= set('0123456789abcdef'),
            'Final training return has not been bound')
    return c


def check_allowance(a, h, c, contract_sha, bundle_sha, auth_sha, now=None):
    now = now or datetime.now(timezone.utc)
    require(a['status'] == 'AUTHORIZED_GPU_REVIEW' and h['status'] == 'GPU_RELEASED_TO_GRAYBOX_REPLAY',
            'A fresh replay allowance/handoff is required')
    for record in [a,h]:
        require(record['kind'] == KIND and record['review_root'] == ROOT, 'Authorized replay scope changed')
        require(record['contract_sha256'] == contract_sha and record['bundle_sha256'] == bundle_sha,
                'Work identity changed')
        require(record['seeds'] == [29,43] and record['fitting_allowed'] is False, 'Fitting/replay seeds changed')
    require(a['automatic_retry'] is False and a['invocation_count'] == 1, 'Repeated invocation prohibited')
    require(type(a['max_slot_seconds']) is int and 90 < a['max_slot_seconds'] <= 1800, 'Replay allowance exceeds bound')
    require(a['cleanup_seconds'] == 60 and a['charge_from'] == 'handoff_utc', 'All held time must be charged')
    require(bool(a['user_approval_reference']), 'Authority reference required')
    require(h['authorization_sha256'] == auth_sha, 'Handoff allowance differs')
    require(h['prior_return_sha256'] == c['inputs']['prior_return']['sha256'], 'Previous training return changed')
    require(h['common_lock'] == LOCK and h['common_lock_inode'] == LOCK_INODE, 'Handoff lock differs')
    issued = utc(h['utc'])
    require(utc(a['not_before_utc']) <= issued <= now < utc(a['expires_utc']), 'Allowance not current')
    charged = (now-issued).total_seconds()
    require(charged <= 900, 'Handoff stale')
    remaining = min(a['max_slot_seconds']-charged, (utc(a['expires_utc'])-now).total_seconds())
    require(remaining > a['cleanup_seconds']+30, 'Handoff/setup exhausted replay slot')
    return {'charged_before_launch_seconds':charged, 'remaining_total_seconds':remaining,
            'remaining_work_seconds':remaining-a['cleanup_seconds']}


def guard(c, output, authorization, auth_sha, handoff, handoff_sha, deadline):
    # Respect stop/failure markers in both original and just-completed training
    # roots, as well as this replay's inputs/output, throughout the invocation.
    for source in [c['source_root'], ORIGINAL, TRAINING]:
        scoped = {**c,'source_root':source}
        check_fixed_review_guards(scoped,output,authorization,auth_sha,handoff,handoff_sha,deadline)
    for directory in [Path(TRAINING)/'training',Path(TRAINING)/'execution']:
        check_fixed_review_guards(c,directory,authorization,auth_sha,handoff,handoff_sha,deadline)
