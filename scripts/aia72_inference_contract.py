"""CPU-only identity and resource gates for the frozen 72h inference phase."""
from datetime import datetime,timezone
from pathlib import Path

from scripts.aia72_replay_contract import require,utc,check_fixed_review_guards,LOCK,LOCK_INODE

ROOT='/home/abmoses2000/graybox_aia72_inference_v1_20261004'
PRIOR='/home/abmoses2000/graybox_aia72_completion_replay_v1_20261004'
RETURN='d0ee440ed05554281ab1b5730ae6ce1fe0270c91f560ffba69afb848dfc56afe'
KIND='aia72_all_frozen_cases_three_seed_inference'


def check_contract(c):
    require(c['kind']==KIND and c['review_root']==ROOT and c['source_root']==ROOT+'/inputs','Wrong inference root/scope')
    require(c['scratch_root']=='/mnt/disks/aia-cache/graybox_aia72_inference_v1_20261004','Unexpected owned scratch root')
    require(c['owner_pointer']=='/home/abmoses2000/graybox_aia72_workers4_20261002/resource_reservation_status.json','Owner differs')
    require(c['common_lock']==LOCK and c['common_lock_inode']==LOCK_INODE,'Original common lock changed')
    require(c['prior_return_sha256']==RETURN,'Prior accepted replay return differs')
    require(c['seeds']==[17,29,43] and c['case_count']==96596 and c['accepted_selection_cases']==3905,'Inference support differs')
    require(c['parent_contract_sha256']=='70fbf61bd9596e19556412121d5677f1cad083e95bd4d1c3e7679602598232db','Trained model parent differs')
    require(c['fitting_allowed'] is False and c['automatic_retry'] is False and c['invocation_count']==1,'Only one inference invocation')
    require(c['maximum_slot_seconds']==14400 and c['cleanup_seconds']==60,'Inference time ceiling changed')
    require(c['minimum_free_bytes']==20*1024**3 and c['scratch_cap_bytes']==14*1024**3,'Storage reserves changed')
    require(c['maximum_download_bytes']==200*1024**3,'Transfer safety ceiling changed')
    require(c['batch_size']==16 and c['num_workers']==4 and c['download_workers']==8,'Execution bounds changed')
    require(c['role_order']==['probability_calibration','conformal_calibration','policy_validation',
                            'retrospective_cycle25','supplementary_2026','train'],'Frozen scheduling order changed')
    return c


def check_allowance(a,h,c,contract_sha,bundle_sha,auth_sha,now=None):
    now=now or datetime.now(timezone.utc)
    require(a['status']=='AUTHORIZED_GPU_INFERENCE' and h['status']=='GPU_RELEASED_TO_GRAYBOX_INFERENCE','Fresh inference allowance required')
    for r in [a,h]:
        require(r['kind']==KIND and r['review_root']==ROOT and r['contract_sha256']==contract_sha
                and r['bundle_sha256']==bundle_sha and r['fitting_allowed'] is False,'Allowance work identity differs')
    require(a['invocation_count']==1 and a['automatic_retry'] is False,'Consumed/repeated invocations prohibited')
    require(a['max_slot_seconds']==c['maximum_slot_seconds'] and a['cleanup_seconds']==60
            and a['charge_from']=='handoff_utc','Reservation counter scope changed')
    require(bool(a['user_approval_reference']),'Standing authorization reference missing')
    require(h['authorization_sha256']==auth_sha and h['prior_return_sha256']==RETURN,'Handoff lineage differs')
    require(h['common_lock']==LOCK and h['common_lock_inode']==LOCK_INODE,'Handoff lock differs')
    issued=utc(h['utc']);require(utc(a['not_before_utc'])<=issued<=now<utc(a['expires_utc']),'Allowance not current')
    charged=(now-issued).total_seconds();require(charged<=900,'Stale handoff')
    remaining=min(a['max_slot_seconds']-charged,(utc(a['expires_utc'])-now).total_seconds())
    require(remaining>90,'No useful inference time remains')
    return {'charged_before_launch_seconds':charged,'remaining_total_seconds':remaining}


def guard(c,output,authorization,auth_sha,handoff,handoff_sha,deadline):
    for root in [c['source_root'],'/home/abmoses2000/graybox_aia72_workers4_20261002',PRIOR,
                 '/home/abmoses2000/graybox_aia72_completion_training_v5_20261004']:
        check_fixed_review_guards({**c,'source_root':root},output,authorization,auth_sha,handoff,handoff_sha,deadline)
