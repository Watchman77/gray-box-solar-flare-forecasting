"""Verify archived replay rows, ensemble and notebook locally without any model run."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

import nbformat
import numpy as np
import pandas as pd
from scipy.special import expit

from scripts.collect_aia72_completed_replay import EXPECTED, RETURN, require


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verify(archive, archive_sha, destination):
    archive,destination=Path(archive),Path(destination)
    require(not destination.exists(),'Preserve an existing verified snapshot')
    require(digest(archive.read_bytes())==archive_sha,'Downloaded archive changed')
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers();names=[m.name for m in members]
        require(len(names)==len(set(names)) and len(names)<=122,'Duplicate/oversized snapshot')
        require(sum(m.size for m in members)<=65*1024**2,'Oversized snapshot')
        for m in members:
            p=PurePosixPath(m.name)
            require(m.isfile() and not p.is_absolute() and '..' not in p.parts
                    and p.as_posix()==m.name,'Unsafe archive member')
        data={m.name:stream.extractfile(m).read() for m in members}
    manifest=json.loads(data['snapshot_manifest.json'])
    require(set(data)==set(manifest)|{'snapshot_manifest.json','snapshot_collection.json'},'Snapshot member set differs')
    for name,pin in manifest.items():
        require(len(data[name])==pin['bytes'] and digest(data[name])==pin['sha256'],'Member changed: '+name)
    for name,pin in EXPECTED.items():require(digest(data[name])==pin,'Final known receipt/source pin differs: '+name)
    c=json.loads(data['bundle/configs/aia72_completion_replay_v1.json'])
    for name,pin in json.loads(data['bundle/replay_bundle_manifest.json'])['files'].items():
        require(digest(data['bundle/'+name])==pin,'Source differs: '+name)
    for pin in c['inputs'].values():require(digest(data['inputs/'+pin['path']])==pin['sha256'],'Input differs')
    receipt=json.loads(data['execution/execution_receipt.json']);result=json.loads(data['execution/replay/result.json'])
    returned=json.loads(data['handoff_return_to_aia.json']);ledger=json.loads(data['slot_ledger.json'])
    require(receipt['status']=='technical_checkpoint_replay_verified' and receipt['returncode']==0
            and receipt['cleanup_verified'] and receipt['worker_reaped'] and receipt['process_group_empty']
            and receipt['cleanup_errors']==[] and receipt['cancellation_signals']==[]
            and receipt['guard_error'] is None and not receipt['work_deadline_reached'],'Unverified execution')
    require(receipt['result_sha256']==digest(data['execution/replay/result.json'])
            and returned['execution_receipt_sha256']==digest(data['execution/execution_receipt.json']),
            'Final receipt chain differs')
    require(ledger['return_receipt']['sha256']==RETURN and ledger['overrun_seconds']==0
            and ledger['remaining_allowance_seconds']==0,'Final resource ledger differs')
    require(result['replayed_seeds']==[29,43] and result['seed17_replay_reused']
            and not result['seed17_repeated'] and result['fitting_steps']==0 and result['inputs_unchanged'],
            'Replay scope differs')
    require(set(result['output_sha256'])=={'three_seed_selection_predictions.csv.gz',
                'seed_29_comparison.csv.gz','seed_29_comparison.json','seed_43_comparison.csv.gz','seed_43_comparison.json'},
            'Output artifact set differs')
    for name,pin in result['output_sha256'].items():
        require(digest(data['execution/replay/'+name])==pin,'Output hash differs: '+name)
    import io
    def frame(name):return pd.read_csv(io.BytesIO(data[name]),compression='gzip')
    cases=frame('inputs/fit_cases.csv.gz');selection=cases[cases.role.eq('model_validation')].reset_index(drop=True)
    require(len(selection)==3905 and selection.forecast_case_id.is_unique,'Selection support differs')
    support=list(zip(selection.forecast_case_id.tolist(),selection.label.astype(int).tolist()))
    require(digest(json.dumps(support,sort_keys=True,separators=(',',':')).encode())==c['ordered_support_sha256'],
            'Selection identities/labels changed')
    refs={s:frame('inputs/'+c['inputs'][f'reference{s}']['path']) for s in [17,29,43]}
    def aligned(f):
        require(f.forecast_case_id.tolist()==selection.forecast_case_id.tolist()
                and np.array_equal(f.label.to_numpy(),selection.label.to_numpy()),'Rows reordered or relabelled')
    for f in refs.values():aligned(f)
    checks=[]
    for seed in [29,43]:
        f=frame(f'execution/replay/seed_{seed}_comparison.csv.gz');aligned(f)
        report=json.loads(data[f'execution/replay/seed_{seed}_comparison.json'])
        saved,got=f.saved_logit.to_numpy(),f.replayed_logit.to_numpy()
        require(np.isfinite(saved).all() and np.isfinite(got).all(),'Nonfinite logits')
        require(np.allclose(saved,refs[seed].logit.to_numpy(),atol=2e-14,rtol=1e-13),'Reference logits differ')
        delta=np.abs(got-saved);within=delta<=c['atol']+c['rtol']*np.abs(saved)
        require(within.all() and np.array_equal(within,f.within_tolerance.to_numpy()),'Prediction mismatch')
        require(np.allclose(delta,f.absolute_difference.to_numpy(),atol=2e-14,rtol=1e-13),'Stored differences disagree')
        y=f.label.to_numpy();loss=float(np.mean(np.logaddexp(0,got)-y*got))
        require(report['cases']==3905 and report['mismatched_cases']==0 and report['all_logits_within_tolerance']
                and report['atol']==1e-6 and report['rtol']==1e-5
                and abs(loss-report['replayed_log_loss'])<1e-12,'Report disagrees with original rows')
        require(abs(float(delta.max())-report['max_absolute_logit_difference'])<2e-14,'Maximum difference disagrees')
        checks.append({'seed':seed,'cases':len(f),'mismatches':int((~within).sum()),
                       'replayed_log_loss':loss,'recomputed_max_absolute_logit_difference':float(delta.max())})
    ensemble=frame('execution/replay/three_seed_selection_predictions.csv.gz');aligned(ensemble)
    probabilities=np.column_stack([expit(refs[s].logit.to_numpy()) for s in [17,29,43]])
    require(np.allclose(ensemble[[f'probability_seed_{s}' for s in [17,29,43]]].to_numpy(),
                        probabilities,atol=2e-14,rtol=1e-13),'Per-seed probabilities differ')
    require(np.allclose(ensemble.probability.to_numpy(),probabilities.mean(axis=1),atol=2e-14,rtol=1e-13),
            'Ensemble is not the mean of all three probabilities')
    name='execution/16_AIA_72h_Completed_Model_Replay_EXECUTED.ipynb'
    notebook=nbformat.reads(data[name].decode(),as_version=4);nbformat.validate(notebook)
    cells=[cell for cell in notebook.cells if cell.cell_type=='code']
    source=nbformat.reads(data['bundle/notebooks/16_AIA_72h_Completed_Model_Replay.ipynb'].decode(),as_version=4)
    require([cell.source for cell in cells]==[cell.source for cell in source.cells if cell.cell_type=='code'],
            'Executed notebook code differs')
    require(len(cells)==6 and [cell.execution_count for cell in cells]==list(range(1,7))
            and not any(o.output_type=='error' for cell in cells for o in cell.outputs),'Notebook execution incomplete')
    destination.mkdir(parents=True)
    for name,raw in data.items():
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(raw)
    verified={'status':'completed_GPU_replay_archive_and_original_predictions_verified',
              'archive_sha256':archive_sha,'archive_bytes':archive.stat().st_size,'snapshot_files':len(manifest),
              'comparisons':checks,'three_seed_probability_average_verified':True,'selection_cases':3905,
              'executed_code_cells':6,'notebook_errors':0,'notebook_code_matches_source':True,
              'notebook_visual_inspection':False,'no_model_forward_or_training_in_verification':True,
              'replay_elapsed_seconds':result['replay_elapsed_seconds'],
              'charged_reserved_seconds':ledger['charged_total_reserved_seconds'],
              'reservation_overrun_seconds':ledger['overrun_seconds'],'returned_owner_sha256':RETURN,
              'scientific_acceptance':False,'later_period_evaluation_complete':False,'fusion_complete':False}
    (destination.parent/'local_verification.json').write_text(json.dumps(verified,indent=2)+'\n')
    return verified


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive');p.add_argument('archive_sha256');p.add_argument('destination')
    a=p.parse_args();print(json.dumps(verify(a.archive,a.archive_sha256,a.destination),indent=2))
