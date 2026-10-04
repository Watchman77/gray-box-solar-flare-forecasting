"""CPU-only collection after the explicit v5 training return; never starts training."""
import argparse
import fcntl,hashlib,json,os,shutil,subprocess,sys,tarfile
from pathlib import Path
from datetime import datetime,timezone


def main(expected):
    if not __debug__:
        raise RuntimeError('Verification requires assertions; do not run Python with -O')
    root=Path('/home/abmoses2000/graybox_aia72_completion_training_v5_20261004')
    execution=root/'execution';training=root/'training'
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def write_new(p,value):
        with p.open('x') as f:f.write(json.dumps(value,indent=2)+'\n')
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['', '-1']
    return_path=root/'handoff_return_to_aia.json';assert sha(return_path)==expected
    returned=json.loads(return_path.read_text())
    assert returned['status']=='GPU_RELEASED_TO_AIA' and returned['review_root']==str(root)
    owner=Path('/home/abmoses2000/graybox_aia72_workers4_20261002/resource_reservation_status.json')
    with Path('/home/abmoses2000/.aia19b2_run.lock').open('r+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);assert os.fstat(lease.fileno()).st_ino==519274
        assert sha(owner)==expected,'Owner differs from explicit return'
        ps=subprocess.check_output(['ps','-eo','pid,ppid,pgid,stat,args'],text=True,timeout=8)
        live=[]
        for line in ps.splitlines()[1:]:
            row=line.split(None,4)
            if len(row)>=4 and not row[3].startswith('Z') and (int(row[0]) in [1699559,1699562] or int(row[2]) in [1699559,1699562]):live.append(line)
        assert not live,'Owned descendants remain'
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=8).strip()
        free={p:shutil.disk_usage(p).free for p in ['/home/abmoses2000','/mnt/disks/aia-cache']}
        assert min(free.values())>=20*1024**3
        clean={'utc':datetime.now(timezone.utc).isoformat(),'common_lock_inode':519274,'lock_acquired':True,
               'owner_and_return_sha256':expected,'controller_and_worker_groups_empty':True,'GPU_empty':True,'disk_free_bytes':free}
        import torch,pandas as pd,numpy as np,nbformat
        assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
        cpath=root/'bundle/configs/aia72_continuation_v1.json'
        assert sha(cpath)=='e312fbbc89a5cb803ab14945c001bb9203a6d819c40f5f10c0d03052fe5a5563'
        c=json.loads(cpath.read_text())
        manifest_path=root/'bundle/continuation_bundle_manifest.json'
        assert sha(manifest_path)=='a1440d43ba382c4afa6fb3ffd44b08c0d10dfd173d2c37c47359f6a237e10128'
        for name,digest in json.loads(manifest_path.read_text())['files'].items():assert sha(root/'bundle'/name)==digest
        for record in c['inputs'].values():
            assert sha(Path(c['source_root'])/record['path'])==record['sha256'],'Original scientific input changed'
        receipt=json.loads((execution/'execution_receipt.json').read_text())
        result=json.loads((execution/'training_result.json').read_text())
        ledger=json.loads((root/'slot_ledger.json').read_text())
        assert receipt['status']=='bounded_training_invocation_verified' and receipt['returncode']==0
        assert receipt['contract_sha256']==sha(cpath) and receipt['bundle_sha256']==sha(manifest_path)
        assert receipt['process_group_empty'] and not receipt['work_deadline_reached']
        assert not receipt.get('cancellation_signals') and receipt.get('guard_error') is None
        assert sha(execution/'training_result.json')==receipt['result_sha256']
        assert sha(execution/'execution_receipt.json')==returned['execution_receipt_sha256']
        assert ledger['status']=='reservation_returned' and ledger['remaining_allowance_seconds']==0
        assert ledger['overrun_seconds']==0 and ledger['charged_total_reserved_seconds']<=129600
        assert ledger['return_receipt']['sha256']==expected
        assert ledger['authorization_sha256']==receipt['authorization_sha256']
        assert result['seed17_repeated'] is False and result['later_period_evaluation'] is False
        assert [row['seed'] for row in result['checkpoints']] in [[29],[29,43]]
        for name,digest in receipt['output_files'].items():assert sha(training/name)==digest,'Output changed: '+name
        for name,record in c['initial_training_files'].items():
            assert sha(root/'initial_training'/name)==record['sha256'],'Immutable initial artifact changed'
            if name.startswith('seed_17_') or (name.startswith('seed_29_epoch_') and '_selection.' in name):
                assert sha(training/name)==record['sha256'],'Preserved training artifact changed'
        frame=pd.read_csv(Path(c['source_root'])/'inputs/fit_cases.csv.gz');selected=frame[frame.role.eq('model_validation')]
        checkpoints=[];recomputed=[]
        for row in result['checkpoints']:
            seed=row['seed'];assert seed in [29,43]
            checkpoint=training/f'seed_{seed}_resume.pt'
            assert sha(checkpoint)==row['resume_sha256']
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True);state=saved['state']
            assert saved['contract_sha256']==c['parent_contract_sha256']
            assert state['steps']==row['steps'] and state['epoch']==row['epoch'] and state['cursor']==row['cursor']
            assert 0<=state['cursor']<=25586 and (state['cursor']%16==0 or state['cursor']==25586)
            assert set(saved['rng'])=={'python','numpy','torch','cuda'} and len(saved['rng']['cuda'])==1
            for value in saved['model'].values():assert torch.isfinite(value).all()
            for opt in saved['optimizer']['state'].values():
                for value in opt.values():
                    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
            initial_history=[]
            if seed==29:
                initial=torch.load(root/'initial_training/seed_29_resume.pt',map_location='cpu',weights_only=True)
                initial_history=initial['state']['history']
                assert state['history'][:len(initial_history)]==initial_history
                assert state['steps']>=10876
            assert [h['epoch'] for h in state['history']]==list(range(1,len(state['history'])+1))
            for history in state['history']:
                epoch=history['epoch'];pred=pd.read_csv(training/f'seed_{seed}_epoch_{epoch:02d}_selection.csv.gz')
                assert len(pred)==3905 and np.isfinite(pred.logit).all()
                assert pred.forecast_case_id.tolist()==selected.forecast_case_id.tolist()
                assert pred.label.astype(int).tolist()==selected.label.astype(int).tolist()
                loss=float(np.mean(np.logaddexp(0,pred.logit)-pred.label*pred.logit))
                assert abs(loss-history['selection_log_loss'])<1e-12
                recomputed.append({'seed':seed,'epoch':epoch,'selection_cases':len(pred),'log_loss':loss})
            done=training/f'seed_{seed}_complete.json'
            assert done.exists()==row['fit_complete']
            if done.exists():
                completion=json.loads(done.read_text())
                assert completion['contract_sha256']==c['parent_contract_sha256']
                assert sha(training/f'seed_{seed}_best.pt')==completion['best_checkpoint_sha256']
                assert completion['status']=='seed_fit_complete_pending_replay' and completion['seed']==seed
                assert state['epoch']==len(state['history'])+1 and state['cursor']==0 and state['loss_sum']==0
                assert state['steps']==1600*len(state['history'])
                chosen=min(state['history'],key=lambda h:h['selection_log_loss'])
                assert state['best_epoch']==chosen['epoch']==completion['best_epoch']
                assert state['best_loss']==chosen['selection_log_loss']==completion['selection_log_loss']
                assert state['stale']==len(state['history'])-chosen['epoch']
                assert state['stale']>=4 or len(state['history'])==20
                assert 1<=len(state['history'])<=20 and completion['epochs_completed']==len(state['history'])
                assert completion['elapsed_seconds']==state['elapsed_seconds']
            checkpoints.append({'seed':seed,'sha256':sha(checkpoint),'state':{k:v for k,v in state.items() if k!='history'},
                                'completed_selection_epochs':len(state['history']),'fit_complete':done.exists()})
        assert sum(x['state']['steps'] for x in checkpoints)-10876==result['additional_steps']
        assert any(x['seed']==29 for x in checkpoints)
        assert result['remaining_seeds_fit_complete']==(len(checkpoints)==2 and all(x['fit_complete'] for x in checkpoints))
        nbpath=execution/'11_AIA_72h_Training_Continuation_EXECUTED.ipynb'
        nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb)
        cells=[cell for cell in nb.cells if cell.cell_type=='code']
        assert len(cells)==6 and [cell.execution_count for cell in cells]==list(range(1,7))
        assert not any(o.output_type=='error' for cell in cells for o in cell.outputs)
        assert not torch.cuda.is_initialized()
        assert sha(owner)==expected and sha(return_path)==expected
        assert all(shutil.disk_usage(p).free>=20*1024**3 for p in ['/home/abmoses2000','/mnt/disks/aia-cache'])
        archive_dir=root/'completed_invocation_archive';archive_dir.mkdir(exist_ok=False)
        verification={'utc':datetime.now(timezone.utc).isoformat(),'status':'bounded_invocation_independently_verified_and_explicit_return_confirmed',
         'cleanup':clean,'checkpoints':checkpoints,'all_selection_losses_recomputed':recomputed,
         'additional_steps':result['additional_steps'],
         'additional_fitting_counter_seconds':sum(x['state']['elapsed_seconds'] for x in checkpoints)-18927.444897180772,
         'reserved_seconds':ledger['charged_total_reserved_seconds'],'reservation_overrun_seconds':ledger['overrun_seconds'],
         'seed17_unchanged':True,'seed43_started':any(x['seed']==43 for x in checkpoints),'notebook_executed_cells':6,
         'notebook_errors':0,'CUDA_initialized_by_verification':False,'scientific_acceptance':False,
         'remaining_seeds_fit_complete':result['remaining_seeds_fit_complete'],
         'total_analysis_priority_returned':False,
         'controller_and_worker_dead_and_common_lock_held_through_archive':True,
         'inherited_legacy_files':'training/learning_curves.png, invocation_result.json, notebook_status.json and benchmark.json are inherited; use execution/training_result.json and current executed notebook.'}
        write_new(archive_dir/'independent_verification.json',verification)
        files=[]
        for directory in [root/'bundle',training,execution,root/'aia_coordination_20261004']:
            files.extend(p for p in directory.rglob('*') if p.is_file())
        files += [return_path,root/'slot_ledger.json']
        assert all(not p.is_symlink() for p in files)
        manifest={str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
        write_new(archive_dir/'snapshot_manifest.json',manifest)
        archive=archive_dir/'snapshot.tar.gz'
        with tarfile.open(archive,'x:gz') as tar:
            for p in files:tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
            for name in ['snapshot_manifest.json','independent_verification.json']:tar.add(archive_dir/name,arcname=name,recursive=False)
        assert all(sha(root/name)==value['sha256'] for name,value in manifest.items())
        assert sha(owner)==expected and sha(return_path)==expected
        assert all(shutil.disk_usage(p).free>=20*1024**3 for p in ['/home/abmoses2000','/mnt/disks/aia-cache'])
        record={'utc':datetime.now(timezone.utc).isoformat(),'status':'completed_invocation_archived_pending_local_copy',
         'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
         'source_file_count':len(manifest),'manifest_sha256':sha(archive_dir/'snapshot_manifest.json'),
         'verification_sha256':sha(archive_dir/'independent_verification.json'),'verification':verification}
        write_new(archive_dir/'archive_receipt.json',record)
        print(json.dumps(record,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--return-sha256",required=True)
    args=parser.parse_args()
    main(args.return_sha256)
