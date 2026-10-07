"""Train the SHARP GRU on one provisional short-horizon label set."""
import argparse, json, tempfile, random
from pathlib import Path
import numpy as np, pandas as pd, torch
try:
    from scripts.train_sharp_temporal import SharpGRU, fit_transform, transform, metrics
except ModuleNotFoundError:
    from train_sharp_temporal import SharpGRU, fit_transform, transform, metrics

def fit_seed(x,y,roles,cfg,seed,out,device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type=='cuda': torch.cuda.manual_seed_all(seed)
    model=SharpGRU(x.shape[2],32,.2).to(device); tr=np.flatnonzero(roles=='train'); va=np.flatnonzero(roles=='model_validation'); prior=float(y[tr].mean())
    with torch.no_grad(): model.head[-1].bias.fill_(np.log(prior/(1-prior)))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001); loss_fn=torch.nn.BCEWithLogitsLoss(); best=float('inf'); stale=0; ck=out/f'seed_{seed}.pt'
    for epoch in range(1,cfg['epochs']+1):
        model.train(); order=np.random.permutation(tr); total=0.
        for s in range(0,len(order),256):
            ii=torch.as_tensor(order[s:s+256],dtype=torch.long,device=device); opt.zero_grad(set_to_none=True); loss=loss_fn(model(x[ii]),torch.as_tensor(y[ii.cpu().numpy()],dtype=torch.float32,device=device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); total+=float(loss)*len(ii)
        model.eval()
        with torch.no_grad(): p=torch.sigmoid(model(x[torch.as_tensor(va,dtype=torch.long,device=device)])).cpu().numpy()
        vl=metrics(y[va],p,prior)['log_loss']
        if vl<best: best=vl; best_epoch=epoch; stale=0; torch.save(model.state_dict(),ck)
        else: stale+=1
        if stale>=5: break
    model.load_state_dict(torch.load(ck,weights_only=True,map_location=device)); return model,{'seed':seed,'epochs':epoch,'best_epoch':best_epoch,'best_validation_log_loss':best}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--horizon',type=int,choices=[24,3],required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--epochs',type=int,default=30); args=ap.parse_args()
    if args.output_dir.exists(): raise ValueError('Output exists; preserve immutable runs')
    root=Path('data/processed/gray_box_aligned_v1'); labroot=Path(f'data/processed/gray_box_multihorizon_labels_{args.horizon}h_v1'); idx=pd.read_csv(root/'input_index.csv.gz',low_memory=False); raw=np.load(root/'sharp.npy',allow_pickle=False); lab=pd.read_csv(labroot/f'candidate_labels_{args.horizon}h.csv.gz',low_memory=False)
    d=idx[['forecast_case_id','tensor_row','region_component_id','issue_utc']].merge(lab[['forecast_case_id','candidate_primary_label']],on='forecast_case_id',validate='one_to_one'); d['issue']=pd.to_datetime(d.issue_utc,utc=True); d['role']='outside_blocks'
    blocks=[('train','2010-01-01','2014-01-01'),('model_validation','2014-01-01','2014-07-01'),('probability_calibration','2014-07-01','2015-01-01'),('conformal_calibration','2015-01-01','2015-07-01'),('policy_validation','2015-07-01','2020-01-01'),('retrospective_cycle25','2021-01-01','2026-01-01'),('supplementary_2026','2026-01-01','2027-01-01')]
    for r,a,b in blocks: d.loc[(d.issue>=pd.Timestamp(a,tz='UTC'))&(d.issue<pd.Timestamp(b,tz='UTC')),'role']=r
    d.loc[d.candidate_primary_label.isna(),'role']='unknown_label_excluded'; d.loc[d.tensor_row<0,'role']='missing_input_excluded'; roles=d.role.to_numpy(); train=roles=='train'; val=roles=='model_validation'; y=d.candidate_primary_label.fillna(-1).astype(int).to_numpy()
    if set(y[train])!={0,1} or set(y[val])!={0,1}: raise ValueError('Both classes required in train and validation')
    params=fit_transform(raw,train); supported=np.isfinite(raw).all(axis=(1,2)); x=np.zeros(raw.shape,dtype=np.float32); x[supported]=transform(raw[supported],params); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); tx=torch.from_numpy(x).to(device)
    args.output_dir.parent.mkdir(parents=True,exist_ok=True); staging=Path(tempfile.mkdtemp(prefix=args.output_dir.name+'.incomplete-',dir=args.output_dir.parent)); np.savez(staging/'transform.npz',**params); probs=[]; records=[]; cfg={'epochs':args.epochs}; si=torch.as_tensor(np.flatnonzero(supported),dtype=torch.long,device=device)
    for seed in (17,29,43):
        model,rec=fit_seed(tx,y,roles,cfg,seed,staging,device); records.append(rec); model.eval()
        with torch.no_grad(): p=np.full(len(y),np.nan); p[supported]=torch.sigmoid(model(tx[si])).cpu().numpy()
        probs.append(p)
    mean=np.nanmean(np.vstack(probs),axis=0); pd.DataFrame({'forecast_case_id':d.forecast_case_id,'role':d.role,'label':y,'probability_gru_mean':mean}).to_csv(staging/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0}); summary={'status':'completed_exploratory_training','horizon_hours':args.horizon,'device':str(device),'gpu_name':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'roles':d.role.value_counts().to_dict(),'seed_records':records,'train_cases':int(train.sum()),'train_positive':int(y[train].sum()),'validation_metrics':metrics(y[val],mean[val],float(y[train].mean())),'limitations':['Provisional labels; not confirmatory truth.','Chronological roles are not event-disjoint validation.','Probabilities are uncalibrated; AIA and GOES predictors are not included.']}; (staging/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); staging.rename(args.output_dir); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
