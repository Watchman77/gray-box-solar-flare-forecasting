"""Train the existing SHARP GRU on one provisional short-horizon label set."""
import argparse, json, tempfile
from pathlib import Path
import numpy as np, pandas as pd, torch
try:
    from scripts.train_sharp_temporal import SharpGRU, fit_transform, transform, train_seed, predict, metrics
except ModuleNotFoundError:
    from train_sharp_temporal import SharpGRU, fit_transform, transform, train_seed, predict, metrics

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--horizon',type=int,choices=[24,3],required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--epochs',type=int,default=30); args=ap.parse_args()
    if args.output_dir.exists(): raise ValueError('Output exists; preserve immutable runs')
    root=Path('data/processed/gray_box_aligned_v1'); labroot=Path(f'data/processed/gray_box_multihorizon_labels_{args.horizon}h_v1')
    idx=pd.read_csv(root/'input_index.csv.gz',low_memory=False); raw=np.load(root/'sharp.npy',allow_pickle=False)
    lab=pd.read_csv(labroot/f'candidate_labels_{args.horizon}h.csv.gz',low_memory=False)
    d=idx[['forecast_case_id','tensor_row','region_component_id','issue_utc']].merge(lab[['forecast_case_id','candidate_primary_label']],on='forecast_case_id',validate='one_to_one')
    d['issue']=pd.to_datetime(d.issue_utc,utc=True); d['role']='outside_blocks'
    blocks=[('train','2010-01-01','2014-01-01'),('model_validation','2014-01-01','2014-07-01'),('probability_calibration','2014-07-01','2015-01-01'),('conformal_calibration','2015-01-01','2015-07-01'),('policy_validation','2015-07-01','2020-01-01'),('retrospective_cycle25','2021-01-01','2026-01-01'),('supplementary_2026','2026-01-01','2027-01-01')]
    for r,a,b in blocks: d.loc[(d.issue>=pd.Timestamp(a,tz='UTC'))&(d.issue<pd.Timestamp(b,tz='UTC')),'role']=r
    d.loc[d.candidate_primary_label.isna(),'role']='unknown_label_excluded'; d.loc[d.tensor_row<0,'role']='missing_input_excluded'
    train=(d.role=='train').to_numpy(); val=(d.role=='model_validation').to_numpy(); known=d.candidate_primary_label.notna().to_numpy()
    if set(d.loc[train,'candidate_primary_label'].astype(int)) != {0,1} or set(d.loc[val,'candidate_primary_label'].astype(int)) != {0,1}: raise ValueError('Both classes required in train and validation')
    params=fit_transform(raw,train); x=np.zeros(raw.shape,dtype=np.float32); supported=np.isfinite(raw).all(axis=(1,2)); x[supported]=transform(raw[supported],params)
    y=d.candidate_primary_label.fillna(-1).astype(int).to_numpy(); cfg={'hidden_size':32,'head_dropout':.2,'learning_rate':.001,'weight_decay':.0001,'gradient_clip':1.,'batch_size':256,'max_epochs':args.epochs,'patience':5,'evaluation_threshold':.5}
    args.output_dir.parent.mkdir(parents=True,exist_ok=True); staging=Path(tempfile.mkdtemp(prefix=args.output_dir.name+'.incomplete-',dir=args.output_dir.parent)); np.savez(staging/'transform.npz',**params)
    tx=torch.from_numpy(x); probs=[]; records=[]
    for seed in (17,29,43):
        model,rec=train_seed(tx,y,d.role.to_numpy(),cfg,seed,staging); records.append(rec); p=np.full(len(y),np.nan); p[supported]=predict(model,tx[supported]); probs.append(p); pd.DataFrame({'forecast_case_id':d.forecast_case_id,'role':d.role,'label':y,f'probability_seed_{seed}':p}).to_csv(staging/f'predictions_seed_{seed}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    mean=np.nanmean(np.vstack(probs),axis=0); out=pd.DataFrame({'forecast_case_id':d.forecast_case_id,'role':d.role,'label':y,'probability_gru_mean':mean}); out.to_csv(staging/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0}); summary={'status':'completed_exploratory_training','horizon_hours':args.horizon,'roles':d.role.value_counts().to_dict(),'seed_records':records,'train_cases':int(train.sum()),'train_positive':int(y[train].sum()),'validation_metrics':metrics(y[val],mean[val],float(y[train].mean())),'limitations':['Provisional labels; not confirmatory truth.','Chronological roles are not event-disjoint validation.','Probabilities are uncalibrated and policy/UQ are not evaluated.','AIA and GOES predictors are not included.']}; (staging/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); staging.rename(args.output_dir); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
