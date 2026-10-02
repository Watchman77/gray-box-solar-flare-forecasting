"""Independently reconcile notebook 05 states, gate calculations and counts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(output, receipt):
    summary=json.loads((output/'summary.json').read_text())
    contract=json.loads((output/'run_contract.json').read_text());cfg=contract['config']
    for name,expected in summary['output_sha256'].items():assert digest(output/name)==expected,name
    out=pd.read_csv(output/'population_decisions.csv.gz')
    src=pd.read_csv(Path(cfg['rolling_dir'])/'population_sets.csv.gz').set_index('forecast_case_id').loc[out.forecast_case_id].reset_index()
    pred=pd.read_csv(Path(cfg['training_dir'])/'predictions.csv.gz')
    gates=json.loads((output/'frozen_gates.json').read_text())
    raw=np.load(Path(cfg['dataset'])/'sharp.npy',allow_pickle=False)
    transform=np.load(Path(cfg['training_dir'])/'transform.npz',allow_pickle=False)
    x=((np.sign(raw)*np.log1p(np.abs(raw))-transform['mean'])/transform['std']).astype(np.float32)[pred.tensor_row].reshape(len(pred),-1).astype(float)
    model=np.load(output/'feature_distance_model.npz',allow_pickle=False)
    centered=x-model['location']
    squared=np.einsum('ij,jk,ik->i',centered,model['precision'],centered)
    saved=out.set_index('forecast_case_id').loc[pred.forecast_case_id]
    np.testing.assert_allclose(squared,saved.feature_distance_squared,rtol=1e-9,atol=1e-8)
    seed=pred[['probability_seed_17','probability_seed_29','probability_seed_43']].to_numpy()
    spread=np.sqrt(((seed-seed.mean(axis=1)[:,None])**2).sum(axis=1)/3)
    np.testing.assert_allclose(spread,saved.seed_spread,rtol=0,atol=1e-14)
    for role,key in [('train','fit'),('model_validation','reference')]:
        ids=sorted(pred.loc[pred.role.eq(role),'forecast_case_id'])
        assert hashlib.sha256('\n'.join(ids).encode()).hexdigest()==gates[key+'_case_ids_sha256']
        assert len(ids)==gates[key+'_cases']
    ref=pred.role.eq('model_validation').to_numpy()
    for policy in cfg['policies']:
        cuts=gates['policies'][policy['name']]
        for key,values,q in [('distance',saved.feature_distance_squared.to_numpy()[ref],policy['distance_quantile']),('spread',spread[ref],policy['spread_quantile'])]:
            if q is None:assert cuts[key] is None
            else:
                exact=np.sort(values)[int(np.ceil((len(values)-1)*q))]
                assert np.isclose(exact,cuts[key],rtol=1e-12,atol=1e-14),(key,policy)
    combinations=0;state_checks=0
    for alpha in cfg['alphas']:
        ac=str(alpha).replace('.','p');chosen=gates['selected_rolling_methods']
        g=src[f'gru_{chosen["gru"]}_alpha{ac}'].to_numpy();l=src[f'logistic_{chosen["logistic"]}_alpha{ac}'].to_numpy()
        gg=src[f'gru_{chosen["gru"]}_guard'].to_numpy();lg=src[f'logistic_{chosen["logistic"]}_guard'].to_numpy()
        for policy in cfg['policies']:
            name=policy['name'];prefix=f'{name}_alpha{ac}_';cuts=gates['policies'][name]
            state=np.full(len(out),'abstain',dtype=object);decision=np.full(len(out),np.nan);probability=np.full(len(out),np.nan)
            # Explicit masked stages, separate from the production state function.
            valid=out.input_contract_ok.to_numpy() & np.isfinite(out.gru_selected) & np.isfinite(out.feature_distance_squared) & np.isfinite(out.seed_spread) & np.isfinite(g)
            if policy['support']:valid&=gg==0
            if cuts['distance'] is not None:valid&=out.feature_distance_squared.le(cuts['distance']).to_numpy()
            opposite=((g==1)&(l==2))|((g==2)&(l==1))
            if policy['conflict']:valid&=~opposite
            main=valid&((g==1)|(g==2))
            if cuts['spread'] is not None:main&=out.seed_spread.le(cuts['spread']).to_numpy()
            state[main]='normal';decision[main]=(g[main]==2).astype(int);probability[main]=out.gru_selected[main]
            if policy['fallback']:
                fallback=valid&~main&(g!=0)&((l==1)|(l==2))&(lg==0)&np.isfinite(out.logistic_selected)
                state[fallback]='degraded';decision[fallback]=(l[fallback]==2).astype(int);probability[fallback]=out.logistic_selected[fallback]
            assert np.array_equal(state,out[prefix+'state']),name
            np.testing.assert_allclose(decision,out[prefix+'decision'],rtol=0,atol=0,equal_nan=True)
            np.testing.assert_allclose(probability,out[prefix+'issued_probability'],rtol=0,atol=1e-14,equal_nan=True)
            assert out.loc[out[prefix+'state'].eq('abstain'),prefix+'issued_probability'].isna().all()
            state_checks+=len(out);combinations+=1
    def check_rows(table,state_table=False):
        checked=0
        for row in table.to_dict('records'):
            ac=str(row['alpha']).replace('.','p');prefix=f'{row["policy"]}_alpha{ac}_'
            mask=out.role.eq(row['role'])&out.label_known_primary_72h
            if state_table:mask&=out[prefix+'state'].eq(row['state'])
            if 'year' in row:mask&=pd.to_datetime(out.issue_utc,utc=True).dt.year.eq(row['year'])
            y=out.loc[mask,'candidate_primary_label_72h'].to_numpy();d=out.loc[mask,prefix+'decision'].to_numpy()
            issued=~np.isnan(d)
            expected={'cases':len(y),'issued':int(issued.sum()),'errors':int((d[issued]!=y[issued]).sum()),
              'flare_cases':int((y==1).sum()),'nonflare_cases':int((y==0).sum()),
              'true_alerts':int(((y==1)&(d==1)).sum()),'false_clears':int(((y==1)&(d==0)).sum()),
              'false_alerts':int(((y==0)&(d==1)).sum()),'true_clears':int(((y==0)&(d==0)).sum()),
              'deferred_flares':int(((y==1)&~issued).sum()),'deferred_nonflares':int(((y==0)&~issued).sum()),
              'flare_issued':int(((y==1)&issued).sum()),'nonflare_issued':int(((y==0)&issued).sum())}
            for key,value in expected.items():assert row[key]==value,(key,row)
            for metric,(num,den) in contract['rate_definitions'].items():
                if expected[den]:assert np.isclose(row[metric],expected[num]/expected[den]),metric
                else:assert pd.isna(row[metric]),metric
            checked+=1
        return checked
    metrics=check_rows(pd.read_csv(output/'metrics.csv'))
    states=check_rows(pd.read_csv(output/'state_metrics.csv'),True)
    years=check_rows(pd.read_csv(output/'yearly_metrics.csv'))
    for row in pd.read_csv(output/'availability.csv').to_dict('records'):
        b=out[pd.to_datetime(out.issue_utc,utc=True).dt.year.eq(row['year'])];p=f'{row["policy"]}_alpha{str(row["alpha"]).replace(".","p")}_'
        assert len(b)==row['population_cases']
        for state in ['normal','degraded','abstain']:assert b[p+'state'].eq(state).sum()==row[state]
        assert row['normal']+row['degraded']+row['abstain']==row['population_cases']
        assert (~b.input_contract_ok).sum()==row['missing_inputs']
    result={'status':'independent_verification_passed','output_dir':str(output),'all_output_hashes_verified':True,
      'distance_scores_independently_recomputed':len(pred),'all_gate_order_statistics_verified':True,'fit_and_reference_membership_hashes_verified':True,
      'seed_spreads_independently_recomputed':len(pred),'policy_alpha_combinations':combinations,'population_state_assignments_verified':state_checks,
      'aggregate_metric_rows_verified':metrics,'state_metric_rows_verified':states,'yearly_metric_rows_verified':years,
      'full_population_state_totals_verified':True,'all_abstentions_have_null_issued_probabilities':True}
    receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--receipt',type=Path,required=True);args=parser.parse_args();verify(args.output_dir,args.receipt)
