"""Independent contingency-table reconciliation for notebook 06."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def verify(output,receipt):
    s=json.loads((output/'summary.json').read_text());contract=json.loads((output/'run_contract.json').read_text());cfg=contract['config']
    for name,digest in s['output_sha256'].items():assert hashlib.sha256((output/name).read_bytes()).hexdigest()==digest,name
    cases=pd.read_csv(output/'case_diagnostics.csv.gz',float_precision='round_trip',low_memory=False)
    parent=pd.read_csv(Path(cfg['policy_dir'])/'population_decisions.csv.gz',float_precision='round_trip',low_memory=False).set_index('forecast_case_id')
    for alpha in cfg['alphas']:
        ac=str(alpha).replace('.','p');b=cases[cases.alpha.eq(alpha)];p=parent.loc[b.forecast_case_id]
        bits=(~p[f'primary_alpha{ac}_support_ok']).astype(int)+2*p[f'primary_alpha{ac}_feature_flag'].astype(int)+4*p[f'primary_alpha{ac}_spread_flag'].astype(int)+8*p[f'primary_alpha{ac}_conflicting_singletons'].astype(int)
        np.testing.assert_array_equal(bits,b.guard_mask)
        for own,old in [('baseline_decision','gru_singleton'),('full_guard_decision','guarded_gru')]:
            np.testing.assert_allclose(b[own],p[f'{old}_alpha{ac}_decision'],rtol=0,atol=0,equal_nan=True)
    def tally(y,d):
        issued=np.isfinite(d)
        return {'cases':len(y),'issued':int(issued.sum()),'errors':int((y[issued]!=d[issued]).sum()),
         'flare_cases':int((y==1).sum()),'nonflare_cases':int((y==0).sum()),
         'true_alerts':int(((y==1)&(d==1)).sum()),'false_alerts':int(((y==0)&(d==1)).sum()),
         'false_clears':int(((y==1)&(d==0)).sum()),'true_clears':int(((y==0)&(d==0)).sum()),
         'deferred_flares':int(((y==1)&~issued).sum()),'deferred_nonflares':int(((y==0)&~issued).sum()),
         'flare_issued':int(((y==1)&issued).sum()),'nonflare_issued':int(((y==0)&issued).sum())}
    metric_rows=pd.read_csv(output/'subset_metrics.csv');checked=0
    for row in metric_rows.to_dict('records'):
        b=cases[cases.alpha.eq(row['alpha'])&cases.role.eq(row['role'])&cases.label_known]
        keep=np.array([not(int(v)&row['active_mask']) for v in b.guard_mask]);d=b.baseline_decision.to_numpy().copy();d[~keep]=np.nan
        expected=tally(b.label.to_numpy(),d)
        for key,value in expected.items():assert row[key]==value,(row,key)
        for key,(num,den) in contract['metric_definitions'].items():
            if expected[den]:assert np.isclose(row[key],expected[num]/expected[den]),key
            else:assert pd.isna(row[key]),key
        checked+=1
    mapping={'true_alert':'true_alerts','false_alert':'false_alerts','false_clear':'false_clears','true_clear':'true_clears'}
    for row in pd.read_csv(output/'gate_attribution.csv').to_dict('records'):
        b=cases[cases.alpha.eq(row['alpha'])&cases.role.eq(row['role'])&cases.label_known];bit=cfg['guard_bits'][row['guard']]
        allocated={k:0. for k in mapping};standalone={k:0 for k in mapping};unique={k:0 for k in mapping}
        for mask,sub in b[b.baseline_decision.notna()].groupby('guard_mask'):
            if int(mask)&bit:
                counts=tally(sub.label.to_numpy(),sub.baseline_decision.to_numpy());n=int(mask).bit_count()
                for k,plural in mapping.items():
                    standalone[k]+=counts[plural];allocated[k]+=counts[plural]/n
                    if mask==bit:unique[k]+=counts[plural]
        for mode,values in [('standalone',standalone),('unique',unique),('allocated',allocated)]:
            for k,v in values.items():assert np.isclose(row[mode+'_'+k],v),(row,k)
            assert np.isclose(row[mode+'_removed'],sum(values.values()))
    for row in pd.read_csv(output/'fallback_routes.csv').to_dict('records'):
        b=cases[cases.alpha.eq(row['alpha'])&cases.role.eq(row['role'])&cases.label_known]
        b=b[b.fallback_route.ne('not_fallback')] if row['route']=='all_fallback' else b[b.fallback_route.eq(row['route'])]
        for k,v in tally(b.label.to_numpy(),b.fallback_decision.to_numpy()).items():assert row[k]==v,(k,row)
        p=parent.loc[b.forecast_case_id];ac=str(row['alpha']).replace('.','p')
        assert p[f'guarded_fallback_alpha{ac}_state'].eq('degraded').all()
        np.testing.assert_allclose(b.fallback_decision,p[f'guarded_fallback_alpha{ac}_decision'],rtol=0,atol=0)
    points=0
    for row in pd.read_csv(output/'paired_intervals.csv').to_dict('records'):
        t=metric_rows[metric_rows.alpha.eq(row['alpha'])&metric_rows.role.eq(row['role'])].set_index('active_mask');mask=15^cfg['guard_bits'][row['removed_guard']]
        expected=t.loc[mask,row['metric']]-t.loc[15,row['metric']]
        assert np.isclose(expected,row['difference']),row
        points+=1
    result={'status':'independent_verification_passed','output_dir':str(output),'all_output_hashes_verified':True,'parent_guard_and_decision_rows_verified':len(cases),
      'subset_metric_rows_verified':checked,'all_gate_contributions_independently_reconciled':True,'all_fallback_route_counts_reconciled':True,
      'paired_difference_point_estimates_verified':points,'operational_policy_changed':False}
    receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();verify(a.output_dir,a.receipt)
