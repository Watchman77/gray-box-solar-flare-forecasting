"""Independent checks of saved rolling thresholds, issued sets and metric numerators."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd


def verify(root, output):
    summary=json.loads((output/'summary.json').read_text())
    contract=json.loads((output/'run_contract.json').read_text());cfg=contract['config']
    for name,expected in summary['output_sha256'].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest()==expected,name
    parent=root/cfg['parent_dir'];fixed=root/cfg['fixed_dir']
    source=pd.read_csv(parent/'calibrated_predictions.csv.gz')
    master=pd.read_csv(root/'data/processed/training_snapshot_20261002/gray_box_aligned_v1/master_cases.csv.gz',usecols=['forecast_case_id','outcome_end_utc_72h'])
    source=source.merge(master,on='forecast_case_id',validate='one_to_one')
    issue=pd.to_datetime(source.issue_utc,utc=True)
    ready=pd.to_datetime(source.outcome_end_utc_72h,utc=True)+pd.Timedelta(hours=cfg['reporting_delay_hours'])
    known=source.label_known & source.label.isin([0,1]) & (issue>=pd.Timestamp(cfg['history_start'],tz='UTC'))
    journal=pd.read_csv(output/'threshold_journal.csv.gz',float_precision='round_trip')
    pop=pd.read_csv(output/'population_sets.csv.gz',float_precision='round_trip',low_memory=False)
    pop_issue=pd.to_datetime(pop.issue_utc,utc=True)
    updates=pd.to_datetime(journal.update_utc,utc=True)
    latest=pd.to_datetime(journal.latest_ready_utc,utc=True)
    assert (latest.isna()|(latest<=updates)).all()
    assert journal[['update_utc','lookback_days','model','alpha','label']].duplicated().sum()==0
    dates=sorted(journal.update_utc.unique())
    sample_dates=set(np.array(dates)[np.linspace(0,len(dates)-1,min(64,len(dates)),dtype=int)])
    sample_dates.update(d for d in dates if d[5:10]=='01-01')
    checked=0
    for date in sorted(sample_dates):
        update=pd.Timestamp(str(date))
        for days in cfg['lookback_days']:
            eligible=known & (issue>=update-pd.Timedelta(days=days)) & (issue<update) & (ready<=update)
            block=source[eligible]
            expected_hash=hashlib.sha256('\n'.join(sorted(block.forecast_case_id)).encode()).hexdigest()
            recorded=journal[journal.update_utc.eq(date)&journal.lookback_days.eq(days)]
            assert recorded.history_case_ids_sha256.eq(expected_hash).all()
            for row in recorded.itertuples():
                subset=block[block.label.eq(row.label)]
                p=subset[f'{row.model}_selected'].to_numpy()
                scores=p if row.label==0 else 1-p
                n=len(scores);groups=subset.region_component_id.nunique()
                rank=math.ceil((n+1)*(1-Fraction(str(row.alpha))))
                guard=n<cfg['minimum_windows_per_class'] or groups<cfg['minimum_region_components_per_class']
                infinite=guard or rank>n
                assert row.windows==n and row.region_components==groups and row.rank==rank
                assert bool(row.guarded)==guard and bool(row.include_all)==infinite
                if infinite:assert pd.isna(row.threshold)
                else:assert abs(row.threshold-sorted(scores)[rank-1])<1e-14
                checked+=1
    fixed_thresholds=json.loads((fixed/'thresholds.json').read_text())['models']
    metrics=pd.read_csv(output/'metrics.csv');annual=pd.read_csv(output/'yearly_metrics.csv')
    checked_sets=0;checked_metrics=0
    for model in cfg['models']:
        applies=pop[f'{model}_selected'].notna() & (pop_issue>=pd.Timestamp(cfg['replay_start'],tz='UTC'))
        p=pop.loc[applies,f'{model}_selected'].to_numpy()
        lookup=pop_issue[applies].dt.floor('D')
        for method in cfg['candidate_methods']:
            for alpha in cfg['alphas']:
                if method=='fixed_2015':
                    pars=fixed_thresholds[model]['class_conditional'][str(alpha)]['thresholds']
                    q=[math.inf if pars[str(k)]['include_all'] else pars[str(k)]['threshold'] for k in [0,1]]
                else:
                    days=int(method.split('_')[1][:-1]);j=journal[journal.model.eq(model)&journal.lookback_days.eq(days)&journal.alpha.eq(alpha)].copy()
                    j['update']=pd.to_datetime(j.update_utc,utc=True)
                    q=[]
                    for k in [0,1]:
                        values=j[j.label.eq(k)].set_index('update').reindex(lookup)
                        assert values.include_all.notna().all()
                        q.append(values.threshold.where(~values.include_all,np.inf).to_numpy())
                sets=np.column_stack([p<=q[0],1-p<=q[1]])
                codes=sets[:,0].astype(int)+2*sets[:,1].astype(int)
                column=f'{model}_{method}_alpha{str(alpha).replace(".","p")}'
                np.testing.assert_array_equal(codes,pop.loc[applies,column].to_numpy())
                assert pop.loc[~applies,column].isna().all()
                checked_sets+=len(codes)
                all_codes=pop[column]
                for role in cfg['evaluation_roles']:
                    mask=pop.role.eq(role)&pop.label_known_primary_72h&applies
                    y=pop.loc[mask,'candidate_primary_label_72h'].to_numpy(dtype=int)
                    c=all_codes[mask].to_numpy(dtype=int)
                    covered=((c>>y)&1).astype(bool)
                    row=metrics[metrics.role.eq(role)&metrics.model.eq(model)&metrics.method.eq(method)&metrics.alpha.eq(alpha)].iloc[0]
                    assert row.covered==covered.sum() and row.flare_covered==covered[y==1].sum()
                    assert row.both_count==(c==3).sum() and row.empty_count==(c==0).sum()
                    assert abs(row.mean_set_size-((c&1)+((c>>1)&1)).mean())<1e-12
                    years=pop_issue[mask].dt.year.to_numpy()
                    for year in set(years):
                        sub=years==year
                        r=annual[annual.role.eq(role)&annual.model.eq(model)&annual.method.eq(method)&annual.alpha.eq(alpha)&annual.year.eq(year)].iloc[0]
                        assert r.covered==covered[sub].sum() and r.flare_covered==covered[sub&(y==1)].sum()
                    checked_metrics+=1
    # Recompute earlier selection from stored policy-case membership, without imported fit functions.
    sel=json.loads((output/'selection.json').read_text())
    policy=pop.role.eq('policy_validation')&pop.label_known_primary_72h
    y=pop.loc[policy,'candidate_primary_label_72h'].to_numpy(dtype=int)
    for model in cfg['models']:
        candidates=[]
        for order,method in enumerate(cfg['candidate_methods']):
            c=pop.loc[policy,f'{model}_{method}_alpha0p1'].to_numpy(dtype=int)
            covered=((c>>y)&1).astype(bool)
            deficit=max(0, .9-covered[y==0].mean(), .9-covered[y==1].mean())
            size=((c&1)+((c>>1)&1)).mean()
            candidates.append((deficit,size,order,method))
        assert min(candidates)[-1]==sel['selected_methods'][model]
    return {'independent_sampled_update_dates':len(sample_dates),'independently_verified_threshold_rows':checked,
            'all_journal_latest_availability_before_cutoff':True,'all_issued_set_memberships_verified':checked_sets,
            'aggregate_metric_comparisons_verified':checked_metrics,'all_annual_coverage_numerators_verified':True,
            'earlier_method_selection_independently_reproduced':True,'missing_and_before_start_sets_remain_null':True,
            'all_output_hashes_verified':True,'output_dir':str(output.relative_to(root))}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    result=verify(root,args.output_dir.resolve())
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
