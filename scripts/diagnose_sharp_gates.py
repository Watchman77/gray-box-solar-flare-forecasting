"""Reconcile frozen decision guards, useful alerts removed and fallback errors."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from scripts.decision_policy_sharp import sha256, save_json, checked_parent, decision_counts, decision_metrics, RATE_DEFINITIONS


def validate_gate_inputs(base, bits):
    base=np.asarray(base,dtype=float);bits=np.asarray(bits)
    if base.ndim!=1 or bits.shape!=base.shape or np.isinf(base).any() or not np.isin(base[np.isfinite(base)],[0,1]).all() or not np.isin(bits,np.arange(16)).all():
        raise ValueError('Invalid binary decisions or four-guard bitmask')
    return base,bits.astype(np.uint8)


def subset_decisions(base, bits, active_mask):
    base,bits=validate_gate_inputs(base,bits)
    if active_mask not in range(16):raise ValueError('Invalid guard subset')
    return np.where((bits & active_mask)==0,base,np.nan)


def outcome_categories(y, base):
    y=np.asarray(y);base=np.asarray(base,dtype=float)
    if not np.isin(y,[0,1]).all():raise ValueError('Unknown outcomes must not enter metrics')
    return np.select([(y==1)&(base==1),(y==0)&(base==1),(y==1)&(base==0),(y==0)&(base==0)],
      ['true_alert','false_alert','false_clear','true_clear'],default='already_deferred')


def gate_attribution(y, base, bits, config):
    base,bits=validate_gate_inputs(base,bits);categories=outcome_categories(y,base)
    issued=np.isfinite(base);hits=np.array([int(v).bit_count() for v in bits])
    rows=[]
    for guard,bit in config['guard_bits'].items():
        triggered=issued & ((bits&bit)!=0)
        unique=issued & (bits==bit)
        weights=np.divide(triggered.astype(float),hits,out=np.zeros(len(bits)),where=hits>0)
        row={'guard':guard,'standalone_removed':int(triggered.sum()),'unique_removed':int(unique.sum()),'allocated_removed':float(weights.sum())}
        for cat in ['true_alert','false_alert','false_clear','true_clear']:
            mask=categories==cat
            row['standalone_'+cat]=int((mask&triggered).sum())
            row['unique_'+cat]=int((mask&unique).sum())
            row['allocated_'+cat]=float(weights[mask].sum())
        rows.append(row)
    table=pd.DataFrame(rows)
    removed=issued&(bits!=0)
    if not np.isclose(table.allocated_removed.sum(),removed.sum()):raise AssertionError('Attribution does not reconcile')
    for cat in ['true_alert','false_alert','false_clear','true_clear']:
        if not np.isclose(table['allocated_'+cat].sum(),(removed&(categories==cat)).sum()):raise AssertionError(cat)
    return table


def paired_guard_intervals(y, full, candidate, groups, config):
    data=pd.DataFrame({'y':y,'full':full,'candidate':candidate,'group':groups})
    if data.group.isna().any():raise ValueError('Missing bootstrap groups')
    a=[];b=[]
    for _,g in data.groupby('group',sort=True):
        a.append(decision_counts(g.y,g.full));b.append(decision_counts(g.y,g.candidate))
    a=pd.DataFrame(a);b=pd.DataFrame(b);n=len(a)
    if not n:return []
    weights=np.random.default_rng(config['bootstrap_seed']).multinomial(n,np.full(n,1/n),size=config['bootstrap_repetitions'])
    da=dict(zip(a.columns,(weights@a.to_numpy()).T));db=dict(zip(b.columns,(weights@b.to_numpy()).T))
    ma=decision_metrics(y,full);mb=decision_metrics(y,candidate);rows=[]
    for metric in ['retention','selective_error','alert_fraction_of_all_flares','false_clear_fraction_of_all_flares','deferred_fraction_of_all_flares','false_alert_fraction_of_all_nonflares']:
        num,den=RATE_DEFINITIONS[metric];valid=(da[den]>0)&(db[den]>0)
        support=min(int(a[den].gt(0).sum()),int(b[den].gt(0).sum()))
        bounds=np.quantile(db[num][valid]/db[den][valid]-da[num][valid]/da[den][valid],[.025,.975]) if support>=2 and valid.mean()>=.95 else [None,None]
        rows.append({'metric':metric,'difference':mb[metric]-ma[metric] if ma[metric] is not None and mb[metric] is not None else None,
            'low':float(bounds[0]) if bounds[0] is not None else None,'high':float(bounds[1]) if bounds[1] is not None else None,
            'groups':n,'groups_with_denominator':support,'valid_repetitions':int(valid.sum())})
    return rows


def load_diagnostic_sources(root,config):
    parent=root/config['policy_dir'];rolling=root/config['rolling_dir']
    checked_parent(parent,config['policy_summary_sha256']);checked_parent(rolling,config['rolling_summary_sha256'])
    frame=pd.read_csv(parent/'population_decisions.csv.gz',low_memory=False,float_precision='round_trip')
    sets=pd.read_csv(rolling/'population_sets.csv.gz',low_memory=False,float_precision='round_trip')
    if frame.forecast_case_id.duplicated().any() or sets.forecast_case_id.duplicated().any() or set(frame.forecast_case_id)!=set(sets.forecast_case_id):raise ValueError('Case identity mismatch')
    sets=sets.set_index('forecast_case_id').loc[frame.forecast_case_id].reset_index()
    for col in ['issue_utc','role','candidate_primary_label_72h']:
        if not frame[col].fillna('__missing__').equals(sets[col].fillna('__missing__')):raise ValueError('Source alignment changed: '+col)
    gates=json.loads((parent/'frozen_gates.json').read_text())
    if not np.array_equal(frame.label_known_primary_72h,frame.candidate_primary_label_72h.isin([0,1])):raise ValueError('Known-label mask changed')
    return frame,sets,gates


def build_case_diagnostic(frame, sets, gates, alpha, config):
    ac=str(alpha).replace('.','p');methods=gates['selected_rolling_methods'];limits=gates['policies']['guarded_gru']
    g=sets[f'gru_{methods["gru"]}_alpha{ac}'].to_numpy(dtype=float)
    l=sets[f'logistic_{methods["logistic"]}_alpha{ac}'].to_numpy(dtype=float)
    flags={'support':sets[f'gru_{methods["gru"]}_guard'].ne(0).to_numpy(),
      'distance':frame.feature_distance_squared.gt(limits['distance']).to_numpy(),
      'spread':frame.seed_spread.gt(limits['spread']).to_numpy(),
      'conflict':np.isin(g,[1,2])&np.isin(l,[1,2])&(g!=l)}
    bits=sum(flags[name].astype(np.uint8)*bit for name,bit in config['guard_bits'].items())
    base=frame[f'gru_singleton_alpha{ac}_decision'].to_numpy(dtype=float)
    full=subset_decisions(base,bits,15)
    np.testing.assert_allclose(full,frame[f'guarded_gru_alpha{ac}_decision'],rtol=0,atol=0,equal_nan=True)
    for name,col in [('support','support_ok'),('distance','feature_flag'),('spread','spread_flag'),('conflict','conflicting_singletons')]:
        expected=~frame[f'primary_alpha{ac}_{col}'].to_numpy() if name=='support' else frame[f'primary_alpha{ac}_{col}'].to_numpy()
        if not np.array_equal(flags[name],expected):raise ValueError('Frozen flag mismatch: '+name)
    fallback=frame[f'guarded_fallback_alpha{ac}_state'].eq('degraded').to_numpy()
    route=np.select([fallback&(g==3),fallback&np.isin(g,[1,2])],['ambiguous_gru','spread_blocked_singleton'],default='not_fallback')
    if (fallback&(route=='not_fallback')).any():raise ValueError('Unexpected fallback route')
    return pd.DataFrame({'forecast_case_id':frame.forecast_case_id,'alpha':alpha,'role':frame.role,'issue_utc':frame.issue_utc,
        'region_component_id':frame.region_component_id,'label':frame.candidate_primary_label_72h,'label_known':frame.label_known_primary_72h,
        'baseline_decision':base,'full_guard_decision':full,'guard_mask':bits,'fallback_route':route,
        'fallback_decision':np.where(fallback,frame[f'guarded_fallback_alpha{ac}_decision'],np.nan),
        'gru_set_code':g,'logistic_set_code':l})


def analyze_cases(cases,config):
    metrics=[];attribution=[];overlaps=[];intervals=[];fallback=[];yearly=[]
    issue=pd.to_datetime(cases.issue_utc,utc=True)
    for alpha in config['alphas']:
        for role in config['evaluation_roles']:
            block=cases[cases.alpha.eq(alpha)&cases.role.eq(role)&cases.label_known]
            y=block.label.to_numpy(dtype=int);base=block.baseline_decision.to_numpy();bits=block.guard_mask.to_numpy()
            full=block.full_guard_decision.to_numpy();cats=outcome_categories(y,base);tags={'alpha':alpha,'role':role}
            if len(block)==0:raise ValueError('No diagnostic cases')
            for mask in range(16):
                active=[name for name,bit in config['guard_bits'].items() if mask&bit]
                d=subset_decisions(base,bits,mask)
                metrics.append({**tags,'active_mask':mask,'active_guards':'+'.join(active) or 'none',**decision_metrics(y,d)})
            for row in gate_attribution(y,base,bits,config).to_dict('records'):attribution.append({**tags,**row})
            for pattern in range(16):
                m=(bits==pattern)&np.isfinite(base)
                overlaps.append({**tags,'guard_mask':pattern,'cases':int(m.sum()),**{cat:int((m&(cats==cat)).sum()) for cat in ['true_alert','false_alert','false_clear','true_clear']}})
            for name,bit in config['guard_bits'].items():
                candidate=subset_decisions(base,bits,15^bit)
                for grouping in config['bootstrap_groups']:
                    groups=block.region_component_id.to_numpy() if grouping=='region_component' else (issue.loc[block.index].astype('int64')//(7*86400*10**9)).to_numpy()
                    for row in paired_guard_intervals(y,full,candidate,groups,config):intervals.append({**tags,'removed_guard':name,'grouping':grouping,**row})
            for route in ['all_fallback','ambiguous_gru','spread_blocked_singleton']:
                m=block.fallback_route.ne('not_fallback') if route=='all_fallback' else block.fallback_route.eq(route)
                sub=block.loc[m];fy=sub.label.to_numpy(dtype=int);fd=sub.fallback_decision.to_numpy()
                score=decision_metrics(fy,fd)
                alerts=int((fd==1).sum());clears=int((fd==0).sum())
                fallback.append({**tags,'route':route,**score,'alert_decisions':alerts,'clear_decisions':clears,
                   'alert_precision':score['true_alerts']/alerts if alerts else None,
                   'clear_error_rate':score['false_clears']/clears if clears else None})
            for year in sorted(issue.loc[block.index].dt.year.unique()):
                m=issue.loc[block.index].dt.year.eq(year).to_numpy()
                for row in gate_attribution(y[m],base[m],bits[m],config).to_dict('records'):yearly.append({**tags,'year':int(year),**row})
    return {name:pd.DataFrame(rows) for name,rows in [('subset_metrics',metrics),('gate_attribution',attribution),('guard_overlap',overlaps),('paired_intervals',intervals),('fallback_routes',fallback),('yearly_attribution',yearly)]}


def run_diagnostic(root,config,output,source_code):
    if output.exists():raise ValueError('Frozen runs cannot be overwritten')
    frame,sets,gates=load_diagnostic_sources(root,config)
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=output.name+'.incomplete-',dir=output.parent))
    try:
        save_json(staging/'protocol_before_diagnostics.json',{'config':config,'recorded_utc':datetime.now(timezone.utc).isoformat()})
        cases=pd.concat([build_case_diagnostic(frame,sets,gates,a,config) for a in config['alphas']],ignore_index=True)
        tables=analyze_cases(cases,config)
        for name,table in tables.items():table.to_csv(staging/f'{name}.csv',index=False)
        cases.to_csv(staging/'case_diagnostics.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        (staging/'executed_diagnostic_code.py').write_text(source_code)
        save_json(staging/'run_contract.json',{'config':config,'versions':{'numpy':np.__version__,'pandas':pd.__version__},
          'source_code_sha256':hashlib.sha256(source_code.encode()).hexdigest(),'frozen_thresholds':gates['policies']['guarded_gru'],
          'metric_definitions':RATE_DEFINITIONS,'operational_policy_changed':False})
        summary={'status':'completed_exploratory_guard_diagnostic','completed_utc':datetime.now(timezone.utc).isoformat(),
          'population_cases':len(frame),'case_alpha_rows':len(cases),'subset_comparisons':len(tables['subset_metrics']),
          'limitations':[
          'This is a post-hoc diagnostic of already inspected data and provisional labels. No new thresholds or operational policy are selected.',
          'Guard contributions describe deterministic routing of frozen decisions, not causes of solar activity, causes of model error or independently validated improvements.',
          'Standalone guard counts overlap. Unique counts measure dropping one guard while keeping the other three. Equal allocation shares each removed case among its triggered guards and reconciles totals; it is not a unique scientific attribution.',
          'A false alert prevented and a true alert suppressed have different operational values. Raw counts and selective error do not supply an agreed cost or utility.',
          'Conditional paired intervals use region or time-block resampling of realized predictions; model fitting, rolling calibration, threshold selection and label uncertainty are not repeated. Intervals are per comparison, without multiplicity correction.',
          'Windows overlap and are not independent flare events. Matched known-outcome metrics exclude unknown labels and missing inputs, which remain in the full-population diagnostic table.',
          'The diagnostic does not change the q99 feature-distance or q95 seed-spread thresholds. Removing the support guard is an accounting ablation, not approval to issue forecasts with weak calibration support.',
          'Fallback cases are selected difficult cases. Their error rate is not the logistic model error rate across the entire population. Neither backbone nor fallback is operationally certified.'
          ],'output_sha256':{p.name:sha256(p) for p in staging.iterdir()}}
        save_json(staging/'summary.json',summary);staging.rename(output);return summary
    except Exception as exc:
        save_json(staging/'failure.json',{'incomplete_run':True,'error':str(exc)});raise
