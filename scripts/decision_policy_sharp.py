"""Research-only selective forecast states using frozen SHARP models and sets."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import sklearn
from sklearn.covariance import LedoitWolf


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checked_parent(path, expected):
    if sha256(path / 'summary.json') != expected:
        raise ValueError('Parent summary changed')
    summary = json.loads((path / 'summary.json').read_text())
    for name, digest in summary['output_sha256'].items():
        if sha256(path / name) != digest:
            raise ValueError(f'Parent output changed: {name}')
    return summary


def gate_threshold(values, quantile):
    if quantile is None:
        return None
    x = np.asarray(values, dtype=float)
    if not 0 < quantile <= 1 or x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid reference values or quantile')
    return float(np.quantile(x, quantile, method='higher'))


def load_policy_inputs(root, config):
    training = root / config['training_dir']
    rolling = root / config['rolling_dir']
    dataset = root / config['dataset']
    train_summary = checked_parent(training, config['training_summary_sha256'])
    roll_summary = checked_parent(rolling, config['rolling_summary_sha256'])
    if train_summary['horizon'] != 72 or train_summary['scope'] != 'primary':
        raise ValueError('Wrong target')
    if sha256(dataset / 'manifest.json') != config['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest changed')
    manifest = json.loads((dataset / 'manifest.json').read_text())
    for name in ['sharp.npy', 'input_index.csv.gz', 'master_cases.csv.gz']:
        if sha256(dataset / name) != manifest['outputs'][name]['sha256']:
            raise ValueError(f'Dataset input changed: {name}')
    frame = pd.read_csv(rolling / 'population_sets.csv.gz')
    predictions = pd.read_csv(training / 'predictions.csv.gz')
    index = pd.read_csv(dataset / 'input_index.csv.gz')
    master = pd.read_csv(dataset / 'master_cases.csv.gz', low_memory=False)
    raw = np.load(dataset / 'sharp.npy', allow_pickle=False)
    if frame.forecast_case_id.duplicated().any() or set(frame.forecast_case_id) != set(master.forecast_case_id):
        raise ValueError('Population mismatch')
    indexed = index.set_index('forecast_case_id').loc[predictions.forecast_case_id]
    if not np.array_equal(indexed.tensor_row, predictions.tensor_row) or not np.array_equal(index.tensor_row, np.arange(len(raw))):
        raise ValueError('Tensor identity mismatch')
    if not np.isfinite(raw).all():
        raise ValueError('Pinned assembled SHARP contains nonfinite data')
    seed_cols = ['probability_seed_17', 'probability_seed_29', 'probability_seed_43']
    seeds = predictions[seed_cols].to_numpy()
    if not np.isfinite(seeds).all() or ((seeds < 0) | (seeds > 1)).any():
        raise ValueError('Invalid seed probabilities')
    predictions['seed_spread'] = seeds.std(axis=1, ddof=0)
    for col in ['issue_utc', 'region_component_id', 'role']:
        aligned = frame.set_index('forecast_case_id').loc[predictions.forecast_case_id, col]
        if not np.array_equal(aligned.to_numpy(), predictions[col].to_numpy()):
            raise ValueError(f'Frozen identity changed: {col}')
    # The pinned probability-selection experiment retained raw values for both models.
    for selected, source in [('gru_selected', 'probability_gru_mean'), ('logistic_selected', 'probability_logistic')]:
        aligned = frame.set_index('forecast_case_id').loc[predictions.forecast_case_id, selected]
        if not np.allclose(aligned, predictions[source], rtol=0, atol=1e-14):
            raise ValueError('Selected probabilities no longer match raw parent')
    with np.load(training / 'transform.npz', allow_pickle=False) as t:
        normalized = ((np.sign(raw) * np.log1p(np.abs(raw)) - t['mean']) / t['std']).astype(np.float32)
    x = normalized[predictions.tensor_row.to_numpy()].reshape(len(predictions), -1).astype(float)
    fit = predictions.role.eq('train').to_numpy()
    reference = predictions.role.eq(config['threshold_reference_role']).to_numpy()
    dates = pd.to_datetime(predictions.issue_utc, utc=True)
    if not fit.any() or not reference.any() or (dates[fit] >= pd.Timestamp('2014-01-01', tz='UTC')).any() or ((dates[reference] < pd.Timestamp('2014-01-01', tz='UTC')) | (dates[reference] >= pd.Timestamp('2014-07-01', tz='UTC'))).any():
        raise ValueError('Gate fitting/reference periods are invalid')
    estimator = LedoitWolf().fit(x[fit])
    predictions['feature_distance_squared'] = estimator.mahalanobis(x)
    timing = indexed.reset_index()
    times = [pd.to_datetime(timing[c], utc=True) for c in ['history_288_UTC','history_192_UTC','history_96_UTC']]
    issues = pd.to_datetime(predictions.issue_utc, utc=True)
    predictions['input_contract_ok'] = ((times[0] < times[1]) & (times[1] < times[2]) & (times[2] < issues)).to_numpy()
    frame = frame.merge(predictions[['forecast_case_id','seed_spread','feature_distance_squared','input_contract_ok']], on='forecast_case_id', how='left', validate='one_to_one')
    frame['input_contract_ok'] = frame.input_contract_ok.eq(True)
    frame = frame.merge(master[['forecast_case_id','outcome_end_utc_72h']], on='forecast_case_id', validate='one_to_one')
    truth = master.set_index('forecast_case_id').loc[frame.forecast_case_id]
    if not np.array_equal(frame.candidate_primary_label_72h.fillna(-1), truth.candidate_primary_label_72h.fillna(-1)) or not np.array_equal(frame.label_known_primary_72h, frame.candidate_primary_label_72h.isin([0,1])):
        raise ValueError('Outcome mismatch')
    gates = {'fit_cases': int(fit.sum()), 'reference_cases': int(reference.sum()),
        'fit_case_ids_sha256': hashlib.sha256('\n'.join(sorted(predictions.loc[fit,'forecast_case_id'])).encode()).hexdigest(),
        'reference_case_ids_sha256': hashlib.sha256('\n'.join(sorted(predictions.loc[reference,'forecast_case_id'])).encode()).hexdigest(),
        'covariance_shrinkage': float(estimator.shrinkage_), 'selected_rolling_methods': roll_summary['selected_methods'], 'policies': {}}
    for policy in config['policies']:
        gates['policies'][policy['name']] = {
            'distance': gate_threshold(predictions.loc[reference,'feature_distance_squared'], policy['distance_quantile']),
            'spread': gate_threshold(predictions.loc[reference,'seed_spread'], policy['spread_quantile'])}
    return frame, gates, {'location': estimator.location_, 'precision': estimator.precision_}


def decide_states(frame, gru_codes, logistic_codes, gru_guards, logistic_guards, policy, thresholds):
    """No labels are read. Null codes are unavailable, not empty sets."""
    g = np.asarray(gru_codes, dtype=float)
    l = np.asarray(logistic_codes, dtype=float)
    gg = np.asarray(gru_guards, dtype=float)
    lg = np.asarray(logistic_guards, dtype=float)
    for values in [g,l,gg,lg]:
        if len(values) != len(frame) or not np.isin(values[np.isfinite(values)], [0,1,2,3]).all():
            raise ValueError('Invalid set or support code')
    p = frame.gru_selected.to_numpy(dtype=float)
    lp = frame.logistic_selected.to_numpy(dtype=float)
    distance = frame.feature_distance_squared.to_numpy(dtype=float)
    spread = frame.seed_spread.to_numpy(dtype=float)
    valid_p = np.isfinite(p) & (p >= 0) & (p <= 1)
    valid_lp = np.isfinite(lp) & (lp >= 0) & (lp <= 1)
    inputs = frame.input_contract_ok.to_numpy(dtype=bool) & valid_p & np.isfinite(distance) & (distance >= 0) & np.isfinite(spread) & (spread >= 0)
    active = np.isfinite(g)
    gs = np.isin(g,[1,2]); ls = np.isin(l,[1,2])
    support_ok = gg == 0 if policy['support'] else np.ones(len(g),dtype=bool)
    fallback_support = lg == 0
    ood = (distance > thresholds['distance']) if thresholds['distance'] is not None else np.zeros(len(g),dtype=bool)
    unstable = (spread > thresholds['spread']) if thresholds['spread'] is not None else np.zeros(len(g),dtype=bool)
    conflict = gs & ls & (g != l) if policy['conflict'] else np.zeros(len(g),dtype=bool)
    ready = inputs & active & support_ok & ~ood & ~conflict
    normal = ready & gs & ~unstable
    # Empty GRU sets are unresolved contradictions. Both-label sets or high seed
    # spread may route to a supported logistic singleton, never through a shared OOD failure.
    degraded = ready & ~normal & (g != 0) & ls & fallback_support & valid_lp if policy['fallback'] else np.zeros(len(g),dtype=bool)
    state = np.select([normal,degraded], ['normal','degraded'], default='abstain')
    decision = np.where(normal,g-1,np.where(degraded,l-1,np.nan))
    probability = np.where(normal,p,np.where(degraded,lp,np.nan))
    reason = np.select([~inputs,~active,~support_ok,ood,conflict,g==0,normal,degraded,unstable,g==3],
        ['missing_or_invalid_input','no_issued_set','insufficient_calibration_support','feature_distance_flag','conflicting_singletons','empty_gru_set','checks_passed','logistic_fallback_candidate','seed_disagreement','ambiguous_gru_set'],default='no_supported_mode')
    return pd.DataFrame({'state':state,'decision':decision,'issued_probability':probability,
        'mode':np.select([normal,degraded],['gru','logistic'],default='none'),'reason':reason,
        'feature_flag':ood,'spread_flag':unstable,'conflicting_singletons':conflict,'support_ok':support_ok})


def decision_counts(y, decision):
    y=np.asarray(y); d=np.asarray(decision,dtype=float)
    if not np.isin(y,[0,1]).all() or not np.isin(d[np.isfinite(d)],[0,1]).all():
        raise ValueError('Metrics require known binary outcomes and valid decisions')
    issued=np.isfinite(d); flare=y==1; quiet=y==0
    return {'cases':len(y),'issued':int(issued.sum()),'errors':int((issued&(d!=y)).sum()),
      'flare_cases':int(flare.sum()),'nonflare_cases':int(quiet.sum()),
      'flare_issued':int((flare&issued).sum()),'nonflare_issued':int((quiet&issued).sum()),
      'true_alerts':int((flare&(d==1)).sum()),'false_clears':int((flare&(d==0)).sum()),
      'deferred_flares':int((flare&~issued).sum()),'false_alerts':int((quiet&(d==1)).sum()),
      'true_clears':int((quiet&(d==0)).sum()),'deferred_nonflares':int((quiet&~issued).sum())}


RATE_DEFINITIONS = {'retention':('issued','cases'),'selective_error':('errors','issued'),
 'flare_retention':('flare_issued','flare_cases'),'nonflare_retention':('nonflare_issued','nonflare_cases'),
 'alert_fraction_of_all_flares':('true_alerts','flare_cases'),
 'false_clear_fraction_of_all_flares':('false_clears','flare_cases'),
 'deferred_fraction_of_all_flares':('deferred_flares','flare_cases'),
 'false_alert_fraction_of_all_nonflares':('false_alerts','nonflare_cases'),
 'false_clear_rate_among_retained_flares':('false_clears','flare_issued'),
 'false_alert_rate_among_retained_nonflares':('false_alerts','nonflare_issued')}


def decision_metrics(y, decision):
    c=decision_counts(y,decision)
    return {**c, **{name:c[num]/c[den] if c[den] else None for name,(num,den) in RATE_DEFINITIONS.items()}}


def decision_intervals(y, decision, groups, config):
    groups=np.asarray(groups)
    if pd.isna(groups).any() or len(groups)!=len(y):raise ValueError('Invalid resampling groups')
    table=pd.DataFrame({'y':np.asarray(y),'d':np.asarray(decision,dtype=float),'group':groups})
    counts=pd.DataFrame([decision_counts(b.y,b.d) for _,b in table.groupby('group',sort=True)])
    n=len(counts)
    if not n:return []
    weights=np.random.default_rng(config['bootstrap_seed']).multinomial(n,np.full(n,1/n),size=config['bootstrap_repetitions'])
    draws=dict(zip(counts.columns,(weights@counts.to_numpy()).T))
    rows=[]
    for metric,(num,den) in RATE_DEFINITIONS.items():
        valid=draws[den]>0; support=int(counts[den].gt(0).sum())
        low,high=np.quantile(draws[num][valid]/draws[den][valid],[.025,.975]) if support>=2 and valid.mean()>=.95 else [None,None]
        rows.append({'metric':metric,'low':float(low) if low is not None else None,'high':float(high) if high is not None else None,'groups':n,'groups_with_denominator':support,'valid_repetitions':int(valid.sum())})
    return rows


def run_policy(root, config, output, source_code):
    if output.exists():raise ValueError('Frozen output cannot be overwritten')
    frame,gates,estimator=load_policy_inputs(root,config)
    output.parent.mkdir(exist_ok=True,parents=True)
    staging=Path(tempfile.mkdtemp(prefix=output.name+'.incomplete-',dir=output.parent))
    try:
        save_json(staging/'protocol_before_evaluation.json',{'config':config,'recorded_utc':datetime.now(timezone.utc).isoformat()})
        save_json(staging/'frozen_gates.json',gates)
        np.savez(staging/'feature_distance_model.npz',**estimator)
        metrics=[]; states=[]; intervals=[]; yearly=[]; reasons=[]; availability=[]; decision_columns={}
        issues=pd.to_datetime(frame.issue_utc,utc=True)
        ready=pd.to_datetime(frame.outcome_end_utc_72h,utc=True)+pd.Timedelta(hours=24)
        scored=frame[['forecast_case_id','issue_utc','role','region_component_id','inputs_available','candidate_primary_label_72h','label_known_primary_72h','gru_selected','logistic_selected','seed_spread','feature_distance_squared','input_contract_ok']].copy()
        for alpha in config['alphas']:
            ac=str(alpha).replace('.','p')
            chosen=gates['selected_rolling_methods']
            cols=[f'{m}_{chosen[m]}_alpha{ac}' for m in ['gru','logistic']]
            guards=[f'{m}_{chosen[m]}_guard' for m in ['gru','logistic']]
            if any(c not in frame for c in cols+guards):raise ValueError('Pinned method is not a supported rolling policy')
            for policy in config['policies']:
                name=policy['name'];tags={'policy':name,'alpha':alpha}
                out=decide_states(frame,frame[cols[0]],frame[cols[1]],frame[guards[0]],frame[guards[1]],policy,gates['policies'][name])
                for col in ['state','decision','issued_probability','mode','reason']:
                    decision_columns[f'{name}_alpha{ac}_{col}']=out[col]
                if name==config['primary_policy']:
                    for col in ['feature_flag','spread_flag','conflicting_singletons','support_ok']:decision_columns[f'primary_alpha{ac}_{col}']=out[col]
                for role in config['evaluation_roles']:
                    mask=frame.role.eq(role)&frame.label_known_primary_72h
                    b=frame.loc[mask];o=out.loc[mask];y=b.candidate_primary_label_72h.to_numpy(dtype=int)
                    metrics.append({**tags,'role':role,**decision_metrics(y,o.decision)})
                    for state in ['normal','degraded','abstain']:
                        s=o.state.eq(state).to_numpy()
                        states.append({**tags,'role':role,'state':state,**decision_metrics(y[s],o.loc[s,'decision'])})
                    if name in ['gru_singleton','guarded_gru','guarded_fallback']:
                        for grouping in ['region_component','calendar_7day_UTC']:
                            groups=b.region_component_id.to_numpy() if grouping=='region_component' else (issues[mask].astype('int64')//(7*86400*10**9)).to_numpy()
                            for row in decision_intervals(y,o.decision,groups,config):intervals.append({**tags,'role':role,'grouping':grouping,**row})
                    if name in ['gru_singleton','guarded_fallback']:
                        for year in sorted(issues[mask].dt.year.unique()):
                            s=issues[mask].dt.year.eq(year).to_numpy()
                            yearly.append({**tags,'role':role,'year':int(year),**decision_metrics(y[s],o.loc[s,'decision'])})
                    if name==config['primary_policy']:
                        for reason,block in o.groupby('reason'):
                            reasons.append({**tags,'role':role,'reason':reason,'cases':len(block),'flare_cases':int(frame.loc[block.index,'candidate_primary_label_72h'].eq(1).sum())})
                if name==config['primary_policy']:
                    for year in sorted(issues.dt.year.unique()):
                        if year<2021:continue
                        # Preserve the calendar population, including missing/unknown cases.
                        mask=issues.dt.year.eq(year);o=out.loc[mask];b=frame.loc[mask]
                        availability.append({**tags,'year':int(year),'population_cases':int(mask.sum()),'known_outcomes':int(b.label_known_primary_72h.sum()),
                           'missing_inputs':int((~b.input_contract_ok).sum()),
                           'normal':int(o.state.eq('normal').sum()),'degraded':int(o.state.eq('degraded').sum()),'abstain':int(o.state.eq('abstain').sum()),
                           'issued_unknown_outcomes':int((o.decision.notna()&~b.label_known_primary_72h).sum()),
                           'known_flares':int(b.candidate_primary_label_72h.eq(1).sum()),
                           'deferred_known_flares':int((b.candidate_primary_label_72h.eq(1)&o.decision.isna()).sum()),
                           'boundary_unmatured_by_year_end':int((ready[mask]>pd.Timestamp(f'{year+1}-01-01',tz='UTC')).sum())})
        for name,rows in [('metrics',metrics),('state_metrics',states),('intervals',intervals),('yearly_metrics',yearly),('reason_counts',reasons),('availability',availability)]:
            pd.DataFrame(rows).to_csv(staging/f'{name}.csv',index=False)
        scored=pd.concat([scored,pd.DataFrame(decision_columns)],axis=1)
        scored.to_csv(staging/'population_decisions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        (staging/'executed_policy_code.py').write_text(source_code)
        save_json(staging/'run_contract.json',{'config':config,'versions':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},'source_code_sha256':hashlib.sha256(source_code.encode()).hexdigest(),
          'state_semantics':{'normal':'Research candidate: GRU singleton passes the selected checks','degraded':'Research candidate: separately evaluated logistic fallback singleton','abstain':'No issued decision or probability; model scores retained separately for audit'},
          'rate_definitions':RATE_DEFINITIONS,'operationally_validated':False})
        summary={'status':'completed_exploratory_decision_policy','completed_utc':datetime.now(timezone.utc).isoformat(),'population_cases':len(frame),
          'primary_policy':config['primary_policy'],'metrics':metrics,'limitations':[
          'Normal and degraded are experimental state names, not certifications of safe or calibrated forecasts. The logistic fallback is a candidate evaluated here, not an approved operational mode.',
          'The q99 distance and q95 spread gates are declared heuristics; no acceptable operational error target or cost ratio has been supplied. Gates are not selected from later outcomes.',
          'Feature distance uses a unimodal covariance summary and is an outlier flag, not a validated OOD detector. Seed disagreement comes from only three same-architecture runs and is not a calibrated uncertainty interval.',
          'Training and January-June 2014 reference data were already used for model fitting and early stopping. The earlier 2015-2019 block already selected rolling windows; its policy metrics are development reuse, not independent validation.',
          'Later datasets have already been inspected. Candidate labels, unresolved associations, historical availability and continuous outcome coverage remain provisional. No source-level HMI quality flag or delivery-delay reconstruction is added here.',
          'Rolling sets use delayed earlier evaluation-period labels under the parent protocol. Threshold updates remain daily, forecasts use inherited native cases, and overlapping windows are not independent flare events.',
          'Classwise errors and deferrals must accompany aggregate selective error. Abstention is neither a correct forecast nor a no-flare decision; deferred flares are reported separately.',
          'Bootstrap intervals condition on realized decisions and do not repeat sequential adaptation, fit gates, or include label, model, calibration or selection uncertainty.',
          'Matched-known-outcome performance and full calendar-population availability have different denominators. Availability includes unknown and boundary cases that are excluded from inherited performance roles.',
          'Both modes require the same SHARP inputs: the logistic fallback does not repair missing SHARP or feature-distance flags. AIA fusion, correlated-outage replay and prospective evaluation remain future experiments.'],
          'output_sha256':{p.name:sha256(p) for p in staging.iterdir()}}
        save_json(staging/'summary.json',summary);staging.rename(output);return summary
    except Exception as exc:
        save_json(staging/'failure.json',{'incomplete_run':True,'error':str(exc)});raise
