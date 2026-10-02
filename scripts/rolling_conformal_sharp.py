"""Past-only daily rolling conformal replay of frozen SHARP probabilities."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from scripts.conformal_sharp import (sha256, save_json, validate_probabilities,
    finite_sample_threshold, predict_sets, set_metrics, coverage_bootstrap, load_sources)


def code_column(model, method, alpha):
    return f'{model}_{method}_alpha{str(alpha).replace(".", "p")}'


def decode_sets(codes):
    codes = np.asarray(codes)
    if not np.isin(codes, [0, 1, 2, 3]).all():
        raise ValueError('Missing sets must not be treated as empty sets')
    codes = codes.astype(int)
    return np.column_stack([(codes & 1) != 0, (codes & 2) != 0])


def prepare_history(frame, config):
    if config['reporting_delay_hours'] < 0:
        raise ValueError('Negative reporting delay')
    issue = pd.to_datetime(frame.issue_utc, utc=True, format='ISO8601')
    ready = pd.to_datetime(frame.outcome_end_utc_72h, utc=True, format='ISO8601') + pd.Timedelta(hours=config['reporting_delay_hours'])
    if not ((ready-issue) >= pd.Timedelta(hours=config['horizon'])).all():
        raise ValueError('Outcome maturity precedes forecast horizon')
    eligible = frame.label_known & frame.label.isin([0, 1]) & (issue >= pd.Timestamp(config['history_start'], tz='UTC'))
    history = frame.loc[eligible, ['forecast_case_id', 'issue_utc', 'region_component_id', 'label', *[f'{m}_selected' for m in config['models']]]].copy()
    history['issue_ns'] = issue[eligible].astype('int64').to_numpy()
    history['ready_ns'] = ready[eligible].astype('int64').to_numpy()
    if history.forecast_case_id.duplicated().any() or history.region_component_id.isna().any():
        raise ValueError('Invalid history identity')
    for model in config['models']:
        validate_probabilities(history[f'{model}_selected'], history.label)
    return history.sort_values(['issue_ns', 'forecast_case_id']).reset_index(drop=True)


def eligible_history(history, update, days):
    if days <= 0:
        raise ValueError('Lookback must be positive')
    update = pd.Timestamp(update)
    if update.tzinfo is None or update != update.floor('D'):
        raise ValueError('Update must be a timezone-aware UTC day boundary')
    update = update.tz_convert('UTC')
    if update != update.floor('D'):
        raise ValueError('Update must occur at midnight UTC')
    t = update.value
    issue = history.issue_ns.to_numpy()
    left, right = np.searchsorted(issue, [t-int(days)*86400*10**9, t], side='left')
    block = history.iloc[left:right]
    return block.loc[block.ready_ns <= t]


def daily_thresholds(block, config, model, alpha):
    p = validate_probabilities(block[f'{model}_selected'], block.label)
    y = block.label.to_numpy(dtype=int)
    thresholds, support = {}, {}
    for label in [0, 1]:
        subset = y == label
        scores = p[subset] if label == 0 else 1-p[subset]
        regions = int(block.loc[subset, 'region_component_id'].nunique())
        q = finite_sample_threshold(scores, alpha)
        guarded = len(scores) < config['minimum_windows_per_class'] or regions < config['minimum_region_components_per_class']
        if guarded:
            q.update(threshold=None, include_all=True)
        thresholds[str(label)] = q
        support[str(label)] = {'windows': len(scores), 'region_components': regions, 'guarded': guarded,
            'latest_ready_utc': pd.Timestamp(int(block.loc[subset, 'ready_ns'].max()), tz='UTC').isoformat() if subset.any() else None}
    return {'method': 'rolling_class_conditional', 'alpha': alpha, 'thresholds': thresholds}, support


def generate_replay(frame, population, fixed, config):
    history = prepare_history(frame, config)
    result = population[['forecast_case_id', 'issue_utc', 'inputs_available', 'candidate_primary_label_72h',
                         'label_known_primary_72h', *[f'{m}_selected' for m in config['models']]]].copy()
    result = result.merge(frame[['forecast_case_id', 'role', 'region_component_id']], on='forecast_case_id', how='left', validate='one_to_one')
    issue = pd.to_datetime(result.issue_utc, utc=True, format='ISO8601')
    start = pd.Timestamp(config['replay_start'], tz='UTC')
    active = issue >= start
    updates = issue.dt.floor('D')
    result['threshold_update_utc'] = updates.where(active).astype(str).replace('NaT', '')
    journal = []
    for model in config['models']:
        available = result[f'{model}_selected'].notna()
        applies = active & available
        result[f'{model}_set_status'] = np.select([~available, ~active], ['no_supported_sharp_input', 'before_replay_start'], default='set_issued_exploratory')
        for method in config['candidate_methods']:
            for alpha in config['alphas']:
                result[code_column(model, method, alpha)] = pd.Series(pd.NA, index=result.index, dtype='Int8')
        for days in config['lookback_days']:
            result[f'{model}_rolling_{days}d_guard'] = pd.Series(pd.NA, index=result.index, dtype='Int8')
        for alpha in config['alphas']:
            sets = predict_sets(result.loc[applies, f'{model}_selected'].to_numpy(), fixed['models'][model]['class_conditional'][str(alpha)])
            result.loc[applies, code_column(model, 'fixed_2015', alpha)] = (sets[:, 0].astype(int)+2*sets[:, 1].astype(int)).astype('int8')
    # One shared information cutoff per UTC day, independent of later outcomes.
    target_groups = result.loc[active].groupby(updates[active], sort=True).groups
    for update, positions in target_groups.items():
        for days in config['lookback_days']:
            block = eligible_history(history, update, days)
            support_hash = hashlib.sha256('\n'.join(sorted(block.forecast_case_id)).encode()).hexdigest()
            for model in config['models']:
                selected = result.loc[positions, f'{model}_selected'].notna()
                indices = np.asarray(positions)[selected.to_numpy()]
                probabilities = result.loc[indices, f'{model}_selected'].to_numpy()
                for alpha in config['alphas']:
                    parameters, support = daily_thresholds(block, config, model, alpha)
                    sets = predict_sets(probabilities, parameters)
                    method = f'rolling_{days}d'
                    result.loc[indices, code_column(model, method, alpha)] = (sets[:, 0].astype(int)+2*sets[:, 1].astype(int)).astype('int8')
                    guard_code = int(support['0']['guarded']) + 2*int(support['1']['guarded'])
                    result.loc[indices, f'{model}_{method}_guard'] = guard_code
                    for label in ['0', '1']:
                        q = parameters['thresholds'][label]
                        journal.append({'update_utc': update.isoformat(), 'lookback_days': days, 'model': model, 'alpha': alpha,
                            'label': int(label), 'rank': q['rank'], 'threshold': q['threshold'], 'include_all': q['include_all'],
                            **support[label], 'history_case_ids_sha256': support_hash, 'issued_cases': len(indices)})
    return result, pd.DataFrame(journal)


def select_on_policy(frame, config):
    selected = frame.role.eq('policy_validation') & frame.label_known_primary_72h
    issue = pd.to_datetime(frame.issue_utc, utc=True, format='ISO8601')
    stop = pd.Timestamp(config['policy_end'], tz='UTC')
    start = pd.Timestamp(config['policy_start'], tz='UTC')
    # The original source split already purged boundary windows; verify again here.
    mature = issue + pd.Timedelta(hours=config['horizon']+config['reporting_delay_hours'])
    if (selected & ~((issue >= start) & (issue < stop) & (mature <= stop))).any():
        raise ValueError('Policy selection includes unmatured or out-of-period outcomes')
    y = frame.loc[selected, 'candidate_primary_label_72h'].to_numpy(dtype=int)
    if set(y) != {0, 1}:
        raise ValueError('Need both classes for policy-period selection')
    scores, choices = [], {}
    alpha = config['primary_alpha']
    for model in config['models']:
        candidates = []
        for order, method in enumerate(config['candidate_methods']):
            sets = decode_sets(frame.loc[selected, code_column(model, method, alpha)].to_numpy())
            score = set_metrics(y, sets)
            deficit = max(0., 1-alpha-score['nonflare_coverage'], 1-alpha-score['flare_coverage'])
            scores.append({'model': model, 'method': method, 'alpha': alpha, 'max_class_coverage_deficit': deficit, **score})
            candidates.append((deficit, score['mean_set_size'], order, method))
        choices[model] = min(candidates)[-1]
    return {'selected_methods': choices, 'policy_scores': scores, 'policy_cases': len(y), 'policy_positive': int(y.sum()),
            'selection_frozen_as_of': config['policy_end'], 'rule': config['selection']}


def paired_set_bootstrap(y, candidate, fixed, groups, config):
    y = np.asarray(y, dtype=int)
    set_metrics(y, candidate); set_metrics(y, fixed)
    if pd.isna(groups).any():
        raise ValueError('Missing bootstrap groups')
    c = candidate[np.arange(len(y)), y].astype(float)
    f = fixed[np.arange(len(y)), y].astype(float)
    data = pd.DataFrame({'group': groups, 'n': 1, 'flare_n': (y == 1).astype(int), 'nonflare_n': (y == 0).astype(int),
        'coverage': c-f, 'flare_coverage': (c-f)*(y == 1), 'nonflare_coverage': (c-f)*(y == 0),
        'mean_set_size': candidate.sum(axis=1)-fixed.sum(axis=1),
        'both_fraction': (candidate.sum(axis=1)==2).astype(int)-(fixed.sum(axis=1)==2).astype(int)})
    grouped = data.groupby('group', sort=True).sum()
    n = len(grouped)
    weights = np.random.default_rng(config['bootstrap_seed']).multinomial(n, np.full(n, 1/n), size=config['bootstrap_repetitions'])
    totals = dict(zip(grouped.columns, (weights @ grouped.to_numpy()).T))
    rows=[]
    for metric, denominator in [('coverage', 'n'), ('nonflare_coverage', 'nonflare_n'), ('flare_coverage', 'flare_n'), ('mean_set_size', 'n'), ('both_fraction', 'n')]:
        denom = grouped[denominator].sum()
        valid = totals[denominator] > 0
        support = int(grouped[denominator].gt(0).sum())
        draws = totals[metric][valid]/totals[denominator][valid]
        bounds = np.quantile(draws, [.025, .975]) if support >= 2 and valid.mean() >= .95 else [None, None]
        rows.append({'metric': metric, 'difference': float(grouped[metric].sum()/denom) if denom else None,
                     'low': float(bounds[0]) if bounds[0] is not None else None, 'high': float(bounds[1]) if bounds[1] is not None else None,
                     'groups_with_denominator': support, 'valid_repetitions': int(valid.sum())})
    return rows


def evaluate_replay(frame, config, selection):
    metrics, yearly, monthly, intervals, differences = [], [], [], [], []
    issue = pd.to_datetime(frame.issue_utc, utc=True, format='ISO8601')
    for role in config['evaluation_roles']:
        available = frame.gru_selected.notna() & frame.logistic_selected.notna()
        eligible = frame.role.eq(role) & frame.label_known_primary_72h & available
        ev = frame.loc[eligible]
        y = ev.candidate_primary_label_72h.to_numpy(dtype=int)
        dates = issue[eligible]
        for model in config['models']:
            for alpha in config['alphas']:
                fixed = decode_sets(ev[code_column(model, 'fixed_2015', alpha)].to_numpy())
                for method in config['candidate_methods']:
                    tags = {'role': role, 'model': model, 'method': method, 'alpha': alpha, 'selected_on_policy': method == selection['selected_methods'][model]}
                    sets = decode_sets(ev[code_column(model, method, alpha)].to_numpy())
                    guards = np.zeros(len(ev), dtype=int) if method == 'fixed_2015' else ev[f'{model}_{method}_guard'].to_numpy(dtype=int)
                    score = set_metrics(y, sets)
                    metrics.append({**tags, **score, 'support_guard_fraction': float((guards > 0).mean()),
                                    'flare_class_guard_fraction': float(((guards & 2) != 0).mean())})
                    for year in sorted(dates.dt.year.unique()):
                        mask = dates.dt.year.eq(year).to_numpy()
                        yearly.append({**tags, 'year': int(year), **set_metrics(y[mask], sets[mask]),
                                       'support_guard_fraction': float((guards[mask] > 0).mean())})
                    for month in sorted(dates.dt.strftime('%Y-%m').unique()):
                        mask = dates.dt.strftime('%Y-%m').eq(month).to_numpy()
                        monthly.append({**tags, 'month': month, **set_metrics(y[mask], sets[mask]),
                                        'support_guard_fraction': float((guards[mask] > 0).mean())})
                    # Full uncertainty receipts for fixed and the earlier-selected method only.
                    if method not in ['fixed_2015', selection['selected_methods'][model]]:
                        continue
                    for grouping in config['bootstrap_groups']:
                        groups = ev.region_component_id.to_numpy() if grouping == 'region_component' else (dates.astype('int64')//(7*86400*10**9)).to_numpy()
                        for row in coverage_bootstrap(y, sets, groups, config['bootstrap_repetitions'], config['bootstrap_seed']):
                            intervals.append({**tags, 'grouping': grouping, 'year': 'all', **row})
                        if method != 'fixed_2015':
                            for row in paired_set_bootstrap(y, sets, fixed, groups, config):
                                differences.append({**tags, 'grouping': grouping, **row})
                        if alpha == config['primary_alpha'] and grouping == 'region_component':
                            for year in sorted(dates.dt.year.unique()):
                                mask = dates.dt.year.eq(year).to_numpy()
                                for row in coverage_bootstrap(y[mask], sets[mask], groups[mask], config['bootstrap_repetitions'], config['bootstrap_seed']):
                                    intervals.append({**tags, 'grouping': grouping, 'year': str(year), **row})
    return {name: pd.DataFrame(rows) for name,rows in [('metrics',metrics),('yearly_metrics',yearly),('monthly_metrics',monthly),('coverage_intervals',intervals),('paired_differences',differences)]}


def run_rolling(parent_dir, fixed_dir, dataset, config, output, source_code):
    if output.exists():
        raise ValueError('Frozen runs cannot be overwritten')
    frame, population, parent = load_sources(parent_dir, dataset, config)
    if sha256(fixed_dir/'summary.json') != config['fixed_summary_sha256']:
        raise ValueError('Fixed conformal summary changed')
    fixed_summary = json.loads((fixed_dir/'summary.json').read_text())
    for name, expected in fixed_summary['output_sha256'].items():
        if sha256(fixed_dir/name) != expected:
            raise ValueError(f'Fixed conformal output changed: {name}')
    fixed = json.loads((fixed_dir/'thresholds.json').read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=output.name+'.incomplete-',dir=output.parent))
    try:
        save_json(staging/'protocol_before_replay.json', {'config': config, 'recorded_utc': datetime.now(timezone.utc).isoformat()})
        replay,journal=generate_replay(frame,population,fixed,config)
        selection=select_on_policy(replay,config)
        save_json(staging/'selection.json',selection)
        # Method selection reads only earlier policy labels, then is frozen before scoring later periods.
        tables=evaluate_replay(replay,config,selection)
        for name,table in tables.items():table.to_csv(staging/f'{name}.csv',index=False)
        replay.to_csv(staging/'population_sets.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        journal.to_csv(staging/'threshold_journal.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        support=journal.groupby(['model','lookback_days','alpha','label']).agg(updates=('update_utc','size'),guarded_updates=('guarded','sum'),minimum_windows=('windows','min'),median_windows=('windows','median'),minimum_components=('region_components','min')).reset_index()
        support.to_csv(staging/'support_summary.csv',index=False)
        pd.DataFrame(selection['policy_scores']).to_csv(staging/'policy_scores.csv',index=False)
        issue=pd.to_datetime(replay.issue_utc,utc=True,format='ISO8601')
        availability=[]
        for year,block in replay.groupby(issue.dt.year):
            availability.append({'year':int(year),'candidate_cases':len(block),'missing_inputs':int(block.gru_selected.isna().sum()),
              'sets_issued':int(block.gru_set_status.eq('set_issued_exploratory').sum()),
              'issued_unknown_outcome':int((block.gru_set_status.eq('set_issued_exploratory') & ~block.label_known_primary_72h).sum())})
        pd.DataFrame(availability).to_csv(staging/'availability.csv',index=False)
        (staging/'executed_rolling_code.py').write_text(source_code)
        save_json(staging/'run_contract.json',{'config':config,'source_code_sha256':hashlib.sha256(source_code.encode()).hexdigest(),
            'versions':{'numpy':np.__version__,'pandas':pd.__version__},'probability_methods':parent['selected_methods'],
            'set_encoding':{'0':'empty','1':'nonflare_only','2':'flare_only','3':'both','null':'no_set_issued'},
            'guard_encoding':{'0':'supported','1':'class0_insufficient','2':'class1_insufficient','3':'both_insufficient','null':'no_set_issued'}})
        summary={'status':'completed_exploratory_rolling_replay','completed_utc':datetime.now(timezone.utc).isoformat(),
            'selected_methods':selection['selected_methods'],'policy_cases':selection['policy_cases'],'policy_positive':selection['policy_positive'],
            'population_rows':len(replay),'missing_input_rows':int(replay.gru_selected.isna().sum()),
            'daily_updates':int(journal.update_utc.nunique()),'metrics':tables['metrics'].to_dict(orient='records'),
            'limitations':[
                'Candidate labels and outcome-coverage/availability limitations remain provisional; 24-hour reporting latency is assumed, not reconstructed.',
                'The method was designed after viewing earlier retrospective results. This is an exploratory extension, not untouched confirmatory validation.',
                'The predictor and probability mapping stay frozen, but rolling calibration uses matured earlier Cycle-25/2026 labels. It is not label-free cross-cycle transfer.',
                'Rolling quantiles with fixed alpha are not adaptive conformal inference with an error-feedback controller, and no distribution-free coverage guarantee is claimed under drift/dependence.',
                'Daily updates use the available native issue-time cohort, not a complete operational forecast schedule. No 2020 scored cases are available in this snapshot.',
                'Low-support guards deliberately include unsupported classes, potentially yielding both labels. This can improve coverage by sacrificing useful specificity.',
                'Selection minimizes point-estimate class coverage deficit then mean set size on one earlier policy block; this does not certify class coverage or safety.',
                'Forecast windows overlap. Bootstrap intervals condition on the realized sequential forecasts and do not rerun adaptation or include calibration/model/label uncertainty.',
                'No operational decision-state policy, prospective deployment, AIA fusion or continuous-target quantile regression is validated here.'
            ],'output_sha256':{p.name:sha256(p) for p in staging.iterdir()}}
        save_json(staging/'summary.json',summary)
        staging.rename(output)
        return summary
    except Exception as exc:
        save_json(staging/'failure.json',{'incomplete_run':True,'error':str(exc)})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-dir',type=Path,required=True)
    parser.add_argument('--fixed-dir',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    result=run_rolling(args.parent_dir,args.fixed_dir,args.dataset,json.loads(args.config.read_text()),args.output_dir,Path(__file__).read_text())
    print(json.dumps({'status':result['status'],'selected_methods':result['selected_methods']},indent=2))
