"""Frozen 72h binary conformal sets: retrospective coverage, not an operational guarantee."""

import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_probabilities(p, y=None):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be a finite vector in [0,1]')
    if y is not None and (np.shape(y) != p.shape or not np.isin(y, [0, 1]).all()):
        raise ValueError('Unknown outcomes cannot enter conformal fitting or metrics')
    return p


def finite_sample_threshold(scores, alpha):
    scores = validate_probabilities(scores)
    a = Decimal(str(alpha))
    if not 0 < a < 1:
        raise ValueError('alpha must be strictly between zero and one')
    n = len(scores)
    rank = int((Decimal(n + 1) * (1-a)).to_integral_value(rounding=ROUND_CEILING))
    # A missing class or rank n+1 means +infinity, represented by JSON null.
    threshold = float(np.partition(scores, rank-1)[rank-1]) if rank <= n else None
    return {'n': n, 'rank': rank, 'threshold': threshold,
            'include_all': threshold is None}


def fit_thresholds(p, y, method, alpha):
    p = validate_probabilities(p, y)
    y = np.asarray(y, dtype=int)
    scores = np.where(y == 1, 1-p, p)
    if method == 'pooled':
        q = finite_sample_threshold(scores, alpha)
        thresholds = {'0': q, '1': q.copy()}
    elif method == 'class_conditional':
        thresholds = {str(k): finite_sample_threshold(scores[y == k], alpha) for k in [0, 1]}
    else:
        raise ValueError('Unknown conformal method')
    return {'method': method, 'alpha': alpha, 'thresholds': thresholds}


def predict_sets(p, parameters):
    p = validate_probabilities(p)
    scores = np.column_stack([p, 1-p])
    q = [np.inf if parameters['thresholds'][str(k)]['include_all'] else parameters['thresholds'][str(k)]['threshold'] for k in [0, 1]]
    # Inclusive ties are conservative; no randomized tie-breaking or forced singleton.
    return scores <= np.asarray(q)


def conformal_mask(frame, config):
    issue = pd.to_datetime(frame.issue_utc, utc=True, format='ISO8601')
    end = pd.to_datetime(frame.outcome_end_utc_72h, utc=True, format='ISO8601')
    start, stop = (pd.Timestamp(config[k], tz='UTC') for k in ['calibration_start', 'calibration_end'])
    if start >= stop or config['reporting_delay_hours'] < 0:
        raise ValueError('Invalid calibration information boundary')
    role = frame.role.eq('conformal_calibration')
    valid_time = (issue >= start) & (issue < stop) & (end + pd.Timedelta(hours=config['reporting_delay_hours']) <= stop)
    if (role & ~valid_time).any():
        raise ValueError('Conformal role violates issue/outcome/reporting boundary')
    known = frame.label.isin([0, 1]) & frame.label_known
    return (role & known).to_numpy()


def freeze_thresholds(frame, config):
    mask = conformal_mask(frame, config)
    if not mask.any():
        raise ValueError('No eligible conformal calibration cases')
    models = {}
    for model in config['models']:
        if frame.loc[mask, f'{model}_selected'].isna().any():
            raise ValueError('Calibration probabilities missing')
        models[model] = {method: {str(alpha): fit_thresholds(frame.loc[mask, f'{model}_selected'].to_numpy(),
                            frame.loc[mask, 'label'].to_numpy(), method, alpha)
                            for alpha in config['alphas']} for method in config['methods']}
    selected = frame.loc[mask]
    return {'models': models, 'support': {'cases': len(selected), 'nonflare': int(selected.label.eq(0).sum()),
            'flare': int(selected.label.eq(1).sum()), 'region_components': int(selected.region_component_id.nunique()),
            'first_issue_utc': str(selected.issue_utc.min()), 'last_issue_utc': str(selected.issue_utc.max())},
            'fit_case_ids_sha256': hashlib.sha256('\n'.join(sorted(selected.forecast_case_id)).encode()).hexdigest()}


def set_metrics(y, sets):
    y = np.asarray(y)
    sets = np.asarray(sets)
    if not len(y) or sets.shape != (len(y), 2) or sets.dtype != bool or not np.isin(y, [0, 1]).all():
        raise ValueError('Need nonempty known binary outcomes and boolean prediction sets')
    y = y.astype(int)
    size = sets.sum(axis=1)
    covered = sets[np.arange(len(y)), y]
    singleton = size == 1
    counts = {'empty': size == 0, 'nonflare_only': (size == 1) & sets[:, 0],
              'flare_only': (size == 1) & sets[:, 1], 'both': size == 2}
    result = {'cases': len(y), 'nonflare_cases': int((y == 0).sum()), 'flare_cases': int((y == 1).sum()),
              'covered': int(covered.sum()), 'coverage': float(covered.mean()), 'mean_set_size': float(size.mean()),
              'singleton_cases': int(singleton.sum()), 'singleton_fraction': float(singleton.mean()),
              'singleton_error': float((~covered[singleton]).mean()) if singleton.any() else None}
    for label, name in [(0, 'nonflare'), (1, 'flare')]:
        selected = y == label
        result[f'{name}_covered'] = int(covered[selected].sum())
        result[f'{name}_coverage'] = float(covered[selected].mean()) if selected.any() else None
    for name, selected in counts.items():
        result[f'{name}_count'] = int(selected.sum())
        result[f'{name}_fraction'] = float(selected.mean())
    result['flare_singleton_nonflare_count'] = int(((y == 1) & counts['nonflare_only']).sum())
    result['flare_singleton_nonflare_rate'] = result['flare_singleton_nonflare_count'] / result['flare_cases'] if result['flare_cases'] else None
    return result


def coverage_bootstrap(y, sets, groups, repetitions=1000, seed=31415):
    y = np.asarray(y, dtype=int)
    set_metrics(y, sets)
    if pd.isna(groups).any():
        raise ValueError('Missing resampling group')
    covered = sets[np.arange(len(y)), y]
    raw = {'group': np.asarray(groups)}
    for name, selected in [('all', np.ones(len(y), dtype=bool)), ('nonflare', y == 0), ('flare', y == 1)]:
        raw[name + '_n'] = selected.astype(int)
        raw[name + '_covered'] = (covered & selected).astype(int)
    grouped = pd.DataFrame(raw).groupby('group', sort=True).sum()
    n = len(grouped)
    weights = np.random.default_rng(seed).multinomial(n, np.full(n, 1/n), size=repetitions)
    totals = weights @ grouped.to_numpy()
    sampled = dict(zip(grouped.columns, totals.T))
    rows = []
    for name in ['all', 'nonflare', 'flare']:
        denom = int(grouped[name + '_n'].sum())
        supported_groups = int(grouped[name + '_n'].gt(0).sum())
        valid = sampled[name + '_n'] > 0
        draws = sampled[name + '_covered'][valid] / sampled[name + '_n'][valid]
        estimable = supported_groups >= 2 and valid.sum() >= .95 * repetitions
        low, high = np.quantile(draws, [.025, .975]) if estimable else (None, None)
        rows.append({'population': name, 'cases': denom, 'groups': n, 'groups_with_class': supported_groups,
                     'coverage': float(grouped[name + '_covered'].sum()/denom) if denom else None,
                     'low': float(low) if low is not None else None, 'high': float(high) if high is not None else None,
                     'valid_repetitions': int(valid.sum()), 'interval_status': 'conditional_cluster_percentile' if estimable else 'insufficient_group_support'})
    return rows


def load_sources(parent_dir, dataset, config):
    if sha256(parent_dir / 'summary.json') != config['calibration_summary_sha256']:
        raise ValueError('Frozen calibration summary changed')
    parent = json.loads((parent_dir / 'summary.json').read_text())
    for name, expected in parent['output_sha256'].items():
        if sha256(parent_dir / name) != expected:
            raise ValueError(f'Frozen calibration artifact changed: {name}')
    contract = json.loads((parent_dir / 'run_contract.json').read_text())
    if contract['config']['horizon'] != 72 or contract['config']['scope'] != 'primary':
        raise ValueError('Wrong parent target')
    if config['horizon'] != 72 or config['scope'] != 'primary':
        raise ValueError('This experiment is for 72h primary targets')
    if sha256(dataset / 'manifest.json') != config['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest changed')
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if sha256(dataset / 'master_cases.csv.gz') != manifest['outputs']['master_cases.csv.gz']['sha256']:
        raise ValueError('Master-case table changed')
    master = pd.read_csv(dataset / 'master_cases.csv.gz', low_memory=False)
    frame = pd.read_csv(parent_dir / 'calibrated_predictions.csv.gz')
    population = pd.read_csv(parent_dir / 'population_predictions.csv.gz')
    frame = frame.merge(master[['forecast_case_id', 'outcome_end_utc_72h', 'candidate_primary_label_72h']],
                        on='forecast_case_id', how='left', validate='one_to_one')
    if frame.outcome_end_utc_72h.isna().any() or not np.array_equal(frame.label, frame.candidate_primary_label_72h.fillna(-1)):
        raise ValueError('Missing cases or conflicting labels')
    if not np.array_equal(frame.label_known, frame.label.isin([0, 1])):
        raise ValueError('Conflicting known-outcome mask')
    if population.forecast_case_id.duplicated().any() or set(population.forecast_case_id) != set(master.forecast_case_id):
        raise ValueError('Full-population identities changed')
    for model in config['models']:
        method = parent['selected_methods'][model]
        if not np.array_equal(frame[f'{model}_selected'], frame[f'{model}_{method}']):
            raise ValueError('Selected probabilities changed')
        validate_probabilities(frame[f'{model}_selected'])
    return frame, population, parent


def run_conformal(parent_dir, dataset, config, output, source_code):
    if output.exists():
        raise ValueError('Choose a new output directory; never overwrite a frozen run')
    frame, population, parent = load_sources(parent_dir, dataset, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.incomplete-', dir=output.parent))
    try:
        frozen = freeze_thresholds(frame, config)
        save_json(staging / 'thresholds.json', frozen)
        frozen = json.loads((staging / 'thresholds.json').read_text())
        # Freeze all thresholds before evaluating any later outcome.
        population = population.merge(frame[['forecast_case_id', 'region_component_id', 'role']], on='forecast_case_id', how='left', validate='one_to_one')
        issue = pd.to_datetime(population.issue_utc, utc=True, format='ISO8601')
        eligible_time = issue >= pd.Timestamp(config['calibration_end'], tz='UTC')
        population = population[['forecast_case_id', 'issue_utc', 'region_component_id', 'role', 'inputs_available',
                                 'candidate_primary_label_72h', 'label_known_primary_72h', 'gru_selected', 'logistic_selected']].copy()
        metrics, yearly, intervals, availability = [], [], [], []
        frame_year = pd.to_datetime(frame.issue_utc, utc=True, format='ISO8601').dt.year
        for model in config['models']:
            available = population[f'{model}_selected'].notna()
            applies = available & eligible_time
            population[f'{model}_set_status'] = np.select([~available, ~eligible_time], ['no_supported_sharp_input', 'before_conformal_fit'], default='set_issued_exploratory')
            for method in config['methods']:
                for alpha in config['alphas']:
                    parameters = frozen['models'][model][method][str(alpha)]
                    column = f'{model}_{method}_alpha{str(alpha).replace(".", "p")}'
                    sets = predict_sets(population.loc[applies, f'{model}_selected'].to_numpy(), parameters)
                    # bit 0 represents class 0, bit 1 class 1; NA is not an empty set.
                    population[column] = pd.Series(pd.NA, index=population.index, dtype='Int8')
                    population.loc[applies, column] = (sets[:, 0].astype(int) + 2*sets[:, 1].astype(int)).astype('int8')
                    for role in config['evaluation_roles']:
                        selected = frame.role.eq(role) & frame.label_known
                        y = frame.loc[selected, 'label'].to_numpy()
                        s = predict_sets(frame.loc[selected, f'{model}_selected'].to_numpy(), parameters)
                        tags = {'role': role, 'model': model, 'method': method, 'alpha': alpha}
                        metrics.append({**tags, **set_metrics(y, s)})
                        years = frame_year[selected].to_numpy()
                        for year in sorted(set(years)):
                            subset = years == year
                            yearly.append({**tags, 'year': int(year), **set_metrics(y[subset], s[subset])})
                        times = pd.to_datetime(frame.loc[selected, 'issue_utc'], utc=True, format='ISO8601')
                        for grouping in config['bootstrap_groups']:
                            groups = frame.loc[selected, 'region_component_id'].to_numpy() if grouping == 'region_component' else (times.astype('int64') // (7*86400*10**9)).to_numpy()
                            for row in coverage_bootstrap(y, s, groups, config['bootstrap_repetitions'], config['bootstrap_seed']):
                                intervals.append({**tags, 'year': 'all', 'grouping': grouping, **row})
                            if alpha == config['primary_alpha'] and grouping == 'region_component':
                                for year in sorted(set(years)):
                                    subset = years == year
                                    for row in coverage_bootstrap(y[subset], s[subset], groups[subset], config['bootstrap_repetitions'], config['bootstrap_seed']):
                                        intervals.append({**tags, 'year': str(year), 'grouping': grouping, **row})
            for year in sorted(issue.dt.year.unique()):
                subset = issue.dt.year.eq(year)
                availability.append({'model': model, 'year': int(year), 'candidate_cases': int(subset.sum()),
                    'missing_inputs': int((subset & ~available).sum()),
                    'before_conformal_fit_with_inputs': int((subset & available & ~eligible_time).sum()),
                    'issued_sets': int((subset & applies).sum()),
                    'issued_with_unknown_outcome': int((subset & applies & ~population.label_known_primary_72h).sum())})
        population.to_csv(staging / 'population_sets.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        for name, rows in [('metrics', metrics), ('yearly_metrics', yearly), ('coverage_intervals', intervals), ('availability', availability)]:
            pd.DataFrame(rows).to_csv(staging / f'{name}.csv', index=False)
        calibration_components = set(frame.loc[conformal_mask(frame, config), 'region_component_id'])
        overlaps = {role: len(calibration_components & set(frame.loc[frame.role.eq(role), 'region_component_id'])) for role in config['evaluation_roles']}
        save_json(staging / 'run_contract.json', {'config': config, 'parent_selected_methods': parent['selected_methods'],
            'source_code_sha256': hashlib.sha256(source_code.encode()).hexdigest(),
            'versions': {'numpy': np.__version__, 'pandas': pd.__version__},
            'set_encoding': {'0': 'empty', '1': 'nonflare_only', '2': 'flare_only', '3': 'both', 'null': 'no_set_issued'},
            'selection': 'No conformal method or alpha selected from evaluation; all declared comparisons retained.'})
        (staging / 'executed_conformal_code.py').write_text(source_code)
        summary = {'status': 'completed_exploratory_conformal_evaluation', 'completed_utc': datetime.now(timezone.utc).isoformat(),
            'support': frozen['support'], 'metrics': metrics, 'population_rows': len(population),
            'missing_input_rows': int(population.gru_selected.isna().sum()),
            'calibration_evaluation_region_overlap_counts': overlaps, 'backbone_or_probability_calibration_refitted': False,
            'policy_labels_used_for_fitting_or_evaluation': False,
            'limitations': [
                'Candidate event labels, unknown-outcome exclusions and unverified continuous event coverage remain provisional.',
                'Counts are overlapping forecast windows, not independent flare events. Temporal dependence and cross-cycle drift prevent claiming nominal distribution-free guarantees here.',
                'Pooled coverage is marginal and can hide low flare-class coverage. Class-conditional coverage also requires within-class exchangeability for its theoretical guarantee.',
                'Prediction sets describe possible binary outcomes; they are not confidence intervals for flare probabilities, confidence levels for individual cases, or calibrated operational trust states.',
                'An empty set is not an OOD diagnosis; a singleton is not a safety certificate. Missing inputs stay unscored and unknown outcomes never become negatives.',
                'Bootstrap intervals are conditional per-comparison group-resampling sensitivities, not simultaneous inference; they omit label, calibration-sample and model-fitting uncertainty.',
                'Cycle-25 and 2026 data have already been inspected: results are retrospective/supplementary, not prospective independent validation.',
                'Only the 72-hour SHARP candidate baseline is evaluated. Policy validation, fusion, rolling updates and other horizons remain separate work.'
            ], 'output_sha256': {p.name: sha256(p) for p in staging.iterdir()}}
        save_json(staging / 'summary.json', summary)
        staging.rename(output)
        return summary
    except Exception as exc:
        save_json(staging / 'failure.json', {'incomplete_run': True, 'error': str(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-dir', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = run_conformal(args.parent_dir, args.dataset, json.loads(args.config.read_text()), args.output_dir, Path(__file__).read_text())
    print(json.dumps({'status': result['status'], 'support': result['support']}, indent=2))
