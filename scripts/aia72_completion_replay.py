"""Read-only saved-model verification for seeds29/43; reuse seed17's accepted replay.

This module has no command-line launcher. A separately reviewed controller must
provide the lease, exact inputs, output directory, deadline and runtime checks.
"""
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from scripts.aia72_model import TemporalAIACNNGRU
from scripts.aia72_replay import load_support, infer_rows, compare_predictions
from scripts.aia72_replay_contract import require, sha, verify_inputs, write_json
from scripts.aia72_vm_data import ExistingAIA72Dataset

PARENT = '70fbf61bd9596e19556412121d5677f1cad083e95bd4d1c3e7679602598232db'
BEST17 = '0038ef7ea4a27f5394da99cc4410b07e7cff0e867c145b322dd31474fec2869d'
COMPLETE17 = '4349d870e306fff089b8faacd7b1de3b39ce98eb239280ae79c58bf9cba38c47'
REFERENCE17 = '342b5f1155cd2be43c5fbd7d782c3da07b65a73b9ccc44bc64f2df23735c9ed8'


def check_design(c):
    needed = {'cases', 'objects', 'sources', 'normalization', 'training_contract',
              'seed17_replay_execution', 'seed17_replay_comparison'}
    needed |= {f'{kind}{seed}' for kind in ['best', 'complete', 'reference'] for seed in [17, 29, 43]}
    require(needed <= set(c['inputs']), 'Completed model input pins are missing')
    require(all(isinstance(c['inputs'][key]['sha256'], str) and len(c['inputs'][key]['sha256']) == 64
                and set(c['inputs'][key]['sha256']) <= set('0123456789abcdef') for key in needed), 'Unresolved input hash')
    require(set(c['selected_epochs']) == {'17', '29', '43'}
            and all(type(v) is int and 1 <= v <= 20 for v in c['selected_epochs'].values())
            and c['selected_epochs']['17'] == 2, 'Unresolved selected epochs')
    require(c['seeds'] == [29, 43] and c['ensemble_seeds'] == [17, 29, 43], 'Fixed seed design changed')
    require(c['parent_contract_sha256'] == PARENT, 'Scientific parent changed')
    require(c['role'] == 'model_validation' and c['case_count'] == 3905, 'Earlier replay support changed')
    require(c['ordered_support_sha256'] == 'f8dab27335622ef3220ebe86872d4b3c9fff3c7f6f7c6206de6f8484b5b982b3', 'Support identity changed')
    require(c['unique_images'] == 4661 and c['batch_size'] == 16 and c['num_workers'] == 4, 'Input design changed')
    require(c['max_cached_images_per_worker'] == 96, 'RAM cache bound changed')
    require(c['atol'] == 1e-6 and c['rtol'] == 1e-5, 'Predeclared tolerance changed')
    require(c['fitting_allowed'] is False and c['automatic_retry'] is False, 'Fitting or retry prohibited')
    for name, digest in [('best17', BEST17), ('complete17', COMPLETE17), ('reference17', REFERENCE17)]:
        require(c['inputs'][name]['sha256'] == digest, 'Previously accepted seed17 changed')
    require(c['inputs']['training_contract']['sha256'] == PARENT, 'Original contract changed')
    return c


def check_selected_reference(complete, weights, reference, selected, parent):
    require(complete['status'] == 'seed_fit_complete_pending_replay', 'Seed fitting not completed')
    require(complete['contract_sha256'] == parent, 'Completed seed parent changed')
    require(sha(weights) == complete['best_checkpoint_sha256'], 'Selected weights changed')
    require(reference.forecast_case_id.is_unique, 'Duplicate selected reference case')
    require(reference.forecast_case_id.tolist() == selected.forecast_case_id.tolist(), 'Reference support/order changed')
    require(np.array_equal(reference.label, selected.label), 'Reference labels changed')
    logits, labels = reference.logit.to_numpy(), reference.label.to_numpy()
    require(np.isfinite(logits).all() and np.isin(labels, [0, 1]).all(), 'Invalid reference predictions')
    loss = float(np.mean(np.logaddexp(0., logits) - labels*logits))
    require(abs(loss-complete['selection_log_loss']) < 1e-12, 'Saved selection loss differs')
    return loss


def selection_ensemble(references):
    require(set(references) == {17, 29, 43}, 'All three fixed seeds are required')
    first = references[17]
    require(len(first) > 0 and first.forecast_case_id.is_unique, 'Empty or duplicate ensemble support')
    require(first.label.isin([0, 1]).all(), 'Unknown ensemble outcomes')
    result = first[['forecast_case_id', 'label']].copy()
    for seed in [17, 29, 43]:
        frame = references[seed]
        require(frame.forecast_case_id.tolist() == first.forecast_case_id.tolist(), 'Ensemble support/order changed')
        require(np.array_equal(frame.label, first.label), 'Ensemble labels changed')
        logits = frame.logit.to_numpy()
        require(np.isfinite(logits).all(), 'Nonfinite ensemble logits')
        result[f'probability_seed_{seed}'] = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
    result['probability'] = result[[f'probability_seed_{s}' for s in [17, 29, 43]]].mean(axis=1)
    return result


def load_completed_support(c, source_root=None):
    """Fail before loading image pixels if any completed model/evidence is absent."""
    check_design(c)
    paths = verify_inputs(c, source_root)
    selected, _, records, normalization = load_support(c, {**paths, 'reference': paths['reference17']})
    references, completions = {}, {}
    for seed in [17, 29, 43]:
        complete = json.loads(paths[f'complete{seed}'].read_text())
        require(complete['seed'] == seed and complete['best_epoch'] == c['selected_epochs'][str(seed)], 'Selected epoch changed')
        reference = pd.read_csv(paths[f'reference{seed}'])
        check_selected_reference(complete, paths[f'best{seed}'], reference, selected, c['parent_contract_sha256'])
        references[seed], completions[seed] = reference, complete
    accepted = json.loads(paths['seed17_replay_execution'].read_text())
    evidence = json.loads(paths['seed17_replay_comparison'].read_text())
    require(accepted['status'] == 'technical_checkpoint_replay_verified'
            and accepted['result_sha256'] == sha(paths['seed17_replay_comparison']), 'Seed17 replay not accepted')
    require(evidence['selected_seed'] == 17 and evidence['selected_epoch'] == 2
            and evidence['cases'] == 3905 and evidence['all_logits_within_tolerance']
            and evidence['atol'] == c['atol'] and evidence['rtol'] == c['rtol'], 'Accepted seed17 replay differs')
    require(abs(evidence['saved_log_loss']-completions[17]['selection_log_loss']) < 1e-12, 'Seed17 replay loss differs')
    return paths, selected, references, completions, records, normalization


def replay_selected_models(c, paths, selected, references, completions, dataset, output,
                           deadline, loader_factory, guard, device='cuda',
                           model_factory=TemporalAIACNNGRU):
    """Use a fresh directory and a caller-supplied loader that retains its lease.

    The caller checks exact runtime, controller context, design and pinned inputs.
    This core has a CPU-injectable model/loader for real inference-loop tests.
    """
    output = Path(output)
    output.mkdir(exist_ok=False)
    before = {name: sha(path) for name, path in paths.items()}
    reports, artifacts = [], {}
    started = time.monotonic()
    for seed in [29, 43]:
        guard()
        require(time.monotonic() < deadline, 'Replay deadline reached before next seed')
        complete = completions[seed]
        check_selected_reference(complete, paths[f'best{seed}'], references[seed], selected, c['parent_contract_sha256'])
        model = model_factory()
        model.load_state_dict(torch.load(paths[f'best{seed}'], map_location='cpu', weights_only=True))
        model.to(device)
        batches = loader_factory(dataset, seed)
        def progress(count):
            write_json(output/'progress.json', {'seed': seed, 'cases': count, 'of': len(selected)})
        replayed = infer_rows(model, batches, selected, device, deadline, progress, guard)
        comparison, report = compare_predictions(replayed, references[seed], c['atol'], c['rtol'])
        report.update(seed=seed, best_epoch=complete['best_epoch'], fitting_steps=0,
                      best_checkpoint_sha256=sha(paths[f'best{seed}']))
        csv = output/f'seed_{seed}_comparison.csv.gz'
        comparison.to_csv(csv, index=False, compression={'method': 'gzip', 'mtime': 0})
        receipt = output/f'seed_{seed}_comparison.json'
        write_json(receipt, report)
        require(report['all_logits_within_tolerance'], 'Saved-model replay mismatch')
        artifacts[csv.name], artifacts[receipt.name] = sha(csv), sha(receipt)
        reports.append(report)
        del model, batches, replayed
        gc.collect()
        guard()
    ensemble = selection_ensemble(references)
    path = output/'three_seed_selection_predictions.csv.gz'
    ensemble.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0})
    artifacts[path.name] = sha(path)
    require(before == {name: sha(path) for name, path in paths.items()}, 'Replay altered pinned inputs')
    guard()
    require(time.monotonic() < deadline, 'Replay deadline reached before publication')
    result = {'status': 'technical_three_seed_replay_verified_pending_controller',
              'replayed_seeds': [29, 43], 'seed17_replay_reused': True, 'seed17_repeated': False,
              'selection_cases': len(ensemble), 'comparisons': reports, 'output_sha256': artifacts,
              'replay_elapsed_seconds': time.monotonic()-started, 'fitting_steps': 0,
              'inputs_unchanged': True, 'scientific_acceptance': False, 'total_analysis_complete': False,
              'later_period_evaluation': False,
              'limitations': 'Saved-model reproducibility on earlier model-selection cases only. Ensemble averages all three saved reference probabilities after replay checks. No new fitting, selection, calibration, later-period evaluation or fusion.'}
    write_json(output/'result.json', result)
    guard()
    return result


def run_completion_replay(c, output, deadline, loader_factory, guard):
    """Controller entry point; all inputs checked before/after read-only inference."""
    guard()
    paths, selected, references, completions, records, normalization = load_completed_support(c)
    dataset = ExistingAIA72Dataset(selected, records, normalization, c['max_cached_images_per_worker'])
    result = replay_selected_models(c, paths, selected, references, completions, dataset,
                                    output, deadline, loader_factory, guard)
    verify_inputs(c)
    guard()
    return result
