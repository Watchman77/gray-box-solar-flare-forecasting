"""Resumable 72-hour AIA fitting; no calibration or evaluation-period data access."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from scripts.aia72_model import TemporalAIACNNGRU
from scripts.aia72_vm_data import ExistingAIA72Dataset, write_json
from scripts.dataset_io import file_sha256


def atomic_save(value, path):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary); temporary.replace(path)


def rng_state():
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(saved):
    random.setstate(saved['python']); n = saved['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(saved['torch'].cpu())
    if saved['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda']])


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(dataset, config, seed, epoch=0, cursor=0, shuffled=False):
    generator = torch.Generator().manual_seed(seed + 100000 + epoch)
    order = torch.randperm(len(dataset), generator=generator).tolist() if shuffled else list(range(len(dataset)))
    options = {'batch_size': config['batch_size'], 'shuffle': False, 'num_workers': config['num_workers'],
               'pin_memory': torch.cuda.is_available(), 'drop_last': False,
               'generator': torch.Generator().manual_seed(seed + 200000 + epoch)}
    if config['num_workers']:
        options.update(multiprocessing_context='spawn', prefetch_factor=2)
    return DataLoader(Subset(dataset, order[cursor:]), **options)


def validation_predictions(model, dataset, config, seed, device, deadline):
    model.eval(); rows = []
    with torch.no_grad():
        for images, labels, identities in loader(dataset, config, seed):
            if time.monotonic() >= deadline:
                return None
            logits = model(images.to(device, non_blocking=True)).detach().cpu().numpy()
            if not np.isfinite(logits).all():
                raise ValueError('Nonfinite validation logits')
            rows.extend({'forecast_case_id': identity, 'label': float(y), 'logit': float(logit)}
                        for identity, y, logit in zip(identities, labels, logits))
    result = pd.DataFrame(rows)
    if result.forecast_case_id.tolist() != dataset.frame.forecast_case_id.tolist():
        raise ValueError('Validation support/order changed')
    return result


def train_seed(train, validation, config, seed, output, contract_sha256, device, deadline,
               model_factory=TemporalAIACNNGRU, stop_after_steps=None):
    """Save next-case cursor and all RNG/optimizer state for an exact restart."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f'seed_{seed}_resume.pt'
    completion = output / f'seed_{seed}_complete.json'
    if completion.exists():
        done = json.loads(completion.read_text())
        if done['contract_sha256'] != contract_sha256 or file_sha256(output / f'seed_{seed}_best.pt') != done['best_checkpoint_sha256']:
            raise ValueError('Completed seed contract/checkpoint changed')
        return done
    set_seed(seed)
    model = model_factory().to(device)
    prior = float(train.frame.label.mean())
    if not 0 < prior < 1:
        raise ValueError('Training needs both outcome classes')
    with torch.no_grad():
        model.head[-1].bias.fill_(math.log(prior / (1 - prior)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    state = {'epoch': 1, 'cursor': 0, 'loss_sum': 0., 'best_loss': None, 'best_epoch': None,
             'stale': 0, 'history': [], 'elapsed_seconds': 0., 'steps': 0}
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved['contract_sha256'] != contract_sha256:
            raise ValueError('Cannot resume under changed code/data/configuration')
        state = saved['state']; model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'])
    start = time.monotonic(); previous_elapsed = state['elapsed_seconds']; last_save = start
    criterion = nn.BCEWithLogitsLoss()
    def save(status='running'):
        nonlocal last_save
        state['elapsed_seconds'] = previous_elapsed + time.monotonic() - start
        atomic_save({'contract_sha256': contract_sha256, 'state': state, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'rng': rng_state()}, checkpoint)
        write_json(output / 'progress.json', {'utc': datetime.now(timezone.utc).isoformat(), 'status': status,
                   'seed': seed, 'epoch': state['epoch'], 'case_cursor': state['cursor'], 'training_cases': len(train),
                   'steps': state['steps'], 'elapsed_seconds': state['elapsed_seconds'],
                   'contract_sha256': contract_sha256, 'full_training_complete': False})
        last_save = time.monotonic()
    save()
    while state['epoch'] <= config['max_epochs'] and state['stale'] < config['patience']:
        epoch = state['epoch']
        if not 0 <= state['cursor'] <= len(train) or (state['cursor'] % config['batch_size'] and state['cursor'] != len(train)):
            raise ValueError('Invalid restart cursor')
        model.train()
        for images, labels, identities in loader(train, config, seed, epoch, state['cursor'], shuffled=True):
            if time.monotonic() >= deadline or (output / 'STOP_REQUESTED').exists():
                save('checkpointed_incomplete'); return {'status': 'checkpointed_incomplete', 'seed': seed}
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device, non_blocking=True)), labels.to(device, non_blocking=True))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            state['cursor'] += len(labels); state['loss_sum'] += float(loss.detach()) * len(labels); state['steps'] += 1
            if time.monotonic() - last_save >= config['checkpoint_interval_seconds']:
                save(); print(json.dumps({'seed': seed, 'epoch': epoch, 'cases': state['cursor'], 'of': len(train)}), flush=True)
            if stop_after_steps is not None and state['steps'] >= stop_after_steps:
                save('test_interruption'); return {'status': 'test_interruption', 'seed': seed}
        save()
        predictions = validation_predictions(model, validation, config, seed, device, deadline)
        if predictions is None:
            save('checkpointed_incomplete'); return {'status': 'checkpointed_incomplete', 'seed': seed}
        y, logits = predictions.label.to_numpy(), predictions.logit.to_numpy()
        val_loss = float(np.mean(np.logaddexp(0., logits) - y * logits))
        if not math.isfinite(val_loss):
            raise ValueError('Invalid model-selection loss')
        predictions['probability'] = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
        predictions.to_csv(output / f'seed_{seed}_epoch_{epoch:02d}_selection.csv.gz', index=False,
                           compression={'method': 'gzip', 'mtime': 0})
        if state['best_loss'] is None or val_loss < state['best_loss']:
            state['best_loss'], state['best_epoch'], state['stale'] = val_loss, epoch, 0
            atomic_save(model.state_dict(), output / f'seed_{seed}_best.pt')
        else:
            state['stale'] += 1
        row = {'seed': seed, 'epoch': epoch, 'training_log_loss': state['loss_sum'] / len(train),
               'selection_log_loss': val_loss, 'best_epoch': state['best_epoch']}
        state['history'].append(row); print(json.dumps(row), flush=True)
        pd.DataFrame(state['history']).to_csv(output / f'seed_{seed}_history.csv', index=False)
        state['epoch'] += 1; state['cursor'] = 0; state['loss_sum'] = 0.; save()
    done = {'status': 'seed_fit_complete_pending_replay', 'seed': seed, 'contract_sha256': contract_sha256,
            'best_epoch': state['best_epoch'], 'selection_log_loss': state['best_loss'],
            'epochs_completed': len(state['history']), 'elapsed_seconds': state['elapsed_seconds'],
            'best_checkpoint_sha256': file_sha256(output / f'seed_{seed}_best.pt')}
    write_json(completion, done)
    return done


def benchmark(train, config, device, output):
    """Four timed batches after two warm-ups; discard all benchmark weights."""
    set_seed(17); model = TemporalAIACNNGRU().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    timings = []; last = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    iterator = iter(loader(train, {**config, 'num_workers': config['num_workers']}, 17))
    for index in range(6):
        images, labels, _ = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(images.to(device)), labels.to(device))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite benchmark loss')
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); torch.cuda.synchronize()
        now = time.monotonic()
        if index >= 2:
            timings.append(now - last)
        last = now
    del iterator
    seconds = float(np.mean(timings))
    result = {'status': 'technical_benchmark_only', 'device': torch.cuda.get_device_name(0),
              'batch_size': config['batch_size'], 'timed_batches': 4, 'seconds_per_batch_including_loader': seconds,
              'rough_training_epoch_minutes': math.ceil(len(train) / config['batch_size']) * seconds / 60,
              'peak_gpu_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3,
              'limitation': 'Tiny sample; cold disk cache, full normalization/data checking and validation can change full-epoch time',
              'weights_discarded': True}
    write_json(Path(output) / 'benchmark.json', result)
    del model, optimizer; torch.cuda.empty_cache()
    return result


def run_training(frame, records, normalization, config, output, contract_sha256, device='cuda'):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    train_frame = frame[frame.role.eq('train')]; validation_frame = frame[frame.role.eq('model_validation')]
    if len(train_frame) != 25586 or len(validation_frame) != 3905:
        raise ValueError('Training support changed')
    train = ExistingAIA72Dataset(train_frame, records, normalization, config['max_cached_images_per_worker'])
    validation = ExistingAIA72Dataset(validation_frame, records, normalization, config['max_cached_images_per_worker'])
    if not (output / 'benchmark.json').exists():
        print(json.dumps(benchmark(train, config, device, output)), flush=True)
    elapsed = 0.
    for seed in config['seeds']:
        path = output / f'seed_{seed}_resume.pt'
        if path.exists():
            saved = torch.load(path, weights_only=True, map_location='cpu')
            if saved['contract_sha256'] != contract_sha256:
                raise ValueError('Changed contract before resumption')
            elapsed += saved['state']['elapsed_seconds']
    remaining = config['max_total_training_seconds'] - elapsed
    if remaining <= 0:
        return {'status': 'training_time_limit_reached', 'full_training_complete': False}
    deadline = time.monotonic() + min(config['max_invocation_seconds'], remaining)
    results = []
    for seed in config['seeds']:
        result = train_seed(train, validation, config, seed, output, contract_sha256, device, deadline)
        results.append(result)
        if result['status'] != 'seed_fit_complete_pending_replay':
            break
    fitted = len(results) == len(config['seeds']) and all(r['status'] == 'seed_fit_complete_pending_replay' for r in results)
    summary = {'status': 'all_seeds_fit_pending_replay' if fitted else 'checkpointed_incomplete',
               'seeds': results, 'full_training_complete': fitted, 'scientific_acceptance': False,
               'later_period_predictions_created': False, 'calibration_or_fusion_performed': False,
               'contract_sha256': contract_sha256}
    write_json(output / 'invocation_result.json', summary)
    return summary
