"""Saved-checkpoint inference and notebook execution. Never imports a fitting driver."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

import nbformat
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.aia72_model import TemporalAIACNNGRU
from scripts.aia72_vm_data import ExistingAIA72Dataset
from scripts.aia72_data import LAGS, validate_case_times
from scripts.aia72_replay_contract import check_contract, check_guards, json_sha, require, sha, verify_inputs, write_json


class ReplayDeadline(Exception):
    pass


def before_deadline(deadline):
    if time.monotonic() >= deadline:
        raise ReplayDeadline('Replay incomplete: reserved cleanup period has begun')


def load_support(contract, paths):
    frame = pd.read_csv(paths['cases'])
    require(frame.forecast_case_id.is_unique, 'Duplicate frozen cases')
    require(frame.role.value_counts().to_dict() == {'train': 25586, 'model_validation': 3905}, 'Frozen roles changed')
    selected = frame.loc[frame.role.eq(contract['role'])].reset_index(drop=True)
    validate_case_times(selected)
    require(len(selected) == contract['case_count'], 'Replay count changed')
    require(selected.label.isin([0, 1]).all(), 'Unknown replay outcomes')
    identities = list(zip(selected.forecast_case_id.tolist(), selected.label.astype(int).tolist()))
    require(json_sha(identities) == contract['ordered_support_sha256'], 'Replay identities/order/labels changed')
    reference = pd.read_csv(paths['reference'])
    require(reference.forecast_case_id.tolist() == selected.forecast_case_id.tolist(), 'Reference support/order changed')
    require(np.array_equal(reference.label.to_numpy(), selected.label.to_numpy()), 'Reference labels changed')
    require(np.isfinite(reference.logit.to_numpy()).all(), 'Invalid reference logits')
    normalization = json.loads(paths['normalization'].read_text())
    require(normalization['training_case_ids'] == sorted(frame.loc[frame.role.eq('train'), 'forecast_case_id'].tolist()),
            'Normalization training support changed')
    require(normalization['fit_role'] == 'train', 'Normalization was not fitted on training cases')
    records = json.loads(paths['sources'].read_text())
    objects = json.loads(paths['objects'].read_text())
    wanted = set(selected[[f'history_uri_tminus{x}' for x in LAGS]].to_numpy().ravel())
    for uri in wanted:
        require(uri in records and uri in objects, 'Replay image metadata missing')
        for name in ['uri', 'generation', 'bytes', 'md5_base64']:
            require(records[uri][name] == objects[uri][name], 'Bound cache metadata differs')
        require(Path(records[uri]['path']).resolve().is_relative_to('/mnt/disks/aia-cache/cycle24'), 'Unexpected cache path')
    require(len(wanted) == contract['unique_images'], 'Replay image support changed')
    return selected, reference, {uri: records[uri] for uri in wanted}, normalization


def infer_rows(model, batches, expected, device, deadline, progress=None, guard=None):
    """Independent inference loop: fixed order, no optimizer, unchanged model buffers."""
    model.eval()
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rows = []
    if guard:
        guard()
    with torch.no_grad():
        for images, labels, identities in batches:
            before_deadline(deadline)
            if guard:
                guard()
            offset = len(rows)
            require(list(identities) == expected.forecast_case_id.iloc[offset:offset + len(identities)].tolist(),
                    'Loader omitted/reordered/duplicated cases')
            require(np.array_equal(labels.numpy(), expected.label.iloc[offset:offset + len(labels)].to_numpy()),
                    'Loader outcomes changed')
            values = model(images.to(device, non_blocking=True)).detach().cpu().numpy()
            require(values.shape == (len(identities),) and np.isfinite(values).all(), 'Invalid replay logits')
            rows.extend({'forecast_case_id': identity, 'label': int(label), 'logit': float(value)}
                        for identity, label, value in zip(identities, labels, values))
            if progress:
                progress(len(rows))
            before_deadline(deadline)
            if guard:
                guard()
    require(len(rows) == len(expected), 'Incomplete replay support')
    require(all(torch.equal(original[k], v.detach().cpu()) for k, v in model.state_dict().items()),
            'Inference mutated parameters or buffers')
    return pd.DataFrame(rows)


def compare_predictions(replayed, reference, atol=1e-6, rtol=1e-5):
    require(replayed.forecast_case_id.is_unique, 'Duplicate replay output')
    require(replayed.forecast_case_id.tolist() == reference.forecast_case_id.tolist(), 'Prediction support/order differs')
    require(np.array_equal(replayed.label.to_numpy(), reference.label.to_numpy()), 'Prediction labels differ')
    got, saved = replayed.logit.to_numpy(), reference.logit.to_numpy()
    require(np.isfinite(got).all() and np.isfinite(saved).all(), 'Nonfinite prediction')
    differences = np.abs(got - saved)
    within = differences <= atol + rtol * np.abs(saved)
    y = reference.label.to_numpy()
    actual_loss = float(np.mean(np.logaddexp(0., got) - y * got))
    saved_loss = float(np.mean(np.logaddexp(0., saved) - y * saved))
    comparison = replayed.rename(columns={'logit': 'replayed_logit'}).copy()
    comparison['saved_logit'] = saved
    comparison['absolute_difference'] = differences
    comparison['within_tolerance'] = within
    return comparison, {'cases': len(y), 'all_logits_within_tolerance': bool(within.all()),
                        'mismatched_cases': int((~within).sum()), 'max_absolute_logit_difference': float(differences.max()),
                        'saved_log_loss': saved_loss, 'replayed_log_loss': actual_loss,
                        'log_loss_difference': actual_loss - saved_loss, 'atol': atol, 'rtol': rtol}


def run_replay(contract, output, deadline, owner_sha):
    """Execute the pinned seed-17 review; any partial/failing result is unverified."""
    check_contract(contract)
    guard = lambda: check_guards(contract, output, owner_sha, deadline)
    guard()
    paths = verify_inputs(contract)
    before_deadline(deadline)
    selected, reference, records, normalization = load_support(contract, paths)
    guard()
    runtime = {'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
               'python': sys.version, 'device': torch.cuda.get_device_name(0)}
    require(runtime == contract['runtime'], 'Training/replay runtime differs')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(17)
    model = TemporalAIACNNGRU()
    weights = torch.load(paths['checkpoint'], map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.to('cuda').eval()
    dataset = ExistingAIA72Dataset(selected, records, normalization, max_cached_images=96)
    batches = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False, num_workers=4,
                         pin_memory=True, multiprocessing_context='spawn', prefetch_factor=2,
                         generator=torch.Generator().manual_seed(200017))
    output = Path(output)
    def progress(count):
        if count % 256 == 0 or count == len(selected):
            guard()
            write_json(output / 'progress.json', {'status': 'replaying_unverified', 'cases': count,
                       'expected_cases': len(selected), 'fitting_steps': 0})
            print(f'Replayed {count:,}/{len(selected):,} cases', flush=True)
    replayed = infer_rows(model, batches, selected, 'cuda', deadline, progress, guard)
    before_deadline(deadline)
    comparison, result = compare_predictions(replayed, reference, contract['atol'], contract['rtol'])
    completion = json.loads(paths['completion'].read_text())
    require(abs(result['saved_log_loss'] - completion['selection_log_loss']) < 1e-12,
            'Saved metrics differ from completion receipt')
    guard()
    comparison.to_csv(output / 'prediction_comparison.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    verify_inputs(contract)
    before_deadline(deadline)
    result.update({'status': 'comparison_passed_pending_controller' if result['all_logits_within_tolerance'] else 'prediction_mismatch',
                   'selected_seed': 17, 'selected_epoch': 2, 'model_state_unchanged': True,
                   'inputs_unchanged': True, 'fitting_steps': 0, 'source_image_downloads': 0,
                   'scientific_acceptance': False, 'later_period_evaluation': False,
                   'limitations': 'Same architecture/preprocessing implementation; this tests serialized-checkpoint reproducibility on the earlier selection block, not independent scientific validation.'})
    guard()
    write_json(output / 'comparison_result.json', result)
    guard()
    require(result['all_logits_within_tolerance'], 'Replay predictions differ from saved values')
    return result


def execute_notebook(root, output, context):
    """Run plain-Python notebook cells in this supervised process group (no detached kernel)."""
    root, output = Path(root), Path(output)
    nb = nbformat.read(root / 'notebooks/10_AIA_72h_Checkpoint_Replay.ipynb', as_version=4)
    target = output / '10_AIA_72h_Checkpoint_Replay_EXECUTED.ipynb'
    namespace = {'__name__': '__replay_notebook__', 'REPLAY_CONTEXT': context}
    count = 0
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        count += 1; cell.execution_count = count; cell.outputs = []
        def display(value):
            data = {'text/plain': str(value)}
            if isinstance(value, pd.DataFrame):
                data['text/html'] = value.to_html(index=False, float_format=lambda x: f'{x:.8g}')
            cell.outputs.append(nbformat.v4.new_output('display_data', data=data))
        namespace['display'] = display
        class Tee(io.StringIO):
            def write(self, text):
                sys.__stdout__.write(text); sys.__stdout__.flush()
                return super().write(text)
        capture = Tee()
        try:
            with contextlib.redirect_stdout(capture):
                exec(compile(cell.source, str(target), 'exec'), namespace)
        except BaseException as error:
            cell.outputs.append(nbformat.v4.new_output('error', ename=type(error).__name__, evalue=str(error),
                                                      traceback=traceback.format_exc().splitlines()))
            raise
        finally:
            if capture.getvalue():
                cell.outputs.append(nbformat.v4.new_output('stream', name='stdout', text=capture.getvalue()))
            nb.metadata['execution_engine'] = 'supervised in-process plain Python; no detached Jupyter kernel'
            temp = target.with_suffix('.tmp'); nbformat.write(nb, temp); temp.replace(target)


def main():
    # The inherited lease survives a controller crash, preventing another cooperating GPU job.
    import fcntl
    from scripts.aia72_replay_contract import LOCK_INODE
    fd = int(os.environ['GRAYBOX_REPLAY_LEASE_FD'])
    require(os.fstat(fd).st_ino == LOCK_INODE, 'No inherited shared-GPU lease')
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    context = json.loads(os.environ['GRAYBOX_REPLAY_CONTEXT'])
    require(context['mode'] == 'authorized_gpu_replay', 'Launcher authorization missing')
    root, output = Path(context['bundle_root']), Path(context['output'])
    execute_notebook(root, output, context)


if __name__ == '__main__':
    main()
