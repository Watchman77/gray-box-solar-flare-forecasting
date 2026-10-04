"""Real CPU inference checks for the post-fit replay; never starts CUDA or the VM."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA before Python starts'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.aia72_completion_replay import (
    PARENT, BEST17, COMPLETE17, REFERENCE17, check_design, check_selected_reference,
    replay_selected_models, selection_ensemble)
from scripts.aia72_replay_contract import sha


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
    def forward(self, x):
        return self.linear(x).squeeze(-1)


class Cases:
    def __init__(self, frame):
        self.frame = frame
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, i):
        return torch.tensor([i/10., 1.]), torch.tensor(float(self.frame.label.iloc[i])), self.frame.forecast_case_id.iloc[i]


class CompletionReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
        torch.set_num_threads(1)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def fixture(self, root):
        frame = pd.DataFrame({'forecast_case_id': ['c0', 'c1', 'c2', 'c3'], 'label': [0, 1, 1, 0]})
        paths, references, completions = {}, {}, {}
        inputs = torch.tensor([[i/10., 1.] for i in range(4)])
        for seed in [17, 29, 43]:
            model = TinyModel()
            with torch.no_grad():
                model.linear.weight.copy_(torch.tensor([[seed/100., -.7]]))
                model.linear.bias.fill_(seed/20.)
                logits = model(inputs).numpy()
            path = root/f'best{seed}.pt'
            torch.save(model.state_dict(), path)
            ref = frame.assign(logit=logits.astype(float))
            complete = {'status': 'seed_fit_complete_pending_replay', 'seed': seed,
                        'best_epoch': 2 if seed == 17 else 3, 'contract_sha256': PARENT,
                        'best_checkpoint_sha256': sha(path),
                        'selection_log_loss': float(np.mean(np.logaddexp(0., ref.logit)-ref.label*ref.logit))}
            references[seed], completions[seed], paths[f'best{seed}'] = ref, complete, path
        return {'parent_contract_sha256': PARENT, 'atol': 1e-6, 'rtol': 1e-5}, paths, frame, references, completions, Cases(frame)

    def call(self, values, output, loader=None, guard=lambda: None, deadline=None):
        return replay_selected_models(*values, output, deadline or time.monotonic()+30,
                                      loader or (lambda dataset, seed: DataLoader(dataset, batch_size=2)),
                                      guard, device='cpu', model_factory=TinyModel)

    def test_reloads_only_29_43_preserves_weights_and_averages_all_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); values = self.fixture(root)
            before = {key: sha(p) for key, p in values[1].items()}
            called = []
            def loader(dataset, seed):
                called.append(seed)
                return DataLoader(dataset, batch_size=2)
            with mock.patch('torch.optim.AdamW', side_effect=AssertionError('No optimizer allowed')):
                result = self.call(values, root/'replay', loader)
            self.assertEqual(called, [29, 43])
            self.assertEqual(result['fitting_steps'], 0)
            self.assertFalse(result['scientific_acceptance'])
            self.assertTrue(result['seed17_replay_reused'])
            self.assertEqual(before, {key: sha(p) for key, p in values[1].items()})
            frame = pd.read_csv(root/'replay/three_seed_selection_predictions.csv.gz')
            expected = np.stack([1/(1+np.exp(-values[3][s].logit.to_numpy())) for s in [17,29,43]]).mean(axis=0)
            np.testing.assert_allclose(frame.probability, expected, rtol=0, atol=1e-15)
            for name, digest in result['output_sha256'].items():
                self.assertEqual(sha(root/'replay'/name), digest)

    def test_deadline_prevents_model_or_loader_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); values = self.fixture(root)
            with mock.patch(__name__+'.TinyModel', side_effect=AssertionError('No model start')):
                with self.assertRaisesRegex(ValueError, 'deadline'):
                    self.call(values, root/'expired', deadline=time.monotonic()-1)
            self.assertFalse((root/'expired/result.json').exists())

    def test_reordered_loader_fails_before_second_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); values = self.fixture(root); called=[]
            def loader(dataset, seed):
                called.append(seed)
                return DataLoader(dataset, batch_size=2, sampler=[1, 0, 2, 3])
            with self.assertRaisesRegex(ValueError, 'Loader omitted/reordered'):
                self.call(values, root/'badorder', loader)
            self.assertEqual(called, [29])
            self.assertFalse((root/'badorder/result.json').exists())

    def test_mismatch_preserves_failed_comparison_without_success_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); values = self.fixture(root)
            ref = values[3][43]
            ref['logit'] += .01
            values[4][43]['selection_log_loss'] = float(np.mean(np.logaddexp(0., ref.logit)-ref.label*ref.logit))
            with self.assertRaisesRegex(ValueError, 'Saved-model replay mismatch'):
                self.call(values, root/'mismatch')
            failed = json.loads((root/'mismatch/seed_43_comparison.json').read_text())
            self.assertEqual(failed['mismatched_cases'], 4)
            self.assertFalse((root/'mismatch/result.json').exists())

    def test_changed_checkpoint_rejected_and_existing_attempt_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); values = self.fixture(root)
            with values[1]['best29'].open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Selected weights changed'):
                self.call(values, root/'changed')
            with self.assertRaises(FileExistsError):
                self.call(values, root/'changed')

    def test_input_mutation_during_replay_rejects_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); values = self.fixture(root); changed = []
            def guard():
                if (root/'mutated/seed_29_comparison.json').exists() and not changed:
                    with values[1]['best17'].open('ab') as stream:
                        stream.write(b'mutated')
                    changed.append(True)
            with self.assertRaisesRegex(ValueError, 'Replay altered pinned inputs'):
                self.call(values, root/'mutated', guard=guard)
            self.assertFalse((root/'mutated/result.json').exists())

    def test_ensemble_rejects_missing_seed_changed_labels_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            refs = self.fixture(Path(tmp))[3]
            with self.assertRaisesRegex(ValueError, 'All three'):
                selection_ensemble({17: refs[17], 29: refs[29]})
            for field, value in [('label', -1), ('logit', np.nan), ('forecast_case_id', 'wrong')]:
                bad = copy.deepcopy(refs); bad[43].loc[0, field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    selection_ensemble(bad)

    def test_design_rejects_unresolved_models_and_changed_tolerance(self):
        names = {'cases','objects','sources','normalization','training_contract',
                 'seed17_replay_execution','seed17_replay_comparison'}
        names |= {f'{k}{s}' for k in ['best','complete','reference'] for s in [17,29,43]}
        c = {'inputs': {key: {'sha256':'a'*64} for key in names}, 'selected_epochs': {'17':2,'29':3,'43':3},
             'seeds':[29,43], 'ensemble_seeds':[17,29,43], 'parent_contract_sha256':PARENT,
             'role':'model_validation','case_count':3905,'unique_images':4661,'batch_size':16,'num_workers':4,
             'ordered_support_sha256':'f8dab27335622ef3220ebe86872d4b3c9fff3c7f6f7c6206de6f8484b5b982b3',
             'max_cached_images_per_worker':96,'atol':1e-6,'rtol':1e-5,'fitting_allowed':False,'automatic_retry':False}
        for key, value in [('best17',BEST17),('complete17',COMPLETE17),('reference17',REFERENCE17),('training_contract',PARENT)]:
            c['inputs'][key]['sha256']=value
        check_design(c)
        for field, value in [('atol',1e-5),('fitting_allowed',True),('seeds',[17,29,43])]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_design({**c,field:value})
        bad=copy.deepcopy(c);bad['inputs']['best43']['sha256']=None
        with self.assertRaisesRegex(ValueError,'Unresolved input hash'):
            check_design(bad)


if __name__ == '__main__':
    unittest.main()
