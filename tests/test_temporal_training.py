import unittest

import numpy as np
import pandas as pd

try:
    from scripts import train_sharp_temporal as training
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    training = None


@unittest.skipIf(training is None, "Install requirements-training.txt for training tests")
class TemporalTrainingTests(unittest.TestCase):
    def test_outcome_and_reporting_delay_purge_with_unknown_labels(self):
        index = pd.DataFrame({"issue_utc": ["2013-12-27T00:00:00Z", "2013-12-29T00:00:00Z",
                                                  "2014-01-02T00:00:00Z", "2014-01-03T00:00:00Z"]})
        master = pd.DataFrame({"outcome_end_utc_72h": ["2013-12-30T00:00:00Z", "2014-01-01T00:00:00Z",
                                                            "2014-01-05T00:00:00Z", "2014-01-06T00:00:00Z"]})
        config = {"horizon": 72, "reporting_delay_hours": 24, "blocks": [
            {"role": "train", "start": "2010-01-01", "end": "2014-01-01"},
            {"role": "model_validation", "start": "2014-01-01", "end": "2014-07-01"}]}
        roles = training.make_roles(index, master, np.array([True, True, False, True]),
                                    np.array([True, True, True, False]), config)
        self.assertEqual(roles.tolist(), ["train", "purged_outcome_boundary", "unknown_label_excluded", "missing_input_excluded"])

    def test_heldout_extremes_cannot_change_training_transform(self):
        raw = np.arange(24, dtype=float).reshape(4, 3, 2)
        selected = np.array([True, True, False, False])
        original = training.fit_transform(raw, selected)
        raw[~selected] = 1e100
        changed = training.fit_transform(raw, selected)
        for key in original:
            np.testing.assert_array_equal(original[key], changed[key])
        np.testing.assert_array_equal(training.transform(raw[:1], original), training.transform(raw, original)[:1])

    def test_metrics_handle_ties_and_single_class_without_invented_auc(self):
        tied = training.metrics(np.array([1, 0, 1, 0]), np.full(4, 0.5), 0.5)
        self.assertEqual(tied["average_precision"], 0.5)
        self.assertEqual(tied["roc_auc"], 0.5)
        self.assertEqual(tied["brier_skill_train_climatology"], 0)
        perfect = training.metrics(np.array([1, 0]), np.array([0.9, 0.1]), 0.5)
        self.assertEqual(perfect["roc_auc"], 1)
        self.assertEqual(perfect["average_precision"], 1)
        self.assertIsNone(training.metrics(np.zeros(3), np.full(3, 0.1), 0.5)["roc_auc"])
        with self.assertRaises(ValueError):
            training.metrics(np.array([-1, 0]), np.array([0.2, 0.3]), 0.5)

    def test_training_and_saved_checkpoint_inference(self):
        import tempfile
        from pathlib import Path
        torch = training.torch
        torch.set_num_threads(1)
        rng = np.random.default_rng(1)
        x = torch.from_numpy(rng.normal(size=(24, 3, 2)).astype(np.float32))
        y = np.tile([0, 1], 12)
        roles = np.array(["train"]*16 + ["model_validation"]*8)
        config = {"hidden_size": 4, "head_dropout": 0.1, "learning_rate": 0.01, "weight_decay": 0.0001,
                  "max_epochs": 2, "patience": 2, "batch_size": 8, "gradient_clip": 1}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            model, record = training.train_seed(x, y, roles, config, 17, output)
            self.assertGreaterEqual(record["best_epoch"], 1)
            restored = training.SharpGRU(2, 4, 0.1)
            restored.load_state_dict(torch.load(output / "seed_17.pt", weights_only=True))
            np.testing.assert_array_equal(training.predict(model, x), training.predict(restored, x))


if __name__ == "__main__":
    unittest.main()
