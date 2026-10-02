"""Focused package alignment and CPU baseline checks."""

import unittest

import numpy as np
import pandas as pd

from scripts.package_model_dataset import merge_labels
from scripts.run_exploratory_sharp import fit_logistic, preprocess, sigmoid


class DatasetPackageTests(unittest.TestCase):
    def test_join_uses_ids_and_keeps_unknown_labels(self):
        cases = pd.DataFrame({"forecast_case_id": ["a", "b"], "horizon_hours": [72, 72], "outcome_end_utc": ["x", "x"]})
        labels = pd.DataFrame([{"forecast_case_id": key, "horizon_hours": horizon, "outcome_end_utc": "end",
            "candidate_primary_label": value, "candidate_patch_label": value, "primary_event_ids": "",
            "patch_event_ids": "", "unassigned_region_mx_event_ids": "u" if value is None else "",
            "nominal_span_contains_window": True} for horizon in [48, 72] for key, value in [("b", None), ("a", 1)]])
        result = merge_labels(cases, labels)
        self.assertEqual(result.forecast_case_id.tolist(), ["a", "b"])
        self.assertEqual(result.iloc[0].candidate_primary_label_72h, 1)
        self.assertTrue(pd.isna(result.iloc[1].candidate_primary_label_72h))
        with self.assertRaises(ValueError):
            merge_labels(cases, labels.iloc[:-1])

    def test_preprocessing_uses_training_only(self):
        raw = np.array([[[1.0, np.nan]], [[3.0, 2.0]], [[1e12, 9e12]]])
        train = np.array([True, True, False])
        x, params = preprocess(raw, train)
        changed = raw.copy(); changed[-1] = -1e12
        x2, params2 = preprocess(changed, train)
        np.testing.assert_array_equal(x[train], x2[train])
        for key in params:
            np.testing.assert_array_equal(params[key], params2[key])
        self.assertTrue(np.isfinite(x).all())

    def test_logistic_constant_feature_recovers_natural_prevalence(self):
        beta, info = fit_logistic(np.zeros((10, 2)), np.array([1., 1.] + [0.]*8))
        self.assertTrue(info["converged"])
        self.assertAlmostEqual(float(sigmoid(beta[0])), 0.2, places=7)

    def test_logistic_learns_signal_with_finite_probabilities(self):
        x = np.arange(-5, 6, dtype=float).reshape(-1, 1)
        y = (x[:, 0] > 0).astype(float)
        beta, info = fit_logistic(x, y)
        self.assertTrue(info["converged"])
        self.assertGreater(beta[1], 0)
        self.assertTrue(np.isfinite(sigmoid(beta[0] + x @ beta[1:])).all())


if __name__ == "__main__":
    unittest.main()
