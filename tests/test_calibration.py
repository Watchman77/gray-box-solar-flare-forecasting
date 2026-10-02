import json
import unittest

import numpy as np
import pandas as pd

try:
    from scripts import calibrate_sharp as calibration
except ModuleNotFoundError as exc:
    if exc.name not in ("scipy", "sklearn"):
        raise
    calibration = None


@unittest.skipIf(calibration is None, "Install requirements-notebook.txt for calibration tests")
class CalibrationTests(unittest.TestCase):
    def config(self):
        return {"calibration_start": "2014-07-01", "inner_selection_start": "2014-12-01", "calibration_end": "2015-01-01",
                "reporting_delay_hours": 24, "candidate_methods": ["raw", "platt", "isotonic"],
                "logit_clip_epsilon": 1e-6, "log_loss_clip_epsilon": 1e-12, "reliability_bins": 10}

    def frame(self):
        issue = pd.to_datetime(["2014-07-01", "2014-08-01", "2014-10-01", "2014-11-01",
                                "2014-12-01", "2014-12-05", "2014-12-10", "2014-12-15",
                                "2025-01-01", "2025-01-02"], utc=True)
        return pd.DataFrame({"issue_utc": issue.astype(str), "outcome_end_utc_72h": (issue+pd.Timedelta(hours=72)).astype(str),
                             "role": ["probability_calibration"]*8+["retrospective_cycle25"]*2,
                             "label": [0, 1, 1, 0, 0, 1, 0, 1, 0, 1], "label_known": True,
                             "probability_gru_mean": [.1, .3, .2, .4, .15, .2, .35, .4, .9, .8],
                             "probability_logistic": [.15, .25, .4, .3, .1, .35, .3, .4, .1, .2]})

    def test_later_outcomes_cannot_change_selection_or_fitted_parameters(self):
        original = self.frame()
        first = calibration.select_and_refit(original, self.config(), .2)
        changed = original.copy()
        changed.loc[8:, "label"] = [1, 0]
        changed.loc[8:, "probability_gru_mean"] = [.01, .99]
        self.assertEqual(first, calibration.select_and_refit(changed, self.config(), .2))

    def test_reporting_window_and_unknown_label_exclusion(self):
        frame = self.frame()
        frame.loc[3, "issue_utc"] = "2014-11-29 00:00:00+00:00"
        frame.loc[3, "outcome_end_utc_72h"] = "2014-12-02 00:00:00+00:00"
        frame.loc[6, ["label", "label_known"]] = [-1, False]
        eligible, fit, selection = calibration.calibration_masks(frame, self.config())
        self.assertTrue(eligible[3])
        self.assertFalse(fit[3])
        self.assertFalse(selection[3])
        self.assertFalse(eligible[6])
        self.assertFalse(eligible[8:].any())

    def test_isotonic_serialization_monotonicity_and_clipped_extrapolation(self):
        parameters = calibration.fit_calibrator(np.array([.2, .4, .6, .8]), np.array([0, 1, 0, 1]), "isotonic")
        restored = json.loads(json.dumps(parameters))
        prediction = calibration.apply_calibrator(np.linspace(0, 1, 21), restored)
        self.assertTrue((np.diff(prediction) >= 0).all())
        self.assertEqual(prediction[0], parameters["y"][0])
        self.assertEqual(prediction[-1], parameters["y"][-1])

    def test_probability_endpoints_and_empty_bins(self):
        table = calibration.reliability_table(np.array([0, 1]), np.array([0., 1.]))
        self.assertEqual(table["count"].sum(), 2)
        self.assertEqual(table.iloc[0]["count"], 1)
        self.assertEqual(table.iloc[-1]["count"], 1)
        self.assertTrue(table.iloc[1:9].observed_fraction.isna().all())
        score = calibration.score_probabilities(np.array([0, 1]), np.array([0., 1.]), .5, self.config(), diagnostics=False)
        self.assertEqual(score["brier"], 0)
        self.assertEqual(score["ece_equal_width_10"], 0)
        with self.assertRaises(ValueError):
            calibration.fit_calibrator(np.array([.2, .4]), np.array([0, -1]), "platt")

    def test_paired_cluster_bootstrap_preserves_constant_difference(self):
        result = calibration.paired_brier_bootstrap(np.zeros(8), np.full(8, np.sqrt(.1)), np.zeros(8),
                                                   np.array([1, 1, 2, 3, 3, 3, 4, 4]), 50, 2)
        self.assertAlmostEqual(result["difference"], .1)
        self.assertAlmostEqual(result["low"], .1)
        self.assertAlmostEqual(result["high"], .1)


if __name__ == "__main__":
    unittest.main()
