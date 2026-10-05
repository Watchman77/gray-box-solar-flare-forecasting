"""Protect the earlier-data boundary and matched fusion calibration."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from scripts.calibrate_multimodal72 import fit_matched_calibration


class MatchedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] / "configs/sharp72_calibration_v1.json").read_text())
        times = list(pd.date_range("2014-08-01", periods=30, freq="D", tz="UTC")) + list(pd.date_range("2014-12-01", periods=20, freq="D", tz="UTC"))
        y = np.arange(50) % 2
        sharp = .2 + .5 * y + .02 * np.cos(np.arange(50))
        aia = .3 + .25 * y + .04 * np.sin(np.arange(50))
        self.frame = pd.DataFrame({
            "forecast_case_id": [f"case-{i}" for i in range(50)], "role": "probability_calibration",
            "issue_utc": [str(v) for v in times], "outcome_end_utc_72h": [str(v + pd.Timedelta(hours=72)) for v in times],
            "label": y, "label_known": True, "sharp_raw": sharp, "aia_raw": aia,
            "sharp_aia_raw": .5 * sharp + .5 * aia, "aia_input_status": "ok", "aia_input_failure_reason": "",
        })

    def test_all_branches_use_identical_earlier_support_and_fixed_weights(self):
        result = fit_matched_calibration(self.frame, self.config, .2)
        self.assertEqual(set(result["selected_methods"]), {"sharp", "aia", "sharp_aia"})
        self.assertEqual(result["matched_support"]["inner_selection"]["cases"], 20)
        self.assertEqual({r["cases"] for r in result["inner_scores"]}, {20})
        self.assertEqual(result["fusion_weights"], {"sharp": .5, "aia": .5})
        self.assertEqual(result["evaluation_roles_read"], [])
        self.assertEqual(len(result["inner_scores"]), 9)

    def test_input_failure_is_excluded_from_every_fit_but_retained_in_denominator(self):
        frame = self.frame.copy()
        frame.loc[0, ["aia_raw", "sharp_aia_raw"]] = np.nan
        frame.loc[0, "aia_input_status"] = "missing_input"
        frame.loc[0, "aia_input_failure_reason"] = "source_not_found"
        result = fit_matched_calibration(frame, self.config, .2)
        self.assertEqual(result["requested_cases"], 50)
        self.assertEqual(result["input_failure_cases"], 1)
        self.assertEqual(result["matched_support"]["final_fit"]["cases"], 49)
        self.assertEqual(len(frame), 50)

    def test_future_role_and_late_maturity_rejected(self):
        for column, value in [("role", "retrospective_cycle25"), ("outcome_end_utc_72h", "2015-01-01 00:00:00+00:00")]:
            with self.subTest(column=column):
                frame = self.frame.copy()
                frame.loc[0, column] = value
                with self.assertRaises(ValueError):
                    fit_matched_calibration(frame, self.config, .2)

    def test_invalid_status_probabilities_or_fusion_are_rejected(self):
        for column, value in [("aia_input_status", "missing_input"), ("aia_raw", np.nan), ("sharp_aia_raw", .9), ("sharp_raw", 1.1)]:
            with self.subTest(column=column):
                frame = self.frame.copy()
                frame.loc[0, column] = value
                with self.assertRaises(ValueError):
                    fit_matched_calibration(frame, self.config, .2)

    def test_duplicate_case_and_missing_class_rejected(self):
        frame = self.frame.copy()
        frame.loc[1, "forecast_case_id"] = frame.loc[0, "forecast_case_id"]
        with self.assertRaises(ValueError):
            fit_matched_calibration(frame, self.config, .2)
        frame = self.frame.copy()
        frame.loc[30:, "label"] = 0
        with self.assertRaises(ValueError):
            fit_matched_calibration(frame, self.config, .2)


if __name__ == "__main__":
    unittest.main()
