"""Scientific boundary checks using small synthetic inputs, not flare results."""

import unittest

import numpy as np
import pandas as pd

from scripts.audit_legacy_72h import FEATURES, build_cases, split_roles, summarize
from scripts.reconcile_legacy_predictions import discrimination


def fixture():
    return pd.DataFrame({
        "T_REC_dt": pd.date_range("2014-01-01", periods=8, freq="D").astype(str),
        "NOAA_AR": [12000] * 8,
        "label_MX_3d": [0, 0, 0, 0, 0, 0, 1, 0],
        **{name: np.ones(8) for name in FEATURES},
    })


class LegacyAuditTests(unittest.TestCase):
    def test_target_is_end_of_past_history_and_source_is_untouched(self):
        source = fixture()
        original = source.copy(deep=True)
        cases, profile = build_cases(source)
        self.assertEqual(cases.label_as_recorded.tolist(), [1, 0])
        self.assertEqual(cases.source_row_zero_based.tolist(), [6, 7])
        self.assertEqual(cases.history_span_hours.tolist(), [144, 144])
        self.assertEqual(profile["clean_rows"], 8)
        pd.testing.assert_frame_equal(source, original)

    def test_missing_feature_does_not_create_a_fabricated_observation(self):
        source = fixture()
        source.loc[3, "USFLUX"] = np.nan
        cases, profile = build_cases(source)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases.iloc[0].largest_history_gap_hours, 48)
        self.assertEqual(profile["rows_removed_by_legacy_complete_case_rule"], 1)

    def test_region_boundaries_are_not_crossed(self):
        source = fixture()
        source.loc[4:, "NOAA_AR"] = 12001
        with self.assertRaisesRegex(ValueError, "No complete"):
            build_cases(source)

    def test_fractional_label_is_not_silently_truncated(self):
        source = fixture()
        source["label_MX_3d"] = source.label_MX_3d.astype(float)
        source.loc[0, "label_MX_3d"] = 0.5
        with self.assertRaisesRegex(ValueError, "Non-binary"):
            build_cases(source)

    def test_historical_rounding_difference(self):
        self.assertEqual(int((split_roles(9438, "floor") == "test").sum()), 1417)
        self.assertEqual(int((split_roles(9438, "round") == "test").sum()), 1415)

    def test_future_outcome_maturity_and_shared_region_are_reported(self):
        # Ten artificial forecast cases, one per day, all from the same region.
        cases = pd.DataFrame({
            "issue_time_as_recorded": pd.date_range("2014-01-01", periods=10),
            "label_as_recorded": [0] * 9 + [1],
            "region_id_as_recorded": [12000] * 10,
        })
        result, _ = summarize(cases, "floor")
        boundary = result["boundaries"]["train_to_validation"]
        self.assertEqual(boundary["earlier_rows_with_72h_outcome_end_after_next_block_start"], 2)
        self.assertEqual(boundary["shared_region_ids"], [12000])

    def test_average_precision_groups_ties_and_is_not_trapezoidal_pr_area(self):
        values = discrimination([1, 0, 1], [0.9, 0.9, 0.1])
        self.assertAlmostEqual(values["average_precision"], 7 / 12)
        self.assertAlmostEqual(values["roc_auc"], 0.25)

    def test_constant_and_perfect_scores(self):
        self.assertEqual(discrimination([0, 1], [0.1, 0.9]), {"average_precision": 1.0, "roc_auc": 1.0})
        self.assertEqual(discrimination([0, 1], [0.5, 0.5]), {"average_precision": 0.5, "roc_auc": 0.5})

    def test_single_class_scores_are_not_given_a_fabricated_auc(self):
        with self.assertRaisesRegex(ValueError, "Both classes"):
            discrimination([0, 0], [0.1, 0.9])


if __name__ == "__main__":
    unittest.main()
