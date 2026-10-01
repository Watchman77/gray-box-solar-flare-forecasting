"""Independent scientific boundary examples for the candidate dataset builder."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.build_dataset_inventory import require_unique, tai_to_utc
from scripts.build_candidate_outcomes import label_cases, normalize_events, parse_regions


LEAPS = """# File expires on 28 June 2027
54832.0 1 1 2009 34
56109.0 1 7 2012 35
57204.0 1 7 2015 36
57754.0 1 1 2017 37
"""


def raw_events(starts, regions, classes=None):
    return pd.DataFrame({
        "flare_id": [str(201401010000 + i) for i in range(len(starts))],
        "time": starts, "start_time": starts, "end_time": [None] * len(starts),
        "active_region": regions, "peak_saturated": [0] * len(starts),
        "flare_class": classes or ["M1.0"] * len(starts),
    })


def cases():
    return pd.DataFrame({"forecast_case_id": ["a", "b"],
        "issue_utc": ["2014-01-01T00:00:00Z"] * 2,
        "NOAA_AR_clean": [12000, 12002], "NOAA_ARS": ["12000,12001", "12002"],
        # Neither absent fusion inputs nor an upstream role may drop a case.
        "upstream_matched_three_slot_inputs": [True, False]})


class DatasetConstructionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.leaps = Path(self.temp.name) / "Leap_Second.dat"
        self.leaps.write_text(LEAPS)

    def test_native_tai_offsets_change_at_leap_epochs(self):
        values = pd.Series(["2010.01.01_00:00:34_TAI", "2012.07.01_00:00:35_TAI",
            "2015.07.01_00:00:36_TAI", "2017.01.01_00:00:37_TAI"])
        actual = tai_to_utc(values, self.leaps)
        expected = pd.Series(pd.to_datetime(["2010-01-01", "2012-07-01", "2015-07-01", "2017-01-01"], utc=True))
        pd.testing.assert_series_equal(actual, expected)

    def test_physical_72_hours_across_leap_is_not_72_posix_hours(self):
        times = tai_to_utc(pd.Series(["2016.12.30_00:00:36_TAI", "2017.01.02_00:00:36_TAI"]), self.leaps)
        self.assertEqual(times.iloc[0], pd.Timestamp("2016-12-30T00:00:00Z"))
        self.assertEqual(times.iloc[1], pd.Timestamp("2017-01-01T23:59:59Z"))
        self.assertEqual((times.iloc[1] - times.iloc[0]).total_seconds(), 72 * 3600 - 1)

    def test_unrepresentable_leap_second_and_expired_table_fail_closed(self):
        for value, message in [("2017.01.01_00:00:36_TAI", "inserted leap"),
                               ("2027.07.01_00:00:37_TAI", "outside")]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                tai_to_utc(pd.Series([value]), self.leaps)

    def run_labels(self, events, end="2014-01-04T00:00:00Z", nominal_end="2014-02-01T00:00:00Z"):
        return label_cases(cases(), normalize_events(events), pd.Series([end, end]),
            pd.Timestamp("2014-01-01T00:00:00Z"), pd.Timestamp(nominal_end))

    def test_window_excludes_issue_includes_end_and_keeps_unmatched_inputs(self):
        events = raw_events(["2014-01-01T00:00:00Z", "2014-01-04T00:00:00Z",
                             "2014-01-04T00:00:01Z"], [2000, 2000, 2000])
        result = self.run_labels(events)
        self.assertEqual(result.forecast_case_id.tolist(), ["a", "b"])
        self.assertEqual(result.primary_event_count.tolist(), [1, 0])
        self.assertEqual(result.candidate_primary_label.tolist(), [1, 0])
        self.assertEqual(result.iloc[0].primary_event_ids, "noaa-science-v1-0-1:201401010001")

    def test_multi_noaa_patch_does_not_silently_replace_primary_target(self):
        result = self.run_labels(raw_events(["2014-01-02T00:00:00Z"], [2001]))
        self.assertEqual(result.candidate_primary_label.tolist(), [0, 0])
        self.assertEqual(result.candidate_patch_label.tolist(), [1, 0])
        self.assertEqual(result.association_convention_changes_recorded_event_presence.tolist(), [True, False])

    def test_unknown_region_masks_negatives_but_preserves_recorded_positive(self):
        result = self.run_labels(raw_events(["2014-01-02T00:00:00Z"] * 2, [2000, np.nan]))
        self.assertEqual(result.iloc[0].candidate_primary_label, 1)
        self.assertTrue(pd.isna(result.iloc[1].candidate_primary_label))
        self.assertEqual(result.unassigned_region_mx_count.tolist(), [1, 1])

    def test_lower_class_unknown_event_does_not_mask_mx_target(self):
        result = self.run_labels(raw_events(["2014-01-02T00:00:00Z"], [np.nan], ["C9.9"]))
        self.assertEqual(result.candidate_primary_label.tolist(), [0, 0])

    def test_incomplete_followup_is_not_a_negative(self):
        result = self.run_labels(raw_events(["2014-01-02T00:00:00Z"], [2000]), nominal_end="2014-01-03T00:00:00Z")
        self.assertTrue(result.candidate_primary_label.isna().all())
        self.assertTrue(result.nominal_span_contains_window.eq(False).all())

    def test_invalid_region_epoch_is_not_modulo_matched(self):
        for value, primary in [("20000", 20000), ("12000,?", 12000), ("12001", 12000)]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_regions(value, primary)
        self.assertEqual(parse_regions("12001,12000,12001", 12000), (12000, 12001))

    def test_event_source_is_preserved_missing_end_is_allowed_and_duplicates_fail(self):
        raw = raw_events(["2014-01-02T00:00:00Z"], [2000])
        original = raw.copy(deep=True)
        events = normalize_events(raw)
        pd.testing.assert_frame_equal(raw, original)
        self.assertTrue(events.end_time.isna().all())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_events(pd.concat([raw, raw]))

    def test_missing_and_duplicate_case_keys_fail_instead_of_expanding_join(self):
        for values in [["a", "a"], ["a", None]]:
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "Missing or duplicate"):
                require_unique(pd.DataFrame({"case": values}), ["case"], "synthetic")


if __name__ == "__main__":
    unittest.main()
