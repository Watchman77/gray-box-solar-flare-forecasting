"""Source ambiguity must not silently become a corrected scientific label."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.reconcile_event_sources import parse_annual, presence_by_primary, reconcile, validate_references


def science(region=None):
    return pd.DataFrame({"event_id": ["science:one"], "start_time": ["2017-04-01T10:00:00Z"],
        "time": ["2017-04-01T10:10:00Z"], "end_time": ["2017-04-01T10:20:00Z"],
        "flare_class": ["M1.0"], "is_mx": [True], "noaa_full_id_candidate": [region]})


def references(regions=(12644,), start="2017-04-01T10:00:00Z"):
    return validate_references(pd.DataFrame({"reference_id": [f"ref:{i}" for i in range(len(regions))],
        "source_family": ["HER_annual"] * len(regions), "start_utc": [start] * len(regions),
        "peak_utc": ["2017-04-01T10:10:00Z"] * len(regions), "end_utc": ["2017-04-01T10:20:00Z"] * len(regions),
        "flare_class": ["M1.0"] * len(regions), "noaa_region": list(regions),
        "clock_rollover_assumed": False, "source_time_problem": False}))


class EventReconciliationTests(unittest.TestCase):
    def test_mixed_region_encoding_preserves_recorded_value(self):
        refs = references((2644,))
        self.assertEqual(refs.iloc[0].noaa_region, 12644)
        self.assertEqual(refs.iloc[0].noaa_region_recorded, 2644)
        self.assertTrue(refs.iloc[0].source_region_was_four_digit_suffix)

    def test_region_normalization_rejects_an_unsupported_historical_epoch(self):
        refs = references()
        for column in ["start_utc", "peak_utc", "end_utc"]:
            refs[column] = refs[column] - pd.DateOffset(years=20)
        with self.assertRaisesRegex(ValueError, "normalization epoch"):
            validate_references(refs)

    def test_exact_unique_match_proposes_region_without_modifying_source(self):
        source = science()
        before = source.copy(deep=True)
        result = reconcile(source, references()).iloc[0]
        self.assertEqual(result.proposed_region_fill, 12644)
        self.assertEqual(result.reference_status, "missing_region_exact_start_peak_candidate")
        pd.testing.assert_frame_equal(source, before)

    def test_peak_only_match_stays_a_review_candidate(self):
        result = reconcile(science(), references(start="2017-04-01T09:58:00Z")).iloc[0]
        self.assertEqual(result.candidate_region_from_exact_peak, 12644)
        self.assertIsNone(result.proposed_region_fill)

    def test_conflicting_regions_never_use_majority_vote(self):
        result = reconcile(science(), references((12644, 12644, 12645))).iloc[0]
        self.assertIsNone(result.proposed_region_fill)
        self.assertEqual(result.reference_status, "missing_region_conflicting_references")

    def test_existing_science_region_is_preserved_when_references_disagree(self):
        result = reconcile(science(12645), references()).iloc[0]
        self.assertEqual(result.original_noaa_region, 12645)
        self.assertIsNone(result.proposed_region_fill)
        self.assertEqual(result.reference_status, "existing_region_reference_conflict")

    def test_nonunique_peak_across_all_science_classes_prevents_fill(self):
        source = pd.concat([science(), science().assign(event_id="science:two", is_mx=False, flare_class="C9.0")])
        result = reconcile(source, references()).iloc[0]
        self.assertIsNone(result.proposed_region_fill)
        self.assertEqual(result.science_peak_multiplicity, 2)

    def test_assumed_rollovers_and_time_problems_are_not_matching_evidence(self):
        for column in ["clock_rollover_assumed", "source_time_problem"]:
            refs = references()
            refs[column] = True
            result = reconcile(science(), refs).iloc[0]
            self.assertIsNone(result.proposed_region_fill)
            self.assertEqual(result.exact_peak_reference_ids, "")

    def test_annual_rollover_and_invalid_order_are_retained_with_flags(self):
        text = "3 events were retrieved from HER\n\nColumns\n" + "\n".join([
            "1-Apr-2017 23:58 00:02 00:05 M1.0 N10W10 12644",
            "2-Apr-2017 10:00 10:20 10:10 M2.0 N10W10 12644",
            "3-Apr-2017 09:00 09:05 09:10 C1.0 N10W10"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annual.txt"
            path.write_text(text)
            result = parse_annual(path)
        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[0].peak_utc, pd.Timestamp("2017-04-02T00:02:00Z"))
        self.assertTrue(result.iloc[0].clock_rollover_assumed)
        self.assertTrue(result.iloc[1].source_time_problem)
        self.assertTrue(pd.isna(result.iloc[2].noaa_region))

    def test_wrong_annual_row_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annual.txt"
            path.write_text("2 events were retrieved\n1-Apr-2017 10:00 10:10 10:20 M1.0 12644\n")
            with self.assertRaisesRegex(ValueError, "row-count mismatch"):
                parse_annual(path)

    def test_positive_evidence_excludes_issue_and_includes_endpoint(self):
        cases = pd.DataFrame({"NOAA_AR_clean": [12644, 12644, 12645]})
        starts = pd.Series(pd.to_datetime(["2017-04-01T10:00:00Z", "2017-04-01T09:00:00Z", "2017-04-01T09:00:00Z"]))
        ends = pd.Series(pd.to_datetime(["2017-04-01T11:00:00Z", "2017-04-01T10:00:00Z", "2017-04-01T10:00:00Z"]))
        self.assertEqual(presence_by_primary(cases, references(), starts, ends).tolist(), [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
