"""Region adjudication must not turn ambiguous associations into certain labels."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.adjudicate_swpc_regions import adjudicate, parse_report, required_dates
from scripts.build_adjudicated_outcomes import apply_decisions
from scripts.build_candidate_outcomes import label_cases


def report_line(bin="1000", start="1200", peak="1210", end="1220", kind="XRA",
                region="2929", quality="5", particulars="M1.5    3.0E-02", selected=True):
    chars = [" "] * 80
    for offset, value in [(0, bin), (5, "+" if selected else " "), (10, start.rjust(5)),
                          (17, peak.rjust(5)), (27, end.rjust(5)), (34, "G16"),
                          (39, quality), (43, kind), (48, "1-8A"),
                          (58, particulars), (76, region)]:
        chars[offset:offset + len(value)] = value
    return "".join(chars)


class RegionAdjudicationTests(unittest.TestCase):
    def parse(self, *lines, date="20220118"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{date}events.txt"
            path.write_text(f":Date: {date[:4]} {date[4:6]} {date[6:]}\n" + "\n".join(lines))
            return parse_report(path)

    def queue(self, original=None, refs="12929", peak="2022-01-18T12:10:00Z", multiplicity=1):
        return pd.DataFrame([{"event_id": "event1", "start_utc": "2022-01-18T12:00:00Z",
            "peak_utc": peak, "science_class": "M2.1", "original_noaa_region": original,
            "exact_peak_known_regions": refs, "science_peak_multiplicity": multiplicity,
            "reference_status": "missing" if original is None else "conflict"}])

    def decide(self, reports, **kwargs):
        queue = self.queue(**kwargs)
        return adjudicate(queue, reports, set(required_dates(queue))).iloc[0]

    def test_direct_missing_fill_preserves_science_class(self):
        row = self.decide(self.parse(report_line(particulars="C9.8    3.0E-02")))
        self.assertEqual(row.status, "filled_missing_direct_xra")
        self.assertEqual(row.candidate_region, 12929)
        self.assertEqual(row.science_class, "M2.1")
        self.assertFalse(row.historical_mx_class_present)

    def test_conflicting_existing_region_is_quarantined_not_overwritten(self):
        row = self.decide(self.parse(report_line()), original=12930)
        self.assertEqual(row.status, "quarantined_science_daily_conflict")
        self.assertTrue(pd.isna(row.candidate_region))
        self.assertEqual(row.original_region, 12930)

    def test_daily_support_can_retain_original_despite_reference_disagreement(self):
        row = self.decide(self.parse(report_line()), original=12929, refs="12929;12930")
        self.assertEqual(row.status, "retained_original_daily_supported")

    def test_multiple_bins_never_choose_nearest_or_majority(self):
        row = self.decide(self.parse(report_line(), report_line(bin="2000")))
        self.assertEqual(row.status, "unresolved_multiple_swpc_groups")

    def test_group_region_conflict_blocks_even_selected_xra(self):
        row = self.decide(self.parse(report_line(), report_line(kind="FLA", region="2930", particulars="SF")))
        self.assertEqual(row.status, "unresolved_conflicting_swpc_regions")

    def test_optical_requires_overlap_and_adequate_quality(self):
        xra = report_line(region="")
        good = report_line(kind="FLA", region="2929", quality="3", particulars="SF")
        row = self.decide(self.parse(xra, good))
        self.assertEqual(row.status, "filled_missing_optical_event_bin")
        for optical in [report_line(kind="FLA", start="1230", peak="1240", end="1250", particulars="SF"),
                        report_line(kind="FLA", quality="2", particulars="SF"),
                        report_line(kind="FLA", start="A1200", particulars="SF")]:
            self.assertEqual(self.decide(self.parse(xra, optical)).status, "unresolved_no_qualified_region_evidence")

    def test_event_bin_is_scoped_to_utc_date(self):
        reports = pd.concat([self.parse(report_line()), self.parse(report_line(region="2930"), date="20220117")])
        self.assertEqual(self.decide(reports).status, "filled_missing_direct_xra")

    def test_nonunique_science_peak_and_reference_conflict_block_fill(self):
        reports = self.parse(report_line())
        self.assertEqual(self.decide(reports, multiplicity=2).status, "unresolved_nonunique_science_peak")
        self.assertEqual(self.decide(reports, refs="12930").status, "unresolved_reference_daily_conflict")

    def test_uncertain_or_nearby_peak_is_not_exact_match(self):
        for peak in ["U1210", "1209"]:
            self.assertEqual(self.decide(self.parse(report_line(peak=peak))).status, "unresolved_no_exact_xra_peak")

    def test_missing_source_prevents_acceptance(self):
        row = adjudicate(self.queue(), self.parse(report_line()), {"20220118"}).iloc[0]
        self.assertEqual(row.status, "unresolved_missing_daily_source")

    def test_empty_report_and_region_field_do_not_invent_values(self):
        self.assertTrue(self.parse().empty)
        row = self.parse(report_line(region="")).iloc[0]
        self.assertTrue(pd.isna(row.region))
        self.assertEqual(self.decide(self.parse()).status, "unresolved_no_exact_xra_peak")

    def test_clock_rollover_and_invalid_clock(self):
        row = self.parse(report_line(start="2350", peak="0010", end="0020")).iloc[0]
        self.assertEqual(row.peak_utc, pd.Timestamp("2022-01-19T00:10Z"))
        bad = self.parse(report_line(peak="2460")).iloc[0]
        self.assertTrue(bad.source_time_problem)
        self.assertFalse(bad.exact_start_peak)

    def test_expanded_archived_cme_keeps_full_region_for_conflict_check(self):
        row = self.parse("6900 +    B1912   ////     A0106  SOH  2   CME  XUV,EUV,UV347-346/FS198       11158").iloc[0]
        self.assertEqual(row.region, 11158)
        self.assertFalse(row.exact_start_peak)

    def test_quarantine_masks_label_without_mutating_original_source(self):
        science = pd.DataFrame([{"event_id": "event1", "noaa_full_id_candidate": 12930,
            "is_mx": True, "start_time": pd.Timestamp("2022-01-18T12:00Z")}])
        decision = self.decide(self.parse(report_line()), original=12930)
        result = apply_decisions(science, pd.DataFrame([decision]))
        self.assertEqual(science.iloc[0].noaa_full_id_candidate, 12930)
        self.assertEqual(result.iloc[0].original_noaa_full_id, 12930)
        self.assertTrue(pd.isna(result.iloc[0].noaa_full_id_candidate))
        cases = pd.DataFrame([{"forecast_case_id": "case", "NOAA_ARS": "12930", "NOAA_AR_clean": 12930,
            "issue_utc": "2022-01-18T10:00Z"}])
        labels = label_cases(cases, result, pd.Series(pd.to_datetime(["2022-01-20T10:00Z"], utc=True)),
            pd.Timestamp("2022-01-01T00:00Z"), pd.Timestamp("2022-02-01T00:00Z"))
        self.assertTrue(pd.isna(labels.iloc[0].candidate_primary_label))
        self.assertEqual(labels.iloc[0].unassigned_region_mx_event_ids, "event1")

    def test_builder_rejects_duplicate_or_mismatched_decisions(self):
        science = pd.DataFrame([{"event_id": "event1", "noaa_full_id_candidate": None}])
        decision = pd.DataFrame([self.decide(self.parse(report_line()))])
        with self.assertRaises(ValueError):
            apply_decisions(science, pd.concat([decision, decision]))
        decision["original_region"] = 12930
        with self.assertRaisesRegex(ValueError, "original region"):
            apply_decisions(science, decision)


if __name__ == "__main__":
    unittest.main()
