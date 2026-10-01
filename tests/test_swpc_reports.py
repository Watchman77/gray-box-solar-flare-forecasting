"""Direct-report checks distinguish exact evidence from uncertain clock strings."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.audit_swpc_label_disagreements import parse_xra


class SwpcReportTests(unittest.TestCase):
    def parse(self, rows, date="2022 01 18"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "20220118events.txt"
            path.write_text(f":Product: 20220118events.txt\n:Date: {date}\n" + "\n".join(rows))
            return parse_xra(path)

    def test_selected_and_unselected_reports_preserve_regions_and_clocks(self):
        rows = self.parse(["2430 + 1701 1744 1817 G16 5 XRA 1-8A M1.5 3.4E-02 2929",
                           "2431 1800 1805 1810 G16 5 XRA 1-8A C9.9 1.0E-02 2929"])
        self.assertEqual(rows.noaa_region.tolist(), [12929, 12929])
        self.assertEqual(rows.is_mx.tolist(), [True, False])
        self.assertEqual(rows.iloc[0].start_utc, pd.Timestamp("2022-01-18T17:01:00Z"))

    def test_missing_region_is_not_taken_from_integrated_flux(self):
        rows = self.parse(["2430 + 1701 1744 1817 G16 5 XRA 1-8A M1.5 3.4E-02"])
        self.assertTrue(pd.isna(rows.iloc[0].noaa_region))

    def test_uncertain_clock_is_not_exact_positive_evidence(self):
        rows = self.parse(["2430 + A1701 1744 1817 G16 5 XRA 1-8A M1.5 3.4E-02 2929"])
        self.assertFalse(rows.iloc[0].exact_clocks)
        self.assertTrue(pd.isna(rows.iloc[0].start_utc))

    def test_report_midnight_rollover_uses_documented_forward_day(self):
        rows = self.parse(["2430 + 2350 0010 0030 G16 5 XRA 1-8A M1.5 3.4E-02 2929"])
        self.assertEqual(rows.iloc[0].peak_utc, pd.Timestamp("2022-01-19T00:10:00Z"))

    def test_wrong_source_day_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "date does not match"):
            self.parse([], date="2022 01 19")


if __name__ == "__main__":
    unittest.main()
