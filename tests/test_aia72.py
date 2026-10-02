"""Scientific boundaries for AIA reuse in the distinct 72-hour experiment."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.aia72_data import AIA72Dataset, fit_normalization, validate_case_times
from scripts.aia_io import CHANNELS
from scripts.stage_aia72_canary import select_cases


class AIA72Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.records = {}
        for name, value in [("training", 1.), ("later", 100.)]:
            path = self.root / (name + ".npz")
            # Deliberately opposite to the manifest's 72-hour positive label.
            np.savez(path, x=np.full((512, 512, 6), value, np.float32),
                     channels=CHANNELS, sample_id=name, y=0, label_48h_final=0)
            self.records["gs://source/" + path.name] = {
                "path_relative_to_repository": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.frame = pd.DataFrame([self.row("train", "training"), self.row("retrospective_cycle25", "later")])

    def row(self, role, name):
        row = {"forecast_case_id": name, "issue_utc": "2020-01-01T06:00:00Z", "role": role, "label": 1}
        for lag, hour in [(288, "01"), (192, "02"), (96, "03")]:
            row[f"history_{lag}_UTC"] = f"2020-01-01T{hour}:00:00Z"
            row[f"history_uri_tminus{lag}"] = f"gs://source/{name}.npz"
        return row

    def test_normalization_excludes_later_rows(self):
        norm = fit_normalization(self.frame, self.records, self.root, pixels_per_file=8)
        self.assertEqual(norm["training_case_ids"], ["training"])
        self.assertEqual(norm["stats_uris"], ["gs://source/training.npz"])
        self.assertEqual(norm["channel_scale"], [1.] * 6)

    def test_source_label_is_never_used(self):
        norm = fit_normalization(self.frame, self.records, self.root, pixels_per_file=8)
        images, label, identity = AIA72Dataset(self.frame, self.records, self.root, norm, image_size=16)[0]
        self.assertEqual(tuple(images.shape), (3, 6, 16, 16))
        self.assertEqual(label.item(), 1.)
        self.assertEqual(identity, "training")

    def test_normalization_rejects_changed_source(self):
        self.records["gs://source/training.npz"]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "checksum"):
            fit_normalization(self.frame, self.records, self.root)

    def test_rejects_wrong_image_identity(self):
        norm = fit_normalization(self.frame, self.records, self.root, pixels_per_file=8)
        record = self.records["gs://source/later.npz"]
        record.update(self.records["gs://source/training.npz"])
        with self.assertRaisesRegex(ValueError, "identity"):
            AIA72Dataset(self.frame, self.records, self.root, norm)[1]

    def test_rejects_future_and_unordered_history(self):
        for col, value in [("history_96_UTC", "2020-01-01T07:00:00Z"),
                           ("history_288_UTC", "2020-01-01T02:30:00Z")]:
            frame = self.frame.copy()
            frame.loc[0, col] = value
            with self.assertRaises(ValueError):
                validate_case_times(frame)

    def test_case_selection_does_not_follow_outcomes(self):
        frame = pd.concat([self.frame.assign(forecast_case_id=self.frame.forecast_case_id + str(i),
                                            issue_utc=f"2020-01-0{i+1}T06:00:00Z") for i in range(3)])
        before = select_cases(frame).forecast_case_id.tolist()
        frame["label"] = [0, 1, 0, 1, 0, 1]
        self.assertEqual(before, select_cases(frame).forecast_case_id.tolist())


if __name__ == "__main__":
    unittest.main()
