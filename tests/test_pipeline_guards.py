"""Failure-path tests for dataset integrity, leakage and recoverable baseline runs."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.dataset_io import load_dataset, preflight_dataset, file_sha256
from scripts.run_exploratory_sharp import assign_roles, main, preprocess, transform
from scripts.check_aia_sample import check_frame, CHANNELS
from scripts.aia_io import load_aia_frame, WAVELENGTHS


class PipelineGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "dataset"
        self.root.mkdir()
        issue = pd.to_datetime(["2011-01-01T12:00Z", "2011-01-02T12:00Z", "2021-01-01T12:00Z"])
        index = pd.DataFrame({"forecast_case_id": ["a", "b", "c"], "source_sample_id": ["s1", "s2", "s3"],
            "region_component_id": ["r1", "r2", "r3"], "tensor_row": [0, 1, 2], "issue_utc": issue})
        for lag in [288, 192, 96]:
            index[f"history_uri_tminus{lag}"] = [f"gs://fixture/{name}-{lag}.npz" for name in index.forecast_case_id]
            index[f"history_{lag}_UTC"] = issue - pd.Timedelta(minutes=lag)
        master = index.copy(); master["inputs_available"] = True
        raw = np.arange(18, dtype=float).reshape(3, 3, 2)
        np.save(self.root / "sharp.npy", raw)
        np.save(self.root / "sharp_missing.npy", np.zeros_like(raw, dtype=bool))
        for h in (48, 72):
            master[f"outcome_end_utc_{h}h"] = issue + pd.Timedelta(hours=h)
            for scope in ("primary", "patch"):
                master[f"candidate_{scope}_label_{h}h"] = pd.Series([0, 1, None], dtype="Int64")
                np.save(self.root / f"y_{scope}_{h}h.npy", np.array([0, 1, -1], dtype=np.int8))
                np.save(self.root / f"known_{scope}_{h}h.npy", np.array([True, True, False]))
        index.to_csv(self.root / "input_index.csv.gz", index=False)
        master.to_csv(self.root / "master_cases.csv.gz", index=False)
        self.manifest = {"dataset_version": "synthetic_fixture", "target_version": "synthetic_fixture",
            "case_count": 3, "sharp_shape": [3, 3, 2], "sharp_dtype": "float64",
            "history_lag_native_minutes": [288, 192, 96], "feature_order": ["one", "two"],
            "operational_training_ready": False, "outputs": {}}
        self.pin()

    def pin(self):
        self.manifest["outputs"] = {p.name: {"sha256": file_sha256(p)} for p in self.root.iterdir() if p.name != "manifest.json"}
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def run_cli(self, output, *extra):
        with patch("sys.argv", ["runner", "--dataset", str(self.root), "--output-dir", str(output), "--fit", *extra]), redirect_stdout(io.StringIO()):
            main()

    def test_hash_verification_is_default(self):
        path = self.root / "sharp.npy"
        with path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_dataset(self.root)

    def test_actual_feature_missingness_must_match_mask(self):
        raw = np.load(self.root / "sharp.npy"); raw[0, 0, 0] = np.nan
        np.save(self.root / "sharp.npy", raw)
        with self.assertRaisesRegex(ValueError, "missingness mask"):
            load_dataset(self.root, verify=False)

    def test_master_array_disagreement_is_rejected_even_when_pinned(self):
        np.save(self.root / "y_primary_48h.npy", np.array([1, 1, -1], dtype=np.int8))
        self.pin()
        with self.assertRaisesRegex(ValueError, "Master/array outcome mismatch"):
            preflight_dataset(self.root)

    def test_unknown_outcome_cannot_be_marked_known(self):
        np.save(self.root / "known_primary_72h.npy", np.array([True, True, True]))
        with self.assertRaisesRegex(ValueError, "Label/mask"):
            load_dataset(self.root, verify=False)

    def test_future_history_is_rejected(self):
        p = self.root / "input_index.csv.gz"; index = pd.read_csv(p)
        index.loc[0, "history_96_UTC"] = "2030-01-01T00:00Z"
        index.to_csv(p, index=False); self.pin()
        with self.assertRaisesRegex(ValueError, "future predictor"):
            preflight_dataset(self.root)

    def test_confirmatory_use_of_provisional_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exploratory"):
            preflight_dataset(self.root, purpose="confirmatory")

    def test_roles_exclude_unmatured_unknown_and_unsupported_cases(self):
        issue = pd.to_datetime(pd.Series(["2019-12-30T00:00Z", "2011-01-01T00:00Z", "2021-01-01T00:00Z", "2026-01-01T00:00Z"]))
        end = (issue + pd.Timedelta(hours=72)).to_numpy()
        roles = assign_roles(issue, end, np.array([True, False, True, True]), np.array([True, True, False, True]))
        self.assertEqual(list(roles), ["unused", "unknown_label_excluded", "missing_feature_excluded", "supplementary_2026"])

    def test_saved_transform_is_batch_independent(self):
        raw = np.arange(24, dtype=float).reshape(4, 3, 2)
        fitted, parameters = preprocess(raw, np.array([True, True, False, False]))
        np.testing.assert_array_equal(transform(raw[3:4], parameters), fitted[3:4])

    def test_interrupted_fit_does_not_poison_output_and_retry_succeeds(self):
        output = Path(self.temp.name) / "run"
        with patch("scripts.run_exploratory_sharp.fit_logistic", side_effect=RuntimeError("injected stop")):
            with self.assertRaisesRegex(RuntimeError, "injected stop"):
                self.run_cli(output)
        self.assertFalse(output.exists())
        self.assertEqual(len(list(Path(self.temp.name).glob("run.incomplete-*/failure.json"))), 1)
        self.run_cli(output)
        self.assertTrue((output / "summary.json").exists())
        with patch("scripts.run_exploratory_sharp.fit_logistic", side_effect=AssertionError("must not refit")):
            self.run_cli(output, "--resume")

    def test_resume_refuses_changed_saved_result(self):
        output = Path(self.temp.name) / "run"
        self.run_cli(output)
        with (output / "model.npz").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Existing result changed"):
            self.run_cli(output, "--resume")

    def test_aia_channel_order_and_nonfinite_pixels_are_rejected(self):
        path = Path(self.temp.name) / "frame.npz"
        pixels = np.ones((2, 2, 6), dtype=np.float32)
        np.savez(path, x=pixels, channels=np.array(CHANNELS))
        self.assertTrue(check_frame(path, shape=(2, 2, 6))["finite"])
        np.savez(path, x=pixels, channels=np.array(CHANNELS[::-1]))
        with self.assertRaisesRegex(ValueError, "channel order"):
            check_frame(path, shape=(2, 2, 6))

    def test_aia_numeric_schema_preserves_pixels_and_rejects_conflicting_names(self):
        path = Path(self.temp.name) / "new-frame.npz"
        pixels = np.arange(24, dtype=np.float32).reshape(2, 2, 6)
        np.savez(path, x=pixels, wavelengths=np.array(WAVELENGTHS))
        x, metadata = load_aia_frame(path, expected_shape=(2, 2, 6))
        np.testing.assert_array_equal(x, pixels)
        self.assertEqual(metadata["metadata_schema"], "numeric_wavelengths")
        np.savez(path, x=pixels, wavelengths=np.array(WAVELENGTHS), channels=np.array(CHANNELS[::-1]))
        with self.assertRaisesRegex(ValueError, "channel order"):
            load_aia_frame(path, expected_shape=(2, 2, 6))
        pixels[0, 0, 0] = np.nan
        np.savez(path, x=pixels, channels=np.array(CHANNELS))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            check_frame(path, shape=(2, 2, 6))


if __name__ == "__main__":
    unittest.main()
