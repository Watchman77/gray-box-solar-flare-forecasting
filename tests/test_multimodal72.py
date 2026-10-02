"""Scientific boundary tests for the cross-project prediction interface."""

import unittest

import numpy as np
import pandas as pd

from scripts.prepare_multimodal72 import id_hash, validate_prediction_import


class TestMultimodalImport(unittest.TestCase):
    def setUp(self):
        self.cases = pd.DataFrame({"forecast_case_id": ["a", "b"],
                                   "issue_utc": ["2021-01-01T00:00:00Z"] * 2,
                                   "candidate_sharp_aia_comparison": [True, True]})
        self.contract = {"horizon_hours": 72, "target_scope": "primary", "target_version": "v2",
                         "dataset_manifest_sha256": "data", "split_manifest_sha256": "split",
                         "training_case_ids_sha256": "train", "branches": ["aia", "goes", "sharp_aia_goes"]}
        self.metadata = {**{k: v for k, v in self.contract.items() if k != "branches"},
                         "branch": "aia", "probability_kind": "raw", "fit_roles": ["train"],
                         "selection_roles": ["model_validation"], "historical_availability": "unverified_retrospective"}
        self.predictions = pd.DataFrame({"forecast_case_id": ["b", "a"], "probability": [np.nan, .4],
                                         "input_status": ["missing_input", "ok"],
                                         "last_observation_utc": [None, "2020-12-31T23:00:00Z"]})

    def check(self, predictions=None, metadata=None):
        return validate_prediction_import(self.predictions if predictions is None else predictions,
                                          self.metadata if metadata is None else metadata, self.contract, self.cases)

    def test_align_by_identity_and_retain_failures(self):
        result = self.check()
        self.assertEqual((result["cases"], result["issued"], result["input_failures"]), (2, 1, 1))
        self.assertIn("not_scientific_acceptance", result["status"])

    def test_reject_48h_wrong_labels_or_splits(self):
        for key, value in [("horizon_hours", 48), ("target_version", "old"),
                           ("split_manifest_sha256", "other"), ("training_case_ids_sha256", "other")]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.check(metadata={**self.metadata, key: value})

    def test_reject_future_observation(self):
        f = self.predictions.copy()
        f.loc[f.input_status.eq("ok"), "last_observation_utc"] = "2021-01-01T00:00:01Z"
        with self.assertRaisesRegex(ValueError, "future"):
            self.check(f)

    def test_require_explicit_missing_cases(self):
        with self.assertRaisesRegex(ValueError, "every requested case"):
            self.check(self.predictions.iloc[1:].copy())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check(pd.concat([self.predictions, self.predictions.iloc[:1]]))

    def test_reject_fabricated_probability_for_missing_input(self):
        f = self.predictions.copy()
        f.loc[f.input_status.eq("missing_input"), "probability"] = .1
        with self.assertRaisesRegex(ValueError, "Probability/status"):
            self.check(f)

    def test_reject_nonfinite_or_out_of_range_probability(self):
        for p in [np.nan, np.inf, -.01, 1.01]:
            f = self.predictions.copy()
            f.loc[f.input_status.eq("ok"), "probability"] = p
            with self.subTest(p=p), self.assertRaisesRegex(ValueError, "Probability/status"):
                self.check(f)

    def test_reject_later_model_or_combiner_fitting(self):
        with self.assertRaisesRegex(ValueError, "roles"):
            self.check(metadata={**self.metadata, "fit_roles": ["train", "retrospective_cycle25"]})
        with self.assertRaisesRegex(ValueError, "Combiner"):
            self.check(metadata={**self.metadata, "branch": "sharp_aia_goes", "combiner_fit_roles": ["supplementary_2026"]})

    def test_asof_claim_needs_evidence_and_causal_delivery(self):
        m = {**self.metadata, "historical_availability": "verified_as_of"}
        with self.assertRaisesRegex(ValueError, "requires"):
            self.check(metadata=m)
        m["availability_evidence_sha256"] = "evidence"
        f = self.predictions.copy()
        f["last_available_utc"] = [None, "2021-01-01T00:01:00Z"]
        with self.assertRaisesRegex(ValueError, "availability"):
            self.check(f, m)
        f["last_available_utc"] = [None, "2020-12-31T23:01:00Z"]
        self.assertEqual(self.check(f, m)["issued"], 1)

    def test_case_set_digest_does_not_depend_on_order(self):
        self.assertEqual(id_hash(["a", "b"]), id_hash(["b", "a"]))
        self.assertNotEqual(id_hash(["a", "b"]), id_hash(["a", "c"]))
        with self.assertRaises(ValueError):
            id_hash(["a", "a"])


if __name__ == "__main__":
    unittest.main()
