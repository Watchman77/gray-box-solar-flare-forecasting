import json
import unittest

import numpy as np
import pandas as pd

from scripts import conformal_sharp as cp


class ConformalTests(unittest.TestCase):
    def test_exact_finite_sample_rank_and_inclusive_ties(self):
        scores = np.arange(1, 10) / 10
        self.assertEqual(cp.finite_sample_threshold(scores, .1)['rank'], 9)
        self.assertEqual(cp.finite_sample_threshold(scores, .2)['rank'], 8)
        self.assertEqual(cp.finite_sample_threshold(scores, .2)['threshold'], .8)
        parameters = cp.fit_thresholds([.25]*9, [0]*9, 'pooled', .1)
        self.assertTrue(cp.predict_sets([.25], parameters)[0, 0])

    def test_small_or_missing_class_uses_infinite_threshold(self):
        parameters = cp.fit_thresholds([.1, .2], [0, 0], 'class_conditional', .05)
        restored = json.loads(json.dumps(parameters, allow_nan=False))
        self.assertTrue(cp.predict_sets([0, .5, 1], restored).all())
        self.assertTrue(restored['thresholds']['1']['include_all'])

    def test_pooled_leave_one_out_rank_coverage(self):
        # Exhaust all equally likely held-out ranks of a fixed exchangeable score bag.
        scores = np.linspace(.01, .99, 10)
        covered = []
        for i, score in enumerate(scores):
            q = cp.finite_sample_threshold(np.delete(scores, i), .2)
            covered.append(q['include_all'] or score <= q['threshold'])
        self.assertEqual(sum(covered), 8)

    def test_output_sets_and_metrics_keep_empty_distinct(self):
        q = lambda x: {'threshold': x, 'include_all': False}
        parameters = {'thresholds': {'0': q(.25), '1': q(.25)}}
        sets = cp.predict_sets([.1, .5, .9], parameters)
        np.testing.assert_array_equal(sets, [[True, False], [False, False], [False, True]])
        metrics = cp.set_metrics(np.array([0, 1, 1]), sets)
        self.assertEqual(metrics['covered'], 2)
        self.assertEqual(metrics['empty_count'], 1)
        self.assertEqual(metrics['flare_coverage'], .5)
        self.assertEqual(metrics['singleton_error'], 0)
        both = cp.predict_sets([.5], {'thresholds': {'0': q(.8), '1': q(.8)}})
        self.assertEqual(cp.set_metrics(np.array([1]), both)['both_count'], 1)

    def frame_and_config(self):
        issue = pd.to_datetime(['2015-01-02', '2015-02-02', '2015-03-02', '2015-04-02', '2016-01-01', '2025-01-01'], utc=True)
        frame = pd.DataFrame({'forecast_case_id': list('abcdef'), 'issue_utc': issue.astype(str),
            'outcome_end_utc_72h': (issue+pd.Timedelta(hours=72)).astype(str),
            'role': ['conformal_calibration']*4+['policy_validation', 'retrospective_cycle25'],
            'label': [0, 1, 0, -1, 1, 0], 'label_known': [True, True, True, False, True, True],
            'gru_selected': [.1, .5, .3, .9, .6, .2], 'region_component_id': list('aabbcc')})
        config = {'calibration_start': '2015-01-01', 'calibration_end': '2015-07-01', 'reporting_delay_hours': 24,
                  'models': ['gru'], 'methods': ['pooled', 'class_conditional'], 'alphas': [.1, .05]}
        return frame, config

    def test_policy_test_and_unknown_labels_do_not_fit_thresholds(self):
        frame, config = self.frame_and_config()
        frozen = cp.freeze_thresholds(frame, config)
        frame.loc[3:, 'gru_selected'] = [.01, .99, .99]
        frame.loc[4:, 'label'] = [0, 1]
        self.assertEqual(cp.freeze_thresholds(frame, config), frozen)
        self.assertEqual(frozen['support']['cases'], 3)

    def test_outcome_plus_reporting_boundary_is_enforced(self):
        frame, config = self.frame_and_config()
        frame.loc[2, 'outcome_end_utc_72h'] = '2015-07-01 00:00:00+00:00'
        with self.assertRaisesRegex(ValueError, 'boundary'):
            cp.freeze_thresholds(frame, config)

    def test_more_conservative_alpha_produces_nested_sets(self):
        rng = np.random.default_rng(4)
        p = rng.uniform(size=100)
        y = np.array([0]*80+[1]*20)
        for method in ['pooled', 'class_conditional']:
            small = cp.predict_sets(np.linspace(0, 1, 100), cp.fit_thresholds(p, y, method, .1))
            large = cp.predict_sets(np.linspace(0, 1, 100), cp.fit_thresholds(p, y, method, .05))
            self.assertTrue((~small | large).all())

    def test_bootstrap_denominators_and_undefined_class(self):
        y = np.zeros(8, dtype=int)
        sets = np.column_stack([np.ones(8, dtype=bool), np.zeros(8, dtype=bool)])
        rows = cp.coverage_bootstrap(y, sets, [0, 0, 1, 1, 2, 2, 3, 3], 100, 2)
        self.assertEqual(rows[0]['coverage'], 1)
        self.assertEqual(rows[0]['low'], 1)
        self.assertEqual(rows[0]['high'], 1)
        self.assertIsNone(rows[2]['coverage'])
        self.assertIsNone(rows[2]['low'])
        with self.assertRaises(ValueError):
            cp.fit_thresholds([.1], [-1], 'pooled', .1)


if __name__ == '__main__':
    unittest.main()
