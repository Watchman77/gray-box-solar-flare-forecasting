import unittest
import numpy as np
import pandas as pd
from scripts.decision_policy_sharp import decide_states, decision_metrics, gate_threshold, decision_intervals

class DecisionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame({'gru_selected':[.2]*8,'logistic_selected':[.3]*8,
            'feature_distance_squared':[2.]*8,'seed_spread':[.02]*8,'input_contract_ok':[True]*8})
        self.policy={'support':True,'conflict':True,'fallback':True}
        self.thresholds={'distance':10.,'spread':.05}
    def decide(self,g,l,gg=None,lg=None):
        return decide_states(self.frame,g,l,np.zeros(8) if gg is None else gg,np.zeros(8) if lg is None else lg,self.policy,self.thresholds)
    def test_three_states_and_reason_precedence(self):
        self.frame.loc[0,'input_contract_ok']=False
        self.frame.loc[3,'feature_distance_squared']=11
        o=self.decide([1,np.nan,1,1,1,0,1,3],[1,1,1,1,2,1,1,2],gg=[0,0,2,0,0,0,0,0])
        self.assertEqual(o.reason.tolist(),['missing_or_invalid_input','no_issued_set','insufficient_calibration_support','feature_distance_flag','conflicting_singletons','empty_gru_set','checks_passed','logistic_fallback_candidate'])
        self.assertEqual(o.state.tolist(),['abstain']*6+['normal','degraded'])
        self.assertEqual(o.decision.iloc[6:].tolist(),[0.,1.])
        self.assertTrue(o.issued_probability.iloc[:6].isna().all())
    def test_shared_feature_failure_never_falls_back(self):
        self.frame.feature_distance_squared=20
        o=self.decide([3]*8,[2]*8)
        self.assertTrue(o.state.eq('abstain').all())
    def test_disagreement_fallback_requires_supported_nonconflicting_singleton(self):
        self.frame.seed_spread=.1
        o=self.decide([1,1,1,3,0,3,3,1],[1,2,1,2,1,3,2,1],lg=[0,0,2,0,0,0,2,0])
        self.assertEqual(o.state.tolist(),['degraded','abstain','abstain','degraded','abstain','abstain','abstain','degraded'])
    def test_boundary_is_inclusive_and_unknown_outcome_irrelevant(self):
        self.frame.feature_distance_squared=10.;self.frame.seed_spread=.05
        self.frame['candidate_primary_label_72h']=[-1,1,0,-1,1,0,1,0]
        first=self.decide([1]*8,[1]*8)
        self.frame['candidate_primary_label_72h']=1-self.frame.candidate_primary_label_72h
        second=self.decide([1]*8,[1]*8)
        pd.testing.assert_frame_equal(first,second)
        self.assertTrue(first.state.eq('normal').all())
    def test_missing_set_is_distinct_from_empty_set(self):
        o=self.decide([np.nan,0,3,1,1,1,1,1],[1]*8)
        self.assertEqual(o.reason.iloc[:3].tolist(),['no_issued_set','empty_gru_set','logistic_fallback_candidate'])
    def test_metrics_do_not_count_abstentions_as_correct_or_negative(self):
        m=decision_metrics([1,1,1,0,0,0],[1,0,np.nan,1,0,np.nan])
        self.assertEqual(m['cases'],6);self.assertEqual(m['issued'],4)
        self.assertEqual(m['selective_error'],.5)
        self.assertEqual(m['deferred_flares'],1)
        self.assertEqual(m['false_clear_fraction_of_all_flares'],1/3)
        self.assertAlmostEqual(sum(m[k] for k in ['alert_fraction_of_all_flares','false_clear_fraction_of_all_flares','deferred_fraction_of_all_flares']),1)
        empty=decision_metrics([1,0],[np.nan,np.nan]);self.assertIsNone(empty['selective_error'])
        with self.assertRaises(ValueError):decision_metrics([-1,0],[0,0])
    def test_reference_quantile_and_bad_values(self):
        self.assertEqual(gate_threshold([1,2,3,4],.5),3)
        self.assertIsNone(gate_threshold([1,2],None))
        for values,q in [([], .9),([1,np.nan],.9),([1,2],0),([1,2],1.1)]:
            with self.assertRaises(ValueError):gate_threshold(values,q)
    def test_bootstrap_unavailable_denominator(self):
        rows=decision_intervals([1,0,1,0],[np.nan]*4,['a','b','a','b'],{'bootstrap_repetitions':100,'bootstrap_seed':3})
        risk=next(r for r in rows if r['metric']=='selective_error')
        self.assertIsNone(risk['low']);self.assertEqual(risk['valid_repetitions'],0)
    def test_invalid_probability_or_code_cannot_issue(self):
        self.frame.loc[0,'gru_selected']=1.1
        self.assertEqual(self.decide([1]*8,[1]*8).state.iloc[0],'abstain')
        with self.assertRaises(ValueError):self.decide([4]*8,[1]*8)

if __name__=='__main__':unittest.main()
