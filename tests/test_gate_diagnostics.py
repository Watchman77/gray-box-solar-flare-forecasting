import unittest
import numpy as np
import pandas as pd
from io import StringIO
from scripts.diagnose_sharp_gates import subset_decisions, gate_attribution, outcome_categories, paired_guard_intervals
CFG={'guard_bits':{'support':1,'distance':2,'spread':4,'conflict':8},'bootstrap_repetitions':100,'bootstrap_seed':4}
class GateDiagnosticTests(unittest.TestCase):
    def test_all_subsets_preserve_original_labels_and_missing(self):
        base=np.array([1,0,1,0,np.nan]);bits=np.array([0,1,2,12,0])
        for active in range(16):
            out=subset_decisions(base,bits,active)
            for i in range(len(base)):
                if np.isfinite(base[i]) and not bits[i]&active:self.assertEqual(out[i],base[i])
                else:self.assertTrue(np.isnan(out[i]))
    def test_overlapping_attribution_reconciles(self):
        t=gate_attribution([1,0,1,0,1],[1,1,0,0,np.nan],[5,2,4,0,15],CFG).set_index('guard')
        self.assertEqual(t.allocated_removed.sum(),3)
        self.assertEqual(t.loc['support','allocated_true_alert'],.5)
        self.assertEqual(t.loc['spread','allocated_true_alert'],.5)
        self.assertEqual(t.loc['spread','unique_false_clear'],1)
        self.assertEqual(t.loc['support','unique_removed'],0)
        self.assertEqual(t.standalone_removed.sum(),4)
    def test_categories_keep_existing_abstentions_distinct(self):
        self.assertEqual(outcome_categories([1,0,1,0,1],[1,1,0,0,np.nan]).tolist(),['true_alert','false_alert','false_clear','true_clear','already_deferred'])
        with self.assertRaises(ValueError):outcome_categories([-1],[0])
    def test_gate_removal_only_recovers_unique_cases(self):
        base=[1,1,0,0];bits=[1,3,4,8]
        full=subset_decisions(base,bits,15);without=subset_decisions(base,bits,14)
        self.assertTrue(np.isnan(full).all());self.assertEqual(without[0],1);self.assertTrue(np.isnan(without[1:]).all())
    def test_round_trip_parser_preserves_inclusive_cutoff(self):
        threshold=379.55586300796773
        saved=pd.DataFrame({'distance':[threshold,np.nextafter(threshold,np.inf)]}).to_csv(index=False)
        values=pd.read_csv(StringIO(saved),float_precision='round_trip').distance
        self.assertEqual(values.gt(threshold).tolist(),[False,True])
    def test_invalid_inputs_rejected(self):
        for base,bits,active in [([2],[0],0),([1],[16],0),([np.inf],[0],0),([1],[0],16),([1,0],[1],1)]:
            with self.assertRaises(ValueError):subset_decisions(base,bits,active)
    def test_paired_identical_decisions_zero_difference(self):
        rows=paired_guard_intervals([1,0,1,0],[1,0,0,1],[1,0,0,1],['a','a','b','b'],CFG)
        for r in rows:self.assertEqual(r['difference'],0);self.assertEqual(r['low'],0);self.assertEqual(r['high'],0)
    def test_zero_issued_denominator_stays_undefined(self):
        rows=paired_guard_intervals([1,0],[np.nan,np.nan],[1,0],['a','b'],CFG)
        r=next(r for r in rows if r['metric']=='selective_error');self.assertIsNone(r['difference']);self.assertIsNone(r['low'])

if __name__=='__main__':unittest.main()
