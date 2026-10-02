import unittest
import numpy as np
import pandas as pd
from scripts import rolling_conformal_sharp as rc
from scripts.conformal_sharp import fit_thresholds


class RollingConformalTests(unittest.TestCase):
    def config(self):
        return {'history_start':'2015-01-01','replay_start':'2015-07-01','policy_start':'2015-07-01','policy_end':'2020-01-01',
                'reporting_delay_hours':24,'horizon':72,'minimum_windows_per_class':2,'minimum_region_components_per_class':2,
                'models':['gru'],'alphas':[.1,.05],'primary_alpha':.1,'lookback_days':[90],
                'candidate_methods':['fixed_2015','rolling_90d'],'selection':'test-only rule',
                'bootstrap_seed':3,'bootstrap_repetitions':100}

    def history(self):
        issue=pd.to_datetime(['2015-06-01','2015-06-02','2015-06-26','2015-06-27','2015-06-28','2015-07-01'],utc=True)
        return pd.DataFrame({'forecast_case_id':list('abcdef'),'issue_utc':issue.astype(str),
            'outcome_end_utc_72h':(issue+pd.Timedelta(hours=72)).astype(str),'label':[0,1,0,1,1,0],
            'label_known':True,'region_component_id':list('abcdef'),'gru_selected':[.1,.4,.2,.5,.01,.9]})

    def test_window_boundary_and_maturity(self):
        history=rc.prepare_history(self.history(),self.config())
        block=rc.eligible_history(history,pd.Timestamp('2015-07-01',tz='UTC'),30)
        self.assertEqual(list(block.forecast_case_id),list('abcd'))
        self.assertEqual(block.iloc[-1].ready_ns,pd.Timestamp('2015-07-01',tz='UTC').value)
        block=rc.eligible_history(history,pd.Timestamp('2015-07-01',tz='UTC'),29)
        self.assertEqual(list(block.forecast_case_id),list('bcd'))

    def test_unmatured_labels_and_scores_cannot_change_current_threshold(self):
        original=self.history(); changed=original.copy()
        changed.loc[4:,['label','gru_selected']]=[[0,.99],[1,.01]]
        cfg=self.config(); date=pd.Timestamp('2015-07-01',tz='UTC')
        a=rc.daily_thresholds(rc.eligible_history(rc.prepare_history(original,cfg),date,90),cfg,'gru',.1)
        b=rc.daily_thresholds(rc.eligible_history(rc.prepare_history(changed,cfg),date,90),cfg,'gru',.1)
        self.assertEqual(a,b)

    def test_unknown_and_pre_fit_history_are_excluded(self):
        frame=self.history();frame.loc[0,'label_known']=False;frame.loc[0,'label']=-1
        frame.loc[1,'issue_utc']='2014-12-31 00:00:00+00:00'
        h=rc.prepare_history(frame,self.config())
        self.assertNotIn('a',set(h.forecast_case_id));self.assertNotIn('b',set(h.forecast_case_id))

    def test_region_support_guard_includes_unsupported_class(self):
        cfg=self.config();cfg['minimum_windows_per_class']=1
        block=rc.prepare_history(self.history(),cfg)
        block.loc[block.label.eq(1),'region_component_id']='one_component'
        parameters,support=rc.daily_thresholds(block,cfg,'gru',.5)
        self.assertTrue(support['1']['guarded'])
        self.assertTrue(rc.predict_sets([0.,.2,1.],parameters)[:,1].all())

    def test_cold_start_both_classes_included(self):
        cfg=self.config();block=rc.prepare_history(self.history(),cfg).iloc[:0]
        parameters,support=rc.daily_thresholds(block,cfg,'gru',.1)
        self.assertTrue(rc.predict_sets([.01,.99],parameters).all())
        self.assertEqual(support['1']['windows'],0)

    def test_same_day_issues_share_threshold_and_missing_inputs_stay_missing(self):
        cfg=self.config();frame=self.history();frame['role']='conformal_calibration'
        population=pd.DataFrame({'forecast_case_id':['x','y','z'],'issue_utc':['2015-07-01 01:00:00+00:00','2015-07-01 20:00:00+00:00','2015-07-01 23:00:00+00:00'],
            'inputs_available':[True,True,False],'candidate_primary_label_72h':[0,1,-1],
            'label_known_primary_72h':[True,True,False],'gru_selected':[.25,.25,np.nan]})
        fixed={'models':{'gru':{'class_conditional':{str(a):fit_thresholds([.1,.4],[0,1],'class_conditional',a) for a in cfg['alphas']}}}}
        replay,journal=rc.generate_replay(frame,population,fixed,cfg)
        col=rc.code_column('gru','rolling_90d',.1)
        self.assertEqual(replay.loc[0,col],replay.loc[1,col]);self.assertTrue(pd.isna(replay.loc[2,col]))
        self.assertEqual(journal.update_utc.nunique(),1)
        self.assertEqual(journal.issued_cases.unique().tolist(),[2])

    def test_later_outcomes_cannot_change_method_selection(self):
        cfg=self.config()
        frame=pd.DataFrame({'role':['policy_validation']*4+['retrospective_cycle25'],
            'issue_utc':['2016-01-01']*4+['2025-01-01'],'label_known_primary_72h':True,
            'candidate_primary_label_72h':[0,0,1,1,1],
            rc.code_column('gru','fixed_2015',.1):[1,1,1,1,1],
            rc.code_column('gru','rolling_90d',.1):[1,1,3,3,3]})
        a=rc.select_on_policy(frame,cfg)
        frame.loc[4,'candidate_primary_label_72h']=0
        frame.loc[4,rc.code_column('gru','rolling_90d',.1)]=0
        self.assertEqual(a,rc.select_on_policy(frame,cfg))
        self.assertEqual(a['selected_methods']['gru'],'rolling_90d')

    def test_paired_bootstrap_identical_methods_have_zero_difference(self):
        y=np.array([0,1,0,1]); sets=rc.decode_sets([1,3,1,2])
        rows=rc.paired_set_bootstrap(y,sets,sets,['a','b','c','d'],self.config())
        for row in rows:
            self.assertEqual(row['difference'],0)
            self.assertEqual(row['low'],0);self.assertEqual(row['high'],0)
        with self.assertRaises(ValueError):rc.decode_sets([1,np.nan])


if __name__=='__main__':unittest.main()
