import copy
import unittest
from cached_crop_eval import evaluate

class CachedCropEvalTests(unittest.TestCase):
    def fixture(self):
        sources=[];predictions=[]
        for i,(family,accepted,weight,action) in enumerate([('a',True,1.,'crop'),('a',False,1.,'crop'),('b',True,1.,'contain'),('b',False,0.,'crop')]):
            c={'task_id':str(i)+'-1x1-minimal','candidate_id':'C00','ratio':'1:1','mode':0,
               'box':[0,0,1,1],'xywh':[0,0,100,100],'acceptable':accepted,'weight':weight}
            sources.append({'id':str(i),'family_id':family,'bucket':'validate','use':'eval','face_context':[0,0,0,0,0,0],'candidates':[c]})
            predictions.append({'source_id':str(i),'task_id':c['task_id'],'candidate_id':'C00' if action=='crop' else None,'xywh':c['xywh'],'research_action':action,'teacher_acceptable':True})
        return {'supervision':'noisy_vlm_rendered_pixels','sources':sources},predictions

    def test_cached_truth_unknowns_and_family_bootstrap(self):
        d,p=self.fixture();r=evaluate(d,p,'eval',resamples=200)
        self.assertEqual(r['overall']['counts']['judged_suggestions'],2)
        self.assertEqual(r['overall']['counts']['accepted_suggestions'],1)
        self.assertEqual(r['overall']['counts']['unjudged_suggestions'],1)
        self.assertEqual(r['overall']['counts']['avoidable_contain_tasks'],1)
        self.assertEqual(r['overall']['task_rates']['acceptance_precision'],.5)
        self.assertEqual(r['overall']['task_rates']['crop_coverage'],.75)
        self.assertEqual(r['overall']['source_families'],2)
        self.assertLess(r['overall']['family_bootstrap_95_percentile']['acceptance_precision']['defined_resamples'],200)
        self.assertEqual(r,evaluate(d,p,'eval',resamples=200))

    def test_off_bank_pixels_and_missing_selections_are_errors(self):
        d,p=self.fixture();q=copy.deepcopy(p);q[0]['xywh']=[1,0,99,100]
        with self.assertRaisesRegex(ValueError,'off-bank geometry'):evaluate(d,q,'eval')
        with self.assertRaisesRegex(ValueError,'every frozen task'):evaluate(d,p[:-1],'eval')
        q=copy.deepcopy(p);q[0]['candidate_id']='new'
        with self.assertRaisesRegex(ValueError,'off-bank selection'):evaluate(d,q,'eval')

    def test_face_veto_is_independent_of_positive_teacher_label(self):
        d,p=self.fixture();d['sources'][0]['face_context']=[1,.4,.1,.6,.4,0]
        c=d['sources'][0]['candidates'][0];c['box']=[0,0,.5,1];c['xywh']=[0,0,50,100];p[0]['xywh']=c['xywh']
        r=evaluate(d,p,'eval',resamples=20)
        self.assertEqual(r['overall']['counts']['proposed_face_box_violations'],1)
        self.assertEqual(r['overall']['counts']['accepted_suggestions'],1)
        self.assertEqual(r['overall']['counts']['safe_accepted_suggestions'],0)

if __name__=='__main__':unittest.main()
