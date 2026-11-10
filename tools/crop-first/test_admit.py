import unittest
from collections import Counter,defaultdict
from admit import admitted_source

class AdmissionTests(unittest.TestCase):
    def fixture(self):
        p={'id':'11000','family_id':'1000','bucket':'train','use':'fit','historical_eval_family':False,'width':600,'height':600,'valid_rect':[0,0,1,1],'tensor_sha256':'fixture','sha256':'fixture','png_sha256':'fixture','directory':'fixture','rois':[],
           'tasks':[{'task_id':'one','ratio':'1:3','mode':'minimal','candidates':[{'id':'A','xywh':[0,0,200,600],'image':{},'missed_proposed_faces':[]},{'id':'B','xywh':[200,0,200,600],'image':{},'missed_proposed_faces':[]}]}]}
        label={'task_id':'one','preferred':'A','candidates':[{'candidate_id':'A','acceptable':True,'face_cut':'no','confidence':'high'},{'candidate_id':'B','acceptable':False,'face_cut':'no','confidence':'high'}]}
        return p,label
    def test_sol_disagreement_removes_label_without_reversing_it(self):
        p,t=self.fixture();s={**t,'preferred':None,'candidates':[{**c,'acceptable':False} for c in t['candidates']]};a=defaultdict(Counter)
        result=admitted_source(p,{'one':t},{'one':s},a)
        self.assertEqual(result['candidates'][0]['weight'],0);self.assertTrue(result['candidates'][0]['acceptable']);self.assertEqual(result['pairs'],[]);self.assertEqual(result['face_context'][5],0)
    def test_canonical_validate_cannot_be_admitted_as_fit(self):
        p,t=self.fixture();p['bucket']='validate'
        with self.assertRaises(ValueError):admitted_source(p,{'one':t},{},defaultdict(Counter))
    def test_unknown_face_coverage_never_sets_reviewed_flag(self):
        p,t=self.fixture();t['candidates'][0]['face_cut']='uncertain';r=admitted_source(p,{'one':t},{},defaultdict(Counter))
        self.assertEqual(r['candidates'][0]['weight'],0);self.assertEqual(r['face_context'][5],0);self.assertFalse(r['face_safety_certified'])
    def test_reject_all_keeps_negatives_without_a_rank_pair(self):
        p,t=self.fixture();t['preferred']=None
        for c in t['candidates']:c['acceptable']=False
        r=admitted_source(p,{'one':t},{},defaultdict(Counter));self.assertEqual(r['pairs'],[]);self.assertEqual(sum(c['weight'] for c in r['candidates']),1.)
if __name__=='__main__':unittest.main()
