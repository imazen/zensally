import json,tempfile,unittest
from pathlib import Path
import teacher


class TeacherContractTests(unittest.TestCase):
    def test_held_out_source_is_rejected_before_reading_images(self):
        for role in ['validate','test','evaluation']:
            with self.assertRaises(ValueError):teacher.build_request({'source_role':role},purpose='training-weak-label')

    def test_incomplete_refused_and_malformed_judgments_are_errors(self):
        for response in [{'status':'incomplete'}, {'status':'completed','output':[{'content':[{'type':'refusal','refusal':'no'}]}]}, {'status':'completed','output':[{'content':[{'type':'output_text','text':'{}'}]}]}]:
            with self.assertRaises(ValueError):teacher.parse_label(response)

    def test_ties_remain_ties_and_never_certify_face_safety(self):
        label={'preferred':'tie','a_acceptable':True,'b_acceptable':True,'face_cut_a':'uncertain','face_cut_b':'no','reason':'Both compositions are plausible.'}
        response={'status':'completed','output':[{'content':[{'type':'output_text','text':json.dumps(label)}]}]}
        self.assertEqual(teacher.parse_label(response),label)

    def test_swapping_preserves_candidate_identity_and_does_not_admit_label(self):
        # Source bytes are not decoded or rewritten by this request-builder test.
        scratch=Path.home()/'tmp'
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            import hashlib
            items=[]
            for i in range(3):
                p=Path(tmp)/f'{i}.png';data=b'\x89PNG\r\n\x1a\n'+bytes([i]);p.write_bytes(data)
                items.append({'id':str(i),'path':str(p),'sha256':hashlib.sha256(data).hexdigest()})
            task={'task_id':'probe','source_id':'fixture','source_role':'evaluation','mode':'minimal','target_ratio':[9,16],'source':items[0],'candidates':items[1:]}
            a,ma=teacher.build_request(task,purpose='protocol-probe')
            b,mb=teacher.build_request(task,purpose='protocol-probe',swap=True)
            self.assertEqual(ma['display_map'],{'A':'1','B':'2'})
            self.assertEqual(mb['display_map'],{'A':'2','B':'1'})
            self.assertEqual(a['input'][1]['content'][1],b['input'][1]['content'][1])
            self.assertFalse(ma['face_safety_certified'] or ma['training_eligible'])
            self.assertEqual(ma['admission'],'unreviewed')

    def test_pool_selection_requires_complete_known_candidate_assessments(self):
        label={'preferred':'C02','acceptable':['C02'],'face_cut':{'C01':'no','C02':'uncertain','C03':'no'},'reason':'Keep subject context'}
        self.assertEqual(teacher.parse_selection(label,['C01','C02','C03']),label)
        for change in [{'preferred':'C99'},{'face_cut':{'C02':'no'}},{'acceptable':['C02','C02']},{'face_cut':{'C01':'no','C02':'yes','C03':'no'}}]:
            with self.assertRaises(ValueError):teacher.parse_selection({**label,**change},['C01','C02','C03'])

    def test_pool_can_reject_every_candidate(self):
        label={'preferred':None,'acceptable':[],'face_cut':{'C01':'uncertain','C02':'yes','C03':'no'},'reason':'No good composition'}
        self.assertIsNone(teacher.parse_selection(label,['C01','C02','C03'])['preferred'])


if __name__=='__main__':unittest.main()
