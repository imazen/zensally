import json
import unittest
from label_quality import build,validate


class LabelQuality(unittest.TestCase):
    def test_test_sources_are_rejected_before_reading_images(self):
        with self.assertRaisesRegex(ValueError,'closed source'):build(dict(role='test'))

    def test_well_formed_semantic_contradictions_are_preserved_as_evidence(self):
        task=dict(case_id='case',candidates=[dict(id=i) for i in 'ABCD'])
        label=dict(case_id='case',source_story='one person',face_inventory='visible_faces_listed',
            faces=[dict(id='f',box=dict(left=10,top=10,right=40,bottom=40),visibility='clear',description='face')],
            grades=[dict(candidate_id=i,new_face_truncation='no',cut_face_ids=[],subject_coherence='yes',acceptable='yes',composition_quality='usable',reason='usable') for i in 'ABCD'],preferred='A')
        label['grades'][0].update(new_face_truncation='yes',cut_face_ids=['f'])
        response=dict(status='completed',output=[dict(content=[dict(type='output_text',text=json.dumps(label))])])
        result=validate(response,task);self.assertFalse(result['training_eligible']);self.assertEqual(len(result['semantic_inconsistencies']),1)
        label['grades'][0]['cut_face_ids']=['unknown'];response['output'][0]['content'][0]['text']=json.dumps(label)
        with self.assertRaisesRegex(ValueError,'unknown/duplicate'):validate(response,task)


if __name__=='__main__':unittest.main()
