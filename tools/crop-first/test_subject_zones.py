import copy
import json
import unittest

from subject_zones import RATIOS,build,validate


class SubjectZones(unittest.TestCase):
    def fixture(self):
        task=dict(id='fixture',purpose='hero')
        box=dict(left=0,top=0,right=100,bottom=100)
        roi=dict(id='person1',kind='person',priority='primary',box=box,description='subject',owner_subject_id=None)
        option=dict(box=box,retained_roi_ids=['person1'],reason='whole subject',selected_subject_ids=['person1'],excluded_subject_ids=[],face_policy_check='whole_faces_or_excluded')
        result=dict(source_id='fixture',purpose='hero',face_coverage='uncertain',rois=[roi],crops=[dict(ratio=r,mode=m,action='crop',options=[copy.deepcopy(option)],reason='proposal') for r in RATIOS for m in ['minimal','maximal']])
        return task,result

    def response(self,result):return dict(status='completed',output=[dict(type='message',content=[dict(type='output_text',text=json.dumps(result))])])

    def test_default_nine_ratios_round_trip_without_legacy_seven_ratio_fallback(self):
        task,result=self.fixture();self.assertEqual(validate(self.response(result),task),result)

    def test_selection_cannot_refer_to_unknown_subjects_or_simultaneously_exclude_them(self):
        for excluded in [['person1'],['absent']]:
            task,result=self.fixture();result['crops'][0]['options'][0]['excluded_subject_ids']=excluded
            with self.assertRaises(ValueError):validate(self.response(result),task)

    def test_face_owner_must_join_to_an_existing_subject(self):
        task,result=self.fixture();face=copy.deepcopy(result['rois'][0]);face.update(id='face1',kind='face_head',owner_subject_id='absent');result['rois'].append(face)
        with self.assertRaises(ValueError):validate(self.response(result),task)

    def test_test_role_is_rejected_before_reading_any_pixels(self):
        with self.assertRaisesRegex(ValueError,'TEST'):build(dict(role='test'))


if __name__=='__main__':unittest.main()
