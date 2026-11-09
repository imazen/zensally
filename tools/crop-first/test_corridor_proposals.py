from copy import deepcopy
import json
import unittest
from propose_crop_corridors import validate,build,RATIOS


class CorridorProposals(unittest.TestCase):
    def fixture(self):
        task=dict(id='source',width=100,height=100,pair_protocol=2)
        pair=dict(id='p',outer=dict(left=0,top=0,right=100,bottom=75),inner=dict(left=25,top=25,right=75,bottom=62.5),subject_story='same subject',outer_excellent=True,inner_excellent=True,reason='range')
        value=dict(source_id='source',face_inventory='visible_faces_listed',faces=[dict(id='f',box=dict(left=30,top=30,right=40,bottom=40),description='face')],ranges=[dict(ratio=r,pairs=[pair] if r=='4:3' else [],reason='range') for r in RATIOS])
        return task,value

    def response(self,value):return dict(status='completed',output=[dict(content=[dict(type='output_text',text=json.dumps(value))])])

    def test_maximum_area_outer_is_required_and_cannot_be_silently_enlarged(self):
        task,value=self.fixture();result=validate(self.response(value),task)['projected_ranges'][0]
        self.assertTrue(result['outer_is_maximum_area']);self.assertTrue(result['ready_for_rendered_review'])
        value['ranges'][0]['pairs'][0]['outer']=dict(left=10,top=10,right=90,bottom=70)
        result=validate(self.response(value),task)['projected_ranges'][0]
        self.assertTrue(result['geometry']['geometrically_admissible']);self.assertFalse(result['outer_is_maximum_area']);self.assertFalse(result['ready_for_rendered_review'])
        self.assertEqual(result['outer'],[10,10,90,70])

    def test_a_face_between_endpoints_invalidates_interval_even_with_minimal_outer(self):
        task,value=self.fixture();value['faces'][0]['box']=dict(left=5,top=5,right=15,bottom=15)
        result=validate(self.response(value),task)['projected_ranges'][0]
        self.assertTrue(result['outer_is_maximum_area']);self.assertFalse(result['ready_for_rendered_review'])

    def test_closed_roles_rejected_before_image_read(self):
        with self.assertRaises(ValueError):build(dict(role='test',model='gpt-6.1-sol'))


if __name__=='__main__':unittest.main()
