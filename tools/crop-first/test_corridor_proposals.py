from copy import deepcopy
import json
import unittest
from unittest.mock import patch
from propose_crop_corridors import validate,build,RATIOS,SCHEMA,PROMPT,MINIMAL_PROMPT


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

    def test_catalogue_interval_preserves_product_not_only_faces(self):
        task,value=self.fixture();task.update(pair_protocol=3,purpose='catalogue')
        value.update(purpose='catalogue',product_inventory='localized',protected_regions=[dict(id='product',kind='featured_product',box=dict(left=30,top=30,right=65,bottom=55),description='featured product')])
        self.assertTrue(validate(self.response(value),task)['projected_ranges'][0]['ready_for_rendered_review'])
        value['protected_regions'][0]['box']['left']=5
        result=validate(self.response(value),task)['projected_ranges'][0]
        self.assertTrue(result['geometry']['geometrically_admissible'])
        self.assertFalse(result['catalogue_product_core_preserved']);self.assertFalse(result['ready_for_rendered_review'])
        value['protected_regions']=[]
        self.assertFalse(validate(self.response(value),task)['projected_ranges'][0]['ready_for_rendered_review'])

    def test_uncertain_product_inventory_stays_unadmitted(self):
        task,value=self.fixture();task.update(pair_protocol=3,purpose='hero')
        value.update(purpose='hero',product_inventory='uncertain',protected_regions=[])
        result=validate(self.response(value),task)['projected_ranges'][0]
        self.assertFalse(result['ready_for_rendered_review']);self.assertFalse(result['training_eligible'])

    @patch('propose_crop_corridors.read_image',return_value={'type':'input_image','image_url':'fixture'})
    def test_prior_request_bodies_remain_exact_and_schema_is_not_mutated(self,read_image):
        for version,prompt in [(1,PROMPT),(2,MINIMAL_PROMPT)]:
            task=dict(id='source',width=100,height=100,pair_protocol=version,role='fit',model='gpt-6.1-sol',image='fixture')
            facts=dict(source_id='source',original_width=100,original_height=100,ratios=RATIOS,purpose='editorial')
            if version==2:facts['pair_protocol']='maximum-area Minimal outer / tighter nested Maximal inner'
            expected=dict(model=task['model'],reasoning=dict(effort='medium'),store=False,tools=[],tool_choice='none',max_output_tokens=4096,
                input=[dict(role='developer',content=prompt),dict(role='user',content=[dict(type='input_text',text=json.dumps(facts)),read_image.return_value])],
                text=dict(format=dict(type='json_schema',name='excellent_nested_crop_ranges_v'+str(version),strict=True,schema=deepcopy(SCHEMA))))
            self.assertEqual(build(task),expected)
            before=deepcopy(SCHEMA)
            build(dict(task,pair_protocol=3,purpose='catalogue',subject_intent='whole featured product'))
            self.assertEqual(SCHEMA,before);self.assertEqual(build(task),expected)


if __name__=='__main__':unittest.main()
