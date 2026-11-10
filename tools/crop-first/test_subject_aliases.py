import copy
import json
import unittest

from repair_subject_aliases import repair
import test_subject_zones as fixtures


class SubjectAliases(unittest.TestCase):
    def fixture(self):
        task,result=fixtures.SubjectZones().fixture();result['rois'][0]['id']='r1'
        face=copy.deepcopy(result['rois'][0]);face.update(id='face1',kind='face_head',owner_subject_id='p1');result['rois'].append(face)
        for crop in result['crops']:
            crop['options'][0].update(selected_subject_ids=['p1'],retained_roi_ids=['r1','face1'])
        response=fixtures.SubjectZones().response(result);return task,result,response

    def test_explicit_alias_changes_only_references_and_preserves_original(self):
        task,original,response=self.fixture();before=copy.deepcopy(response)
        result=repair(response,task,{'p1':'r1'})
        self.assertEqual(response,before);self.assertEqual(result['rois'][1]['owner_subject_id'],'r1')
        self.assertEqual([r['box'] for r in result['rois']],[r['box'] for r in original['rois']])
        self.assertEqual(result['crops'][0]['options'][0]['selected_subject_ids'],['r1'])

    def test_alias_cannot_overwrite_existing_ids_or_point_to_a_face(self):
        task,_,response=self.fixture()
        for mapping in [{'r1':'r1'},{'p1':'face1'},{'unused':'r1'}]:
            with self.assertRaises(ValueError):repair(response,task,mapping)


if __name__=='__main__':unittest.main()
