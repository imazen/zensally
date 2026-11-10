import unittest
from refine_nested_origin import refine
from crop_corridors import audit,domains
from subject_policy import face_relation


class NestedRefinementTests(unittest.TestCase):
    def test_enlarges_inner_to_prevent_crossing_a_face_between_safe_endpoints(self):
        outer=[0,0,8,8];inner=[4,4,8,8];heads=[[1,1,2,2],[6,6,7,7]]
        self.assertFalse(audit(inner,outer,12,8,'1:1',heads)['geometrically_admissible'])
        actual=refine(12,8,outer,inner,'1:1',heads);self.assertEqual(actual['inner'],[1,1,7,7]);self.assertTrue(actual['inner_enlarged'])
        for row in domains(actual['inner'],actual['outer'],'1:1'):
            for x in range(row['x'][0],row['x'][1]+1):
                for y in range(row['y'][0],row['y'][1]+1):
                    crop=[x,y,x+row['width'],y+row['height']];self.assertEqual([face_relation(crop,h) for h in heads],['contained','contained'])

    def test_minimal_remains_available_when_no_strict_safe_child_exists(self):
        actual=refine(12,8,[0,0,8,8],[2,2,6,6],'1:1',[[0,3,8,5]])
        self.assertEqual(actual['outer'],[0,0,8,8]);self.assertIsNone(actual['inner']);self.assertEqual(actual['status'],'no-strict-child-retaining-outer-heads')

    def test_outer_translation_preserves_extent_and_whole_person_exclusion(self):
        actual=refine(20,8,[0,0,8,8],[2,2,6,6],'1:1',[[7,2,10,4],[17,2,19,4]])
        self.assertEqual(actual['outer'],[2,0,10,8]);self.assertTrue(audit(actual['inner'],actual['outer'],20,8,'1:1',[[7,2,10,4],[17,2,19,4]])['geometrically_admissible'])

    def test_unchecked_and_infeasible_outputs_are_explicit(self):
        unchecked=refine(12,8,[0,0,8,8],[2,2,6,6],'1:1',[]);self.assertFalse(unchecked['geometry_checked'])
        impossible=refine(12,8,[0,0,8,8],[2,2,6,6],'1:1',[[0,0,12,8]]);self.assertIsNone(impossible['outer']);self.assertIsNone(impossible['inner'])


if __name__=='__main__':unittest.main()
