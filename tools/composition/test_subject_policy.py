import unittest

from subject_policy import admissible,assess,face_relation


class SubjectChoice(unittest.TestCase):
    faces=[[10,20,30,50],[70,20,90,50]]

    def test_choose_one_face_and_exclude_the_other_wholly(self):
        self.assertTrue(admissible([0,0,50,100],self.faces,[0],complete_faces_reviewed=True))
        self.assertEqual(assess([0,0,50,100],self.faces,[0])['relations'],['contained','excluded'])

    def test_selected_person_does_not_license_slicing_a_neighbor(self):
        self.assertFalse(admissible([0,0,80,100],self.faces,[0],complete_faces_reviewed=True))

    def test_whole_group_is_allowed_when_it_fits(self):
        self.assertTrue(admissible([0,0,100,100],self.faces,[0,1],complete_faces_reviewed=True))

    def test_selected_face_cannot_be_replaced_by_another(self):
        self.assertFalse(admissible([50,0,100,100],self.faces,[0],complete_faces_reviewed=True))

    def test_unknown_coverage_remains_closed(self):
        self.assertFalse(admissible([0,0,50,100],self.faces,[0]))

    def test_touching_boundary_excludes_but_sliver_intersection_cuts(self):
        self.assertEqual(face_relation([0,0,70,100],self.faces[1]),'excluded')
        self.assertEqual(face_relation([0,0,70.001,100],self.faces[1]),'cut')

    def test_crop_inside_one_large_face_is_still_a_cut(self):
        self.assertEqual(face_relation([20,25,25,30],self.faces[0]),'cut')

    def test_invalid_or_empty_subject_selection_is_rejected(self):
        for selected in [[],[2],[0,0]]:
            with self.assertRaises(ValueError):assess([0,0,50,100],self.faces,selected)


if __name__=='__main__':unittest.main()
