import unittest
from subject_geometry import audit,project


class Geometry(unittest.TestCase):
    def test_whole_neighbor_exclusion_and_false_teacher_claim(self):
        rois=[dict(id='a',kind='person',owner_subject_id=None,box=dict(left=0,top=0,right=40,bottom=100)),
            dict(id='fa',kind='face_head',owner_subject_id='a',box=dict(left=10,top=10,right=30,bottom=30)),
            dict(id='fb',kind='face_head',owner_subject_id='b',box=dict(left=70,top=10,right=90,bottom=30))]
        self.assertTrue(audit([0,0,60,100],rois,['a'],['b'],100,100)['geometry_compatible'])
        bad=audit([0,0,80,100],rois,['a'],['b'],100,100)
        self.assertEqual(bad['cut_faces'],['fb'])
        self.assertEqual(bad['excluded_faces_not_excluded'],['fb'])
        self.assertFalse(bad['geometry_compatible'])

    def test_missing_selected_face_is_not_accepted(self):
        roi=dict(id='a',kind='person',owner_subject_id=None,box=dict(left=0,top=0,right=100,bottom=100))
        self.assertFalse(audit([0,0,100,100],[roi],['a'],[],100,100)['geometry_compatible'])

    def test_integer_projection_maximum_area_and_tighter_extent(self):
        b=dict(left=20,top=35,right=98,bottom=93.5)
        self.assertEqual(project(b,720,960,'1:1','minimal'),[0,240,720,960])
        box=project(b,720,960,'1:1','maximal')
        self.assertEqual(box[2]-box[0],box[3]-box[1])
        self.assertLess(box[2]-box[0],720)


if __name__=='__main__':unittest.main()
