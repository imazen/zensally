import unittest
from subject_candidate_bank import generate

class ModeBanks(unittest.TestCase):
    def test_all_rectangles_are_bounded_native_exact_aspect(self):
        for size in [(1024,683),(257,129),(129,257),(720,960)]:
            for ratio in [(1,1),(3,1),(1,3),(6,1),(1,6),(9,16)]:
                for mode in ['minimal','maximal']:
                    boxes=generate(*size,ratio,mode)
                    self.assertLessEqual(len(boxes),128);self.assertEqual(len(boxes),len(set(boxes)))
                    for l,t,r,b in boxes:
                        self.assertTrue(0<=l<r<=size[0] and 0<=t<b<=size[1]);self.assertEqual((r-l)*ratio[1],(b-t)*ratio[0])
    def test_minimal_spends_budget_on_maximum_area_origins(self):
        boxes=generate(1440,480,(1,1),'minimal')
        self.assertEqual({(r-l,b-t) for l,t,r,b in boxes},{(480,480)})
        self.assertGreater(len(boxes),100)
        self.assertIn((0,0,480,480),boxes);self.assertIn((960,0,1440,480),boxes)
    def test_maximal_permits_broad_and_tighter_compositions(self):
        boxes=generate(720,960,(3,4),'maximal')
        self.assertIn((0,0,720,960),boxes);self.assertTrue(any((r-l)<360 for l,t,r,b in boxes))
    def test_oracle_roi_anchors_need_no_crop_reference(self):
        roi=dict(id='p',kind='person',priority='primary',box=dict(left=71,top=21,right=85,bottom=79))
        a=generate(1024,683,(1,3),'maximal',rois=[roi]);b=generate(1024,683,(1,3),'maximal',rois=[dict(roi,description='ignored')])
        self.assertEqual(a,b);self.assertLessEqual(len(a),128)

if __name__=='__main__':unittest.main()

