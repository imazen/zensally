import unittest
from crop_policy import Rect,_region_windows,plan_crops

class RegionTests(unittest.TestCase):
 def test_constrained_offsets_recover_a_window_missed_by_global_grid(self):
  protected=Rect(0,31,100,63);height=100/3
  coarse=[Rect(0,(100-height)*i/4,100,(100-height)*i/4+height) for i in range(5)]
  self.assertFalse(any(r.contains(protected) for r in coarse))
  constrained=_region_windows(100,100,(3,1),(protected,),(1.,),3)
  self.assertGreater(len(constrained),0)
  self.assertTrue(all(r.contains(protected) for r in constrained))
  self.assertEqual(plan_crops(100,100,(3,1),protected_faces=(protected,),faces_reviewed=False).kind,'contain')
 def test_widely_separated_regions_cannot_be_ranked_into_a_narrow_crop(self):
  regions=(Rect(5,5,40,30),Rect(60,70,95,95))
  self.assertEqual(_region_windows(100,100,(3,1),regions,(1.,.9,.8,.7,.65),3),())
if __name__=='__main__':unittest.main()
