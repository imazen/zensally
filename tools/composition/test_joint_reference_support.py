import itertools
import unittest
import numpy as np
from joint_reference_support import bank_compatibility,interval_mask
from crop_corridors import audit


class JointSupportTests(unittest.TestCase):
    def test_all_small_pairs_match_existing_continuous_interval_audit(self):
        boxes=[[x,y,x+k,y+k] for k in range(1,5) for x in range(5-k) for y in range(5-k)];eligible=np.ones((2,len(boxes)),bool);heads=[[1,1,2,2],[3,2,4,3]];strict=bank_compatibility(boxes,boxes,eligible);safe,retained=interval_mask(boxes,boxes,eligible,heads)
        for i,j in itertools.product(range(len(boxes)),repeat=2):
            expected=audit(boxes[j],boxes[i],4,4,'1:1',heads)['geometrically_admissible'] and boxes[i]!=boxes[j]
            self.assertEqual(bool(strict[i,j] and safe[i,j]),expected)
            if strict[i,j] and retained[i,j]:self.assertTrue(expected)

    def test_mixed_banks_keep_eligibility_and_allow_whole_head_exclusion(self):
        outer=[[0,0,4,4],[0,0,2,2]];inner=[[0,0,2,2],[0,0,1,1]];e=np.array([[False,True],[True,True]]);strict=bank_compatibility(outer,inner,e);safe,retained=interval_mask(outer,inner,e,[[3,3,4,4]])
        self.assertEqual(np.argwhere(strict&safe).tolist(),[[1,1]])
        self.assertFalse((strict&retained).any())


if __name__=='__main__':unittest.main()
