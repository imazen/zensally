import unittest
import numpy as np
from relationship_features import features


class Relationships(unittest.TestCase):
    def test_group_retention_separation_and_order_invariance(self):
        people=[dict(head=[.1,.2,.2,.4],target=[.75,.3],in_frame_score=1),dict(head=[.7,.2,.8,.4],target=[.15,.3],in_frame_score=1)]
        boxes=[[0,0,1,1],[0,0,.5,1]];values=features(boxes,people)
        self.assertEqual(values[0,3],1);self.assertEqual(values[0,4],0)
        self.assertEqual(values[1,3],0);self.assertEqual(values[1,4],1)
        self.assertEqual(values[0,7],1);self.assertEqual(values[1,8],1)
        np.testing.assert_allclose(values,features(boxes,list(reversed(people))),atol=1e-7)
        self.assertTrue(np.array_equal(features(boxes,[]),np.zeros((2,16),np.float32)))

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):features([[0,0,0,1]],[])
        with self.assertRaises(ValueError):features([[0,0,1,1]],[dict(head=[0,0,1,1],target=[float('nan'),0],in_frame_score=1)])


if __name__=='__main__':unittest.main()
