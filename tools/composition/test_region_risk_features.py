import unittest
import torch
from region_risk_features import relationships


class RelationshipTests(unittest.TestCase):
    def test_native_contact_and_partial_overlap(self):
        crops=torch.tensor([[0,0,5,5],[5,0,10,5],[2,2,7,7]])
        actual=relationships(crops,[[1,1,4,4]],10,10)
        self.assertEqual(actual[:,1:4].tolist(),[[1.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
        self.assertAlmostEqual(float(actual[2,4]),4/9,places=6)
        self.assertAlmostEqual(float(actual[2,5]),4/9,places=6)
        self.assertAlmostEqual(float(actual[2,6]),.09,places=6)
        self.assertAlmostEqual(float(actual[0,7]),.09,places=6)

    def test_integer_translation_preserves_relationships(self):
        crop=torch.tensor([[2,3,9,11],[0,0,5,6]])
        before=relationships(crop,[[3,4,7,8],[8,10,10,12]],20,20)
        after=relationships(crop+torch.tensor([3,2,3,2]),[[6,6,10,10],[11,12,13,14]],20,20)
        self.assertTrue(torch.equal(before,after))

    def test_empty_and_duplicate_regions_are_explicit(self):
        crop=torch.tensor([[0,0,10,10]])
        self.assertTrue(torch.equal(relationships(crop,[],10,10),torch.zeros(1,8)))
        one=relationships(crop,[[1,1,4,4]],10,10);two=relationships(crop,[[1,1,4,4],[1,1,4,4]],10,10)
        self.assertTrue(torch.equal(one[:,1:],two[:,1:]));self.assertGreater(float(two[0,0]),float(one[0,0]))


if __name__=='__main__':unittest.main()
