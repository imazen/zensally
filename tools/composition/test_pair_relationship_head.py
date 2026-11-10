import unittest
import torch
from pair_relationship_head import PairCorrection


class PairHeadTests(unittest.TestCase):
    def test_initial_scores_and_finite_learning(self):
        hidden=torch.randn(2,3,2,5,96);base=torch.randn(2,3,2,5);eligible=torch.ones(2,3,2,5,dtype=torch.bool)
        for arm in ['additive-linear','pair-real','pair-permuted']:
            model=PairCorrection(arm,torch.zeros(96),torch.ones(96));actual=model(hidden,base,eligible)
            self.assertEqual(actual.shape,(2,3,5,5));self.assertTrue(torch.equal(actual,base[...,0,:,None].double()+base[...,1,None,:].double()))
            actual[...,0,1].sum().backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(float(model.inner.weight.grad.abs().sum()),0)

    def test_bilinear_can_rank_joint_choices_that_additive_cannot(self):
        model=PairCorrection('pair-real',torch.zeros(96),torch.ones(96))
        with torch.no_grad():
            model.outer.weight.zero_();model.outer.bias.zero_();model.outer.weight[0,0]=1;model.inner.weight[0,0]=1
        hidden=torch.zeros(1,2,2,96);hidden[0,:,0,0]=1;hidden[0,:,1,0]=-1
        scores=model(hidden,torch.zeros(1,2,2),torch.ones(1,2,2,dtype=torch.bool))[0]
        # Positive diagonal with negative off-diagonal violates additive rectangle equality.
        self.assertGreater(float((scores[0,0]+scores[1,1]).detach()),float((scores[0,1]+scores[1,0]).detach()))


if __name__=='__main__':unittest.main()
