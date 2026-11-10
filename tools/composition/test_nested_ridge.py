import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from nested_crop_head import context,decode
from nested_ridge import fractions,native,ridge


class RidgeTests(unittest.TestCase):
    def test_fraction_inverse_replays_odd_source_boundary_crops(self):
        ctx=context([[413,197],[197,413]],['1:1','3:1','1:3']);torch.manual_seed(137);_,boxes=decode(torch.randn(2,3,5)*10,ctx,integer=True);_,replay=native(fractions(boxes,ctx),ctx);self.assertTrue(torch.equal(boxes,replay))

    def test_dual_fit_satisfies_weighted_primal_normal_equations(self):
        torch.manual_seed(31);x=torch.randn(13,7,dtype=torch.float64);y=torch.randn(13,5,dtype=torch.float64);weights=torch.rand(13,dtype=torch.float64);weights/=weights.sum();lambdas=torch.tensor([.001,.1,10.],dtype=torch.float64);model=ridge(x,y,weights,lambdas);z=(x-model['mean'])/model['scale'];centered=y-model['target_mean']
        for i,value in enumerate(lambdas):
            lhs=(z.T@(z*weights[:,None])+value*torch.eye(7,dtype=x.dtype))@model['coefficients'][i];rhs=z.T@(centered*weights[:,None]);self.assertLess(float((lhs-rhs).abs().max()),1e-10)


if __name__=='__main__':unittest.main()
