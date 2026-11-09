import unittest
import torch
from distill_attention_linear import Student


class Fusion(unittest.TestCase):
    def test_nonlinear_zero_initialization_frozen_base_and_compact_output(self):
        torch.manual_seed(137);base=torch.nn.Linear(96,1);hidden=torch.randn(2,3,96)
        model=Student(base.state_dict(),torch.randn(1,32),torch.randn(96),torch.rand(96)+.5,nonlinear=True)
        self.assertTrue(torch.equal(model(hidden,None),base(hidden)[...,0]))
        optimizer=torch.optim.AdamW(model.probe.parameters(),lr=.001)
        model.features(hidden).sub(1).square().mean().backward();optimizer.step()
        self.assertTrue(all(torch.equal(v,base.state_dict()[k]) for k,v in model.base.state_dict().items()))
        torch.testing.assert_close(model.compact_scores(hidden),model(hidden,None),atol=1e-5,rtol=1e-5)
        self.assertIsNone(model.fused())

    def test_nonzero_normalization_bias_and_residual_fuse(self):
        torch.manual_seed(31);base=torch.nn.Linear(96,1)
        model=Student(base.state_dict(),torch.randn(1,32),torch.randn(96),torch.rand(96)+.5)
        with torch.no_grad():
            model.probe.weight.normal_(0,.1);model.probe.bias.normal_(0,.1)
            hidden=torch.randn(2,3,96)
            expected=model(hidden,None);actual=torch.nn.functional.linear(hidden,**model.fused())[...,0]
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
        self.assertTrue(all(not v.requires_grad for v in model.base.parameters()))


if __name__=='__main__':unittest.main()
