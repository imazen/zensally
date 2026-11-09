import unittest
import torch
from distill_attention_linear import Student


class Fusion(unittest.TestCase):
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
