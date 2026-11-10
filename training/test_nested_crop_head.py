import unittest
import torch
from nested_crop_head import context,decode,endpoint_loss


class NestedHeadTests(unittest.TestCase):
    def test_extreme_logits_preserve_exact_native_aspect_and_strict_nesting(self):
        data=context([[413,197],[197,413],[19,23]],['1:1','4:3','3:4','3:1','1:3','6:1','1:6']);torch.manual_seed(31)
        for logits in [torch.randn(3,7,5)*100,torch.zeros(3,7,5)]:
            _,native=decode(logits,data,integer=True);outer,inner=native.unbind(-2);self.assertTrue(bool((outer[...,:2]<=inner[...,:2]).all() and (inner[...,2:]<=outer[...,2:]).all()));self.assertTrue(bool((native[...,:2]>=0).all() and (native[...,2:]<=data['sizes'][:,None,None]).all()));size=native[...,2:]-native[...,:2];p,q=data['pairs'].unbind(-1);self.assertTrue(bool((size[...,0]*q[None,:,None]==size[...,1]*p[None,:,None]).all()));self.assertTrue(bool((size[:,:,1,0]<size[:,:,0,0]).all()));self.assertTrue(bool((size[:,:,0]==data['maximum'][...,None]*data['pairs'][None]).all()))

    def test_unknown_requests_supply_no_coordinate_gradient(self):
        prediction=torch.rand(2,3,2,4,requires_grad=True);targets=torch.rand(2,3,2,2,4);valid=torch.zeros(2,3,2,dtype=torch.bool);valid[0,1,0]=True;valid[1,0,:]=True;loss=endpoint_loss(prediction,targets,valid);loss.backward();self.assertTrue(bool(torch.isfinite(prediction.grad).all()));self.assertTrue(bool((prediction.grad[~valid.any(-1)]==0).all()));self.assertGreater(float(prediction.grad[valid.any(-1)].abs().sum()),0)


if __name__=='__main__':unittest.main()
