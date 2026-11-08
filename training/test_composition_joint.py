import unittest
import torch
from torch import nn
from composition_joint import AuxiliaryScorer,auxiliary_loss

class JointTests(unittest.TestCase):
 def test_auxiliary_gradient_does_not_change_original_logits_or_fake_missing_pose(self):
  torch.manual_seed(31);base=nn.Sequential(nn.Linear(10,8),nn.ReLU(),nn.Linear(8,2));model=AuxiliaryScorer(base);x=torch.randn(2,3,10,requires_grad=True);expected=base(x).detach();raw=model(x)
  self.assertTrue(torch.equal(raw[...,:2],expected));target=torch.zeros(2,3,32);mask=torch.zeros_like(target);mask[...,24:]=1;value=auxiliary_loss(raw,target,mask);value.backward()
  self.assertGreater(float(x.grad.abs().sum()),0);self.assertTrue(torch.equal(model.auxiliary.weight.grad[:24],torch.zeros_like(model.auxiliary.weight.grad[:24])));self.assertIsNone(base[0].weight.grad)
if __name__=='__main__':unittest.main()
