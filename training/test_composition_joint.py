import unittest
import torch
from torch import nn
from composition_joint import AuxiliaryScorer,auxiliary_loss

class JointTests(unittest.TestCase):
 def test_inactive_heads_do_not_advance_existing_adam_momentum(self):
  torch.manual_seed(137);model=AuxiliaryScorer(nn.Sequential(nn.Linear(10,8),nn.ReLU(),nn.Linear(8,2)));optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1);x=torch.randn(2,3,10);mask=torch.ones(2,3,32);target=torch.zeros_like(mask)
  model.task='auxiliary';optimizer.zero_grad(set_to_none=True);auxiliary_loss(model(x),target,mask).backward();optimizer.step();aux_before=[p.detach().clone() for p in model.auxiliary.parameters()]
  model.task='crop';optimizer.zero_grad(set_to_none=True);model(x).square().mean().backward();optimizer.step()
  self.assertTrue(all(p.grad is None and torch.equal(p,b) for p,b in zip(model.auxiliary.parameters(),aux_before)))
  crop_before=[p.detach().clone() for p in model.base.parameters()];model.task='auxiliary';optimizer.zero_grad(set_to_none=True);auxiliary_loss(model(x),target,mask).backward();optimizer.step()
  self.assertTrue(all(p.grad is None and torch.equal(p,b) for p,b in zip(model.base.parameters(),crop_before)))
 def test_auxiliary_gradient_does_not_change_original_logits_or_fake_missing_pose(self):
  torch.manual_seed(31);base=nn.Sequential(nn.Linear(10,8),nn.ReLU(),nn.Linear(8,2));model=AuxiliaryScorer(base);x=torch.randn(2,3,10,requires_grad=True);expected=base(x).detach();raw=model(x)
  self.assertTrue(torch.equal(raw[...,:2],expected));target=torch.zeros(2,3,32);mask=torch.zeros_like(target);mask[...,24:]=1;value=auxiliary_loss(raw,target,mask);value.backward()
  self.assertGreater(float(x.grad.abs().sum()),0);self.assertTrue(torch.equal(model.auxiliary.weight.grad[:24],torch.zeros_like(model.auxiliary.weight.grad[:24])));self.assertTrue(torch.equal(base[0].weight.grad,torch.zeros_like(base[0].weight.grad)))
if __name__=='__main__':unittest.main()
