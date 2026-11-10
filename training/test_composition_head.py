import unittest
import torch
from torch import nn
from composition_head import CompositionHead

class HeadTests(unittest.TestCase):
 def test_initial_logits_match_frozen_head_and_new_geometry_can_learn(self):
  torch.set_num_threads(1);torch.manual_seed(31);base=nn.Sequential(nn.Linear(7,5),nn.ReLU(),nn.Linear(5,2));x=torch.randn(2,4,7);extras=torch.rand(2,4,32)
  heads=[CompositionHead(base,n) for n in [0,16,24,32]]
  self.assertEqual(len({sum(p.numel() for p in h.parameters()) for h in heads}),1)
  for h in heads:self.assertTrue(torch.equal(h(x,extras),base(x)))
  h=heads[-1];h(x,extras).square().sum().backward();self.assertGreater(h.extra.weight.grad.abs().sum(),0)
  h=heads[0];h(x,extras).square().sum().backward();self.assertEqual(float(h.extra.weight.grad.abs().sum()),0)
if __name__=='__main__':unittest.main()
