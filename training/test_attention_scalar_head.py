import unittest
import torch
from attention_scalar_head import AttentionScalarHead


class AttentionHead(unittest.TestCase):
    def test_initial_replay_and_masked_features_under_gradient(self):
        torch.manual_seed(31);base=torch.nn.Linear(96,1);hidden=torch.randn(2,3,96);extra=torch.randn(2,3,32)
        for active in [0,16,24,32]:
            head=AttentionScalarHead(base.state_dict(),active)
            self.assertTrue(torch.equal(head(hidden,extra),base(hidden)[...,0]))
            head(hidden,extra).sum().backward()
            self.assertTrue(torch.equal(head.extra.weight.grad[:,active:],torch.zeros_like(head.extra.weight.grad[:,active:])))
            if active:self.assertGreater(float(head.extra.weight.grad[:,:active].abs().sum()),0.)


if __name__=='__main__':unittest.main()
