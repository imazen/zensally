import unittest
import torch
from torch import nn
from composition_focal_head import FocalPreferenceHead


class FocalHeadTests(unittest.TestCase):
    def test_initial_ranking_is_preserved_and_conditioning_receives_gradients(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        base = nn.Sequential(nn.LayerNorm(62), nn.Linear(62, 64), nn.ReLU(),
                             nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        features = torch.randn(2, 7, 62)
        conditions = torch.randn(2, 7, 2)
        head = FocalPreferenceHead(base.state_dict(), -0.7)
        actual = head(features, conditions)
        self.assertTrue(torch.equal(actual[..., :1], base(features)))
        self.assertTrue(torch.equal(actual[..., 1], torch.full((2, 7), -0.7)))
        actual.square().sum().backward()
        self.assertGreater(float(head.condition.weight.grad.abs().sum()), 0)
        self.assertGreater(float(head.accept.weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
