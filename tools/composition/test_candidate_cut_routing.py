import unittest
import torch
from candidate_cut_routing import route


class CutRoutingTests(unittest.TestCase):
    def test_unavailable_pair_never_returns_a_fallback_crop(self):
        choice,present=route(torch.ones(1,2,2),torch.ones(1,2,2,dtype=torch.bool),torch.ones(1,2,2))
        self.assertEqual(choice.tolist(),[[-1,-1]]);self.assertEqual(present.tolist(),[False])

    def test_zero_logit_is_rejected_and_ties_remain_ascending(self):
        choice,present=route(torch.ones(1,3,3),torch.ones(1,3,3,dtype=torch.bool),torch.tensor([[[0.,-1.,-1.],[-1.,0.,-1.]]]))
        self.assertEqual(choice.tolist(),[[1,0]]);self.assertEqual(present.tolist(),[True])


if __name__=='__main__':unittest.main()
