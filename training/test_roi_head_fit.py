import unittest
import torch
from roi_head_fit import per_roi_objective, source_weights


class SourceBalancedLoss(unittest.TestCase):
    def test_duplicate_roi_does_not_increase_source_weight(self):
        sources = [dict(source_id='a'), dict(source_id='a'), dict(source_id='b')]
        weights = source_weights(sources, [0, 1, 2])
        torch.testing.assert_close(weights, torch.tensor([.25, .25, .5]))
        boxes = torch.tensor([[.1, .2, .6, .7], [.1, .2, .6, .7], [.2, .1, .7, .8]])
        targets = torch.tensor([[.2, .1, .7, .8], [.2, .1, .7, .8], [.1, .1, .6, .7]])
        duplicated = (per_roi_objective(boxes, targets)*weights).sum()
        reference = per_roi_objective(boxes[[0, 2]], targets[[0, 2]]).mean()
        torch.testing.assert_close(duplicated, reference)


if __name__ == '__main__':
    unittest.main()
