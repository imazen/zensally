import unittest
import torch
from nested_pair_selection import select
from nested_pair_training import compatibility,pair_choices,paired_top_loss


class PairTrainingTests(unittest.TestCase):
    def test_exhaustive_selector_and_gradient(self):
        boxes=torch.tensor([[0,0,12,12],[0,0,6,6],[6,6,12,12],[3,3,9,9],[0,0,0,0]])
        eligible=torch.tensor([[1,1,1,1,0],[1,1,1,1,0]],dtype=torch.bool)
        mask=compatibility(boxes,eligible)
        for seed in range(23):
            scores=torch.randint(-2,3,(2,5),generator=torch.Generator().manual_seed(seed)).float().requires_grad_()
            expected=select(boxes.tolist(),boxes.tolist(),*scores.tolist(),*eligible.tolist(),'joint-sum')
            choice=pair_choices(scores,mask)
            self.assertEqual(choice.tolist(),list(expected))
            loss=paired_top_loss(scores.unsqueeze(0),mask.unsqueeze(0),choice.unsqueeze(0)).sum();loss.backward()
            self.assertTrue(torch.isfinite(scores.grad).all());self.assertEqual(scores.grad[:,4].tolist(),[0,0])

    def test_missing_pair_is_error(self):
        with self.assertRaises(ValueError):pair_choices(torch.zeros(2,3),torch.zeros(3,3,dtype=torch.bool))


if __name__=='__main__':unittest.main()
