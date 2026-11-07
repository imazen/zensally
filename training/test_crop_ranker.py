import unittest
import torch
from crop_ranker import CropRanker, preference_loss


class CropRankerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        self.model = CropRanker().eval()
        self.image = torch.rand(1, 3, 64, 64)
        self.boxes = torch.tensor([[[0., 0., 1., 1.], [0., 0., .7, .8], [.3, .2, 1., 1.]]])
        self.args = (torch.tensor([4 / 3]), torch.tensor([[1., 9 / 16, 16 / 9]]), torch.tensor([[0., 1., 1.]]), torch.tensor([[0., 0., 0., 0., 0., 1.]]), torch.ones(1, 3, dtype=torch.bool), torch.tensor([[0., .125, 1., .875]]))

    def test_one_encoding_for_many_ratios_and_cached_chunk_scores(self):
        calls = []
        hook = self.model.stem.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            features = self.model.encode(self.image)
            full = self.model.score(features, self.boxes, *self.args)
            parts = []
            for i in range(3):
                sa, ta, mode, faces, legal, valid = self.args
                parts.append(self.model.score(features, self.boxes[:, i:i+1], sa, ta[:, i:i+1], mode[:, i:i+1], faces, legal[:, i:i+1], valid))
        hook.remove()
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(full, torch.cat(parts, 1))
        self.assertTrue(torch.isfinite(full).all())

    def test_illegal_candidates_are_never_ranked_as_valid(self):
        sa, ta, mode, faces, legal, valid = self.args
        legal[:] = False
        scores = self.model(self.image, self.boxes, sa, ta, mode, faces, legal, valid)
        self.assertTrue(torch.isneginf(scores).all())
        with self.assertRaises(ValueError):
            preference_loss(scores, torch.tensor([[0, 0, 1]]), torch.ones(1), torch.zeros(1, 3, dtype=torch.long))

    def test_pairwise_supervision_reaches_shared_encoder(self):
        sa, _, _, faces, legal, valid = self.args
        scores = self.model(self.image, self.boxes, sa, torch.ones(1, 3), torch.ones(1, 3), faces, legal, valid)
        loss = preference_loss(scores, torch.tensor([[0, 0, 1]]), torch.ones(1), torch.zeros(1, 3, dtype=torch.long))
        loss.backward()
        grad = self.model.stem[0].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.)

    def test_cross_task_preference_is_rejected(self):
        with self.assertRaises(ValueError):
            preference_loss(torch.tensor([[1., 0.]]), torch.tensor([[0, 0, 1]]), torch.ones(1), torch.tensor([[0, 1]]))

    def test_face_union_and_review_are_hard_even_with_incorrect_legal_mask(self):
        sa, ta, mode, _, legal, valid = self.args
        faces = torch.tensor([[1., .1, .1, .2, .2, 1.]])
        with torch.no_grad():
            scores = self.model(self.image, self.boxes, sa, ta, mode, faces, legal, valid)
            self.assertTrue(torch.isfinite(scores[0, 0]))
            self.assertTrue(torch.isneginf(scores[0, 2]))
            faces[:, 5] = 0
            scores = self.model(self.image, self.boxes, sa, ta, mode, faces, legal, valid)
            self.assertTrue(torch.isneginf(scores).all())

    def test_candidate_order_does_not_change_scores(self):
        sa, ta, mode, faces, legal, valid = self.args
        order = torch.tensor([2, 0, 1])
        with torch.no_grad():
            original = self.model(self.image, self.boxes, *self.args)
            reordered = self.model(self.image, self.boxes[:, order], sa, ta[:, order], mode[:, order], faces, legal[:, order], valid)
        torch.testing.assert_close(reordered, original[:, order])

    def test_thin_content_stays_finite(self):
        sa, ta, mode, faces, legal, _ = self.args
        valid = torch.tensor([[.499, 0., .501, 1.]])
        with torch.no_grad():
            scores = self.model(self.image, self.boxes, sa, ta, mode, faces, legal, valid)
        self.assertTrue(torch.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
