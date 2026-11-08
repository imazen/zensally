import unittest
import torch
from local_rank_distillation import LocalCropRanker, family_weights, objective


class LocalRanks(unittest.TestCase):
    def test_repeating_a_whole_family_does_not_change_its_total_weight(self):
        original = [dict(family_id=f) for f in ['a', 'a', 'b']]
        repeated = [dict(family_id=f) for f in ['a', 'a', 'a', 'a', 'b']]
        a = (family_weights(original, [0, 1, 2])*torch.tensor([1., 3., 7.])).sum()
        b = (family_weights(repeated, list(range(5)))*torch.tensor([1., 3., 1., 3., 7.])).sum()
        self.assertEqual(float(a), float(b))

    def test_padding_never_becomes_a_preference_or_mse_target(self):
        scores = torch.tensor([[1., -1., 5.]])
        target = torch.tensor([[1., -1., 0.]])
        pairs = torch.tensor([[[0, 1], [2, 0]]])
        pm = torch.tensor([[True, False]])
        mask = torch.tensor([[True, True, False]])
        available = torch.tensor([True])
        expected = objective(scores, target, pairs, pm, mask, available)
        scores[0, 2], target[0, 2] = 1e9, -1e9
        self.assertTrue(torch.equal(expected, objective(scores, target, pairs, pm, mask, available)))
        scores[0, 0] = -1
        self.assertGreater(float(objective(scores, target, pairs, pm, mask, available)), float(expected))

    def test_geometry_path_leaves_warmed_encoder_optimizer_and_state_unchanged(self):
        torch.set_num_threads(2)
        model = LocalCropRanker()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        image = torch.rand(2, 3, 256, 256)
        boxes = torch.tensor([[[0., 0., .8, .8]], [[.1, .1, 1., 1.]]])
        shape = torch.ones(2)
        content = torch.tensor([[0., 0., 1., 1.]]).repeat(2, 1)
        model.train_arm()
        model.scores(model.feature_maps(image), boxes, shape, shape, shape, content).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        before = {k:v.clone() for k,v in model.ranker.state_dict().items() if not k.startswith('scorer.')}
        encoder = [p for name,p in model.ranker.named_parameters() if not name.startswith('scorer.')]
        saved = {p:{key:value.clone() if isinstance(value, torch.Tensor) else value
                    for key,value in optimizer.state[p].items()} for p in encoder if p in optimizer.state}
        self.assertTrue(saved)
        model.configure('geometry-only')
        model.train_arm()
        features = model.feature_maps(image)
        self.assertTrue(all(bool((value == 0).all()) for value in features))
        model.scores(features, boxes, shape, shape, shape, content).sum().backward()
        optimizer.step()
        after = {k:v for k,v in model.ranker.state_dict().items() if not k.startswith('scorer.')}
        self.assertTrue(all(torch.equal(value, after[key]) for key,value in before.items()))
        self.assertTrue(all(torch.equal(value, optimizer.state[p][key]) if isinstance(value, torch.Tensor)
                            else value == optimizer.state[p][key]
                            for p,state in saved.items() for key,value in state.items()))


if __name__ == '__main__':
    unittest.main()
