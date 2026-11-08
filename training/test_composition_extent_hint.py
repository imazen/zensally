import copy
import unittest

import torch

from composition_extent import ExtentHead, extent_objective
from composition_extent_hint import HintedExtentModel, hint_loss
from test_composition_extent_image import TinyRanker


class ExtentHintTests(unittest.TestCase):
    def test_shared_forward_and_inactive_adapter_momentum(self):
        torch.manual_seed(31)
        model = HintedExtentModel(TinyRanker(), ExtentHead(), torch.zeros(512), torch.ones(512))
        image = torch.randn(2, 3, 16, 16)[:, :, ::2, ::2]
        geometry = torch.zeros(2, 4)
        target = torch.tensor([[.1, .2, .8, .9], [0., 0., 1., 1.]])
        box, hint = model.forward_with_hint(image, geometry)
        torch.testing.assert_close(box, model(image, geometry), rtol=0, atol=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        loss = hint_loss(hint, torch.zeros_like(hint), torch.ones(256))
        loss.backward()
        self.assertGreater(float(model.ranker.project.weight.grad.abs().sum()), 0)
        optimizer.step()
        warmed = copy.deepcopy(model.hint_adapter.state_dict())
        optimizer.zero_grad(set_to_none=True)
        extent_objective(model(image, geometry), target).backward()
        for parameter in model.hint_adapter.parameters():
            self.assertIsNone(parameter.grad)
        optimizer.step()
        for name, value in warmed.items():
            torch.testing.assert_close(value, model.hint_adapter.state_dict()[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
