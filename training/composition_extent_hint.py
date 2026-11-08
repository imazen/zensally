"""Training-only spatial feature hints for the small primary-extent encoder."""
import torch
from torch import nn
from torch.nn import functional as F

from composition_extent_image import ImageExtentModel


class HintedExtentModel(ImageExtentModel):
    def __init__(self, ranker, head, mean, std):
        super().__init__(ranker, head, mean, std)
        self.hint_adapter = nn.Conv2d(16, 16, 1)

    def forward_with_hint(self, image, geometry):
        features = self.ranker.encode(image)
        pools = [F.adaptive_avg_pool2d(f, (4, 4)) for f in features]
        visual = torch.cat([f.flatten(1) for f in pools], -1)
        visual = ((visual.float() - self.visual_mean) / self.visual_std).clamp(-10, 10)
        box = self.head(torch.cat((visual, geometry), -1))
        return box, self.hint_adapter(pools[0]).flatten(1)


def hint_loss(predicted, target, fit_std):
    return ((predicted - target) / fit_std.clamp_min(0.1)).square().mean()
