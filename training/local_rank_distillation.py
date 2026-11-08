"""Scalar crop-rank research model and family-balanced weak-target objective."""
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F

from crop_fit import TrainingModel

ARMS = ['frozen-image', 'trainable-image', 'geometry-only', 'permuted-image']


class LocalCropRanker(nn.Module):
    def __init__(self, initial=None):
        super().__init__()
        base = TrainingModel()
        if initial is not None:
            base.load_state_dict(initial)
        self.ranker = base.ranker
        previous = self.ranker.scorer[-1]
        scalar = nn.Linear(previous.in_features, 1)
        with torch.no_grad():
            scalar.weight.copy_(previous.weight[:1])
            scalar.bias.copy_(previous.bias[:1])
        self.ranker.scorer[-1] = scalar
        self.configure('trainable-image')

    def configure(self, arm):
        if arm not in ARMS:
            raise ValueError('unknown registered rank arm')
        self.arm = arm
        self.requires_grad_(arm in ['trainable-image', 'permuted-image'])
        self.ranker.scorer.requires_grad_(True)

    def train_arm(self):
        self.train()
        if self.arm in ['frozen-image', 'geometry-only']:
            for name, module in self.ranker.named_children():
                if name != 'scorer':
                    module.eval()

    def feature_maps(self, image):
        if self.arm == 'geometry-only':
            return (image.new_zeros(len(image), 16, 32, 32), image.new_zeros(len(image), 16, 16, 16))
        return self.ranker.encode(image)

    def scores(self, features, boxes, source_aspect, target_aspect, mode, content):
        count = boxes.shape[1]
        return self.ranker.raw_scores(features, boxes, source_aspect,
            target_aspect[:, None].expand(-1, count), mode[:, None].expand(-1, count),
            boxes.new_zeros(len(boxes), 6), content)[..., 0]


def family_weights(sources, indices):
    count = Counter(sources[i]['family_id'] for i in indices)
    return torch.tensor([1/(len(count)*count[sources[i]['family_id']]) for i in indices])


def objective(scores, target, pairs, pair_valid, eligible, query_valid):
    delta = scores.gather(1, pairs[..., 0])-scores.gather(1, pairs[..., 1])
    pair_loss = (F.softplus(-delta)*pair_valid).sum(-1)/pair_valid.sum(-1).clamp_min(1)
    mse = ((scores-target).square()*eligible).sum(-1)/eligible.sum(-1).clamp_min(1)
    return (pair_loss+.25*mse)*query_valid
