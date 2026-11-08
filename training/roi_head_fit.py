"""Source-balanced weak head-localization losses; estimates are not safety bounds."""
from collections import Counter
import torch
from torchvision.ops import generalized_box_iou_loss


def source_weights(sources, indices):
    counts = Counter(sources[i]['source_id'] for i in indices)
    return torch.tensor([1/(len(counts)*counts[sources[i]['source_id']]) for i in indices], dtype=torch.float32)


def per_roi_objective(boxes, targets):
    return 5*(boxes-targets).abs().mean(-1)+2*generalized_box_iou_loss(boxes, targets, reduction='none')
