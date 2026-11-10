"""Auxiliary person-record geometry for crop transfer; never a face guard."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def statistics(records):
    if not records or len(records) > 1024:
        raise ValueError('one to1024 annotation records required')
    boxes = np.asarray([r['box'] for r in records], np.float32)
    if not np.isfinite(boxes).all() or not ((boxes[:,:2]>=0).all() and (boxes[:,2:]<=1).all() and (boxes[:,2:]>boxes[:,:2]).all()):
        raise ValueError('valid normalized annotation boxes required')
    individual = np.asarray([not r['flags']['IsGroupOf'] for r in records])
    target = np.zeros(10, np.float32)
    valid = np.ones(10, np.float32)
    target[:2] = boxes[:,:2].min(0)
    target[2:4] = boxes[:,2:].max(0)
    centers = (boxes[individual,:2]+boxes[individual,2:])/2
    if len(centers):
        target[4:6] = centers.mean(0)
        target[6:8] = centers.std(0)*2
    else:
        valid[4:8] = 0
    target[8] = math.log1p(int(individual.sum()))/math.log(1025)
    target[9] = math.log1p(int((~individual).sum()))/math.log(1025)
    return target, valid


class PersonHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(517,64),nn.ReLU(),nn.Linear(64,10))

    def forward(self, features, content, aspect, *, geometry=False):
        if geometry:
            visual = content.new_zeros(len(content),512)
        else:
            visual = torch.cat([F.adaptive_avg_pool2d(f,(4,4)).flatten(1) for f in features],-1)
        value = self.layers(torch.cat([visual,content,aspect.log()[:,None]],-1)).sigmoid()
        lo = value[:,:2]
        hi = lo+(1-lo)*value[:,2:4]
        return torch.cat([lo,hi,value[:,4:]],-1)


def auxiliary_loss(prediction, target, valid):
    from torchvision.ops import generalized_box_iou_loss
    per_source = ((prediction-target).abs()*valid).sum(-1)/valid.sum(-1)
    return per_source+.2*generalized_box_iou_loss(prediction[:,:4],target[:,:4],reduction='none')


def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
