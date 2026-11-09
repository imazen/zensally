"""Tensor pair operations matching the binary64 research selection contract."""
import torch
from torch.nn import functional as F


def compatibility(boxes, eligible):
    """XYXY integer boxes (...,N,4), eligibility (...,2,N)."""
    outer=boxes.unsqueeze(-2);inner=boxes.unsqueeze(-3)
    contained=(outer[...,:2]<=inner[...,:2]).all(-1)&(outer[...,2:]>=inner[...,2:]).all(-1)
    area=(boxes[...,2]-boxes[...,0])*(boxes[...,3]-boxes[...,1])
    return contained&(area.unsqueeze(-2)<area.unsqueeze(-1))&eligible[...,0,:,None]&eligible[...,1,None,:]


def pair_scores(scores, mask):
    if not mask.any(dim=(-2,-1)).all():raise ValueError('every request requires a feasible strict pair')
    if not torch.isfinite(scores).all():raise ValueError('finite scores required')
    return (scores[...,0,:,None].double()+scores[...,1,None,:].double()).masked_fill(~mask,-torch.inf)


def pair_choices(scores, mask):
    count=scores.shape[-1];flat=pair_scores(scores,mask).flatten(-2).argmax(-1)
    return torch.stack([flat//count,flat%count],-1)


def paired_top_loss(scores, mask, target):
    count=scores.shape[-1];logits=pair_scores(scores,mask).flatten(-2)
    label=target[...,0]*count+target[...,1]
    return F.cross_entropy(logits,label,reduction='none')
