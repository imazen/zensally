"""Explicit paired abstention at a fixed candidate cut-logit threshold."""
import torch


def route(pair_scores,compatible,cut_logits):
    allowed=compatible&(cut_logits[...,0,:,None]<0)&(cut_logits[...,1,None,:]<0)
    present=allowed.any((-2,-1));count=pair_scores.shape[-1]
    flat=pair_scores.masked_fill(~allowed,-torch.inf).flatten(-2).argmax(-1)
    chosen=torch.stack([flat//count,flat%count],-1).masked_fill(~present[...,None],-1)
    return chosen,present
