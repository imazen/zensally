"""Detached CPU snapshots for bounded best-state retention during research fits."""
import copy
import torch


def snapshot(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:snapshot(v) for k,v in value.items()}
    if isinstance(value,list):return [snapshot(v) for v in value]
    if isinstance(value,tuple):return tuple(snapshot(v) for v in value)
    return copy.deepcopy(value)
