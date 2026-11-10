"""Matched scalar extension over frozen image/ranker hidden values."""
import torch
from torch import nn


class AttentionScalarHead(nn.Module):
    def __init__(self,scorer_state,active):
        super().__init__()
        if active not in [0,16,24,32]:raise ValueError('registered feature blocks required')
        self.base=nn.Linear(96,1);self.base.load_state_dict(scorer_state)
        self.extra=nn.Linear(32,1,bias=False);nn.init.zeros_(self.extra.weight)
        self.register_buffer('mask',(torch.arange(32)<active).float())

    def forward(self,hidden,extra):
        return (self.base(hidden)+self.extra(extra*self.mask))[...,0]
