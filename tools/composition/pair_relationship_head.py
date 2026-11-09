"""Small research-only pair corrections on fixed endpoint embeddings."""
import math
import torch
from torch import nn


class PairCorrection(nn.Module):
    def __init__(self,arm,mean,std):
        super().__init__()
        if arm not in ['additive-linear','pair-real','pair-permuted']:raise ValueError('registered arm required')
        self.arm=arm;self.register_buffer('mean',mean.clone());self.register_buffer('std',std.clone())
        width=1 if arm=='additive-linear' else 8
        self.outer=nn.Linear(96,width);self.inner=nn.Linear(96,width)
        nn.init.zeros_(self.inner.weight);nn.init.zeros_(self.inner.bias)
        if arm=='additive-linear':nn.init.zeros_(self.outer.weight);nn.init.zeros_(self.outer.bias)

    def forward(self,hidden,base,eligible):
        outer=(hidden[...,0,:,:]-self.mean)/self.std;inner=(hidden[...,1,:,:]-self.mean)/self.std
        if self.arm=='pair-permuted':
            # Fixed eligible-only cyclic derangement; never move placeholder features.
            inner=inner.clone();flat=inner.reshape(-1,inner.shape[-2],96);masks=eligible[...,1,:].reshape(-1,inner.shape[-2])
            for values,mask in zip(flat,masks):
                indices=mask.nonzero().flatten()
                if len(indices)>1:values[indices]=values[indices.roll(1)]
        a=self.outer(outer);b=self.inner(inner)
        correction=a[...,0].unsqueeze(-1)+b[...,0].unsqueeze(-2) if self.arm=='additive-linear' else a@b.transpose(-2,-1)/math.sqrt(8)
        return base[...,0,:,None].double()+base[...,1,None,:].double()+correction.double()
