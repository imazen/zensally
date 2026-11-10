"""Matched experimental scoring heads over frozen candidate descriptors."""
import copy
import torch
from torch import nn

class CompositionHead(nn.Module):
 def __init__(self,scorer,active=32):
  super().__init__()
  if active not in [0,16,24,32]:raise ValueError('registered feature blocks only')
  self.base=copy.deepcopy(scorer);self.extra=nn.Linear(32,self.base[0].out_features,bias=False)
  nn.init.zeros_(self.extra.weight);self.register_buffer('mask',(torch.arange(32)<active).float())
 def forward(self,descriptor,extras):
  hidden=self.base[0](descriptor)+self.extra(extras*self.mask)
  return self.base[2](self.base[1](hidden))
