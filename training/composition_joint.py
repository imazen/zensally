"""Separate removable geometry head for the experimental shared crop encoder."""
import torch
from torch import nn

class AuxiliaryScorer(nn.Module):
 def __init__(self,scorer):
  super().__init__();self.base=scorer;self.auxiliary=nn.Linear(scorer[0].in_features,32);self.task='both'
 def forward(self,descriptor):
  if self.task=='crop':return self.base(descriptor)
  if self.task=='auxiliary':return self.auxiliary(descriptor)
  if self.task!='both':raise ValueError('unknown head task')
  return torch.cat((self.base(descriptor),self.auxiliary(descriptor)),dim=-1)

def auxiliary_loss(raw,targets,mask):
 if raw.shape[-1] not in [32,34]:raise ValueError('auxiliary-only or combined head output required')
 error=(raw[...,-32:].float()-targets).square()*mask
 return (error.sum((1,2))/mask.sum((1,2)).clamp_min(1)).mean()
