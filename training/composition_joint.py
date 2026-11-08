"""Separate removable geometry head for the experimental shared crop encoder."""
import torch
from torch import nn

class AuxiliaryScorer(nn.Module):
 def __init__(self,scorer):
  super().__init__();self.base=scorer;self.auxiliary=nn.Linear(scorer[0].in_features,32)
 def forward(self,descriptor):return torch.cat((self.base(descriptor),self.auxiliary(descriptor)),dim=-1)

def auxiliary_loss(raw,targets,mask):
 error=(raw[...,2:].float()-targets).square()*mask
 return (error.sum((1,2))/mask.sum((1,2)).clamp_min(1)).mean()
