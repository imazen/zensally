"""Experimental shared encoder with a primary-extent head, without face assertions."""
import torch
from torch import nn
from torch.nn import functional as F

class ImageExtentModel(nn.Module):
 def __init__(self,ranker,head,mean,std):
  super().__init__();self.ranker=ranker;self.head=head;self.register_buffer('visual_mean',mean.clone());self.register_buffer('visual_std',std.clone())
 def visual(self,image):
  features=self.ranker.encode(image)
  return torch.cat([F.adaptive_avg_pool2d(f,(4,4)).flatten(1) for f in features],-1)
 def forward(self,image,geometry):
  visual=((self.visual(image).float()-self.visual_mean)/self.visual_std).clamp(-10,10)
  return self.head(torch.cat((visual,geometry),-1))
