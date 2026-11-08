"""Frozen-encoder primary-extent heads; predicted regions remain soft evidence."""
import torch
from torch import nn

class ExtentHead(nn.Module):
 def __init__(self):
  super().__init__();self.layers=nn.Sequential(nn.Linear(516,64),nn.ReLU(),nn.Linear(64,6))
 def forward(self,features):
  logits=self.layers(features).float().reshape(*features.shape[:-1],2,3);p=logits.softmax(-1);left=(1-1e-4)*p[...,0];right=1-(1-1e-4)*p[...,2]
  return torch.stack((left[...,0],left[...,1],right[...,0],right[...,1]),-1)

def box_metrics(predicted,target):
 intersection=(torch.minimum(predicted[...,2:],target[...,2:])-torch.maximum(predicted[...,:2],target[...,:2])).clamp_min(0).prod(-1);area=(predicted[...,2:]-predicted[...,:2]).clamp_min(0).prod(-1);other=(target[...,2:]-target[...,:2]).clamp_min(0).prod(-1);iou=intersection/(area+other-intersection).clamp_min(1e-12)
 return {'edge_mse':float((predicted-target).square().mean()),'edge_mae':float((predicted-target).abs().mean()),'mean_iou':float(iou.mean())},iou


def extent_objective(predicted,target):
 from torchvision.ops import generalized_box_iou_loss
 return 5*(predicted-target).abs().mean()+2*generalized_box_iou_loss(predicted,target,reduction="mean")
