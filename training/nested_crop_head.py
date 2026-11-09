"""Research single nested crop pair with optional measured-attention distillation."""
import torch
from torch import nn
from direct_crop import geometry


class NestedCropHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(2050,64),nn.ReLU())
        self.attention=nn.Linear(64,32)
        self.crop=nn.Sequential(nn.Linear(98,32),nn.ReLU(),nn.Linear(32,5))

    def forward(self,visual,query,measured=None):
        hidden=self.trunk(torch.cat((visual[:,None].expand(-1,query.shape[1],-1),query),-1));prediction=self.attention(hidden);aux=prediction if measured is None else measured
        return self.crop(torch.cat((hidden,aux,query),-1)),prediction


def context(sizes,ratios,*,device=None):
    original=geometry(sizes,ratios,device=device)
    if not bool((original['maximum']>=2).all()):raise ValueError('strict pair requires at least two native ratio units')
    return dict(sizes=original['sizes'],pairs=original['pairs'][::2],maximum=original['maximum'][:,::2],query=original['query'][:,::2,:2])


def subset(data,indices):return dict(data,sizes=data['sizes'][indices],maximum=data['maximum'][indices],query=data['query'][indices])


def decode(logits,data,*,integer=False):
    extent=data['maximum'][...,None]*data['pairs'][None];origin=logits[...,:2].sigmoid()*(data['sizes'][:,None]-extent)
    if integer:origin=origin.round()
    k=1+logits[...,2].sigmoid()*(data['maximum']-2)
    if integer:k=k.round()
    inner_extent=k[...,None]*data['pairs'][None];offset=logits[...,3:5].sigmoid()*(extent-inner_extent)
    if integer:offset=offset.round()
    inner_origin=origin+offset;native=torch.stack((torch.cat((origin,origin+extent),-1),torch.cat((inner_origin,inner_origin+inner_extent),-1)),-2)
    return native/data['sizes'][:,None,None].repeat(1,1,1,2),native


def endpoint_loss(prediction,targets,valid):
    errors=(prediction[:,:,None]-targets).abs().mean((-1,-2));known=valid.any(-1);best=errors.masked_fill(~valid,torch.inf).amin(-1);per_source=torch.where(known,best,torch.zeros_like(best)).sum(-1)/known.sum(-1).clamp_min(1)
    return per_source.sum()/known.any(-1).sum().clamp_min(1)
