"""Research two-alternative aspect-conditioned crop regression, with native geometry."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class DirectCropHead(nn.Module):
    def __init__(self,visual=2048,hidden=32):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(visual+3,hidden),nn.ReLU())
        self.origins=nn.Linear(hidden,4)
        self.scales=nn.Linear(hidden,2)
        self.offer=nn.Linear(hidden,1)

    def forward(self,visual,query):
        count=query.shape[1]
        hidden=self.trunk(torch.cat((visual[:,None].expand(-1,count,-1),query),-1))
        return self.origins(hidden).reshape(len(visual),count,2,2),self.scales(hidden),self.offer(hidden)[...,0]

    def clear_inactive_grads(self,regression_active,maximal_active):
        # Absent tasks must not advance AdamW momentum or decay on their heads.
        for module,active in [(self.origins,regression_active),(self.scales,maximal_active)]:
            if not active:
                for parameter in module.parameters():parameter.grad=None


def geometry(sizes,ratios,*,device=None):
    sizes=torch.as_tensor(sizes,dtype=torch.float32,device=device)
    pairs=[]
    for ratio in ratios:
        p,q=map(int,ratio.split(':'));g=math.gcd(p,q);pairs.append([p//g,q//g])
    pairs=torch.tensor(pairs,dtype=torch.float32,device=device).repeat_interleave(2,dim=0)
    maximum=torch.floor(sizes[:,None,:]/pairs[None]).amin(-1)
    if not bool(torch.isfinite(sizes).all()) or not bool((maximum>=1).all()):raise ValueError('positive finite native exact-aspect dimensions required')
    mode=torch.arange(len(pairs),device=device)%2
    query=torch.stack(((sizes[:,0]/sizes[:,1]).log()[:,None].expand(-1,len(pairs)),
        (pairs[:,0]/pairs[:,1]).log()[None].expand(len(sizes),-1),mode[None].expand(len(sizes),-1)),dim=-1)
    return dict(sizes=sizes,pairs=pairs,maximum=maximum,mode=mode,query=query)


def decode(origins,scales,context,*,integer=False):
    maximum=context['maximum'][...,None]
    k=1+scales.sigmoid()*(maximum-1)
    k=torch.where(context['mode'][None,:,None]==0,maximum,k)
    if integer:k=k.round()
    extent=k[...,None]*context['pairs'][None,:,None]
    origin=origins.sigmoid()*(context['sizes'][:,None,None]-extent)
    if integer:origin=origin.round()
    native=torch.cat((origin,origin+extent),-1)
    return native/context['sizes'][:,None,None].repeat(1,1,1,2),native


def loss(model,prediction,offer_logits,target,valid,offer_target,offer_valid,mode):
    error=(prediction[...,None,:]-target[:,:,None]).abs().mean(-1)
    two=valid[...,1]
    matched=torch.minimum((error[...,0,0]+error[...,1,1])/2,(error[...,0,1]+error[...,1,0])/2)
    one=(error[...,0,0]+error[...,1,0])/2
    known=valid.any(-1)
    regression=(torch.where(two,matched,one)*known).sum()/known.sum().clamp_min(1)
    classification=(F.binary_cross_entropy_with_logits(offer_logits,offer_target,reduction='none')*offer_valid).sum()/offer_valid.sum().clamp_min(1)
    return regression+.1*classification,dict(regression_active=bool(known.any()),maximal_active=bool((known&(mode[None]==1)).any()))
