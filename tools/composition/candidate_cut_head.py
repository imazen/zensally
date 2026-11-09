"""Research-only weak head-cut classifier; unknown candidates stay unlabeled."""
import torch
from torch import nn
from torch.nn import functional as F

ARMS=['geometry','hidden','hidden-pose','hidden-attention','permuted-attention']


def geometry_features(sources,ratios,boxes):
    count=len(sources);q=len(ratios)*2;aspect=torch.tensor([s['native_size'][0]/s['native_size'][1] for s in sources]).log()[:,None,None].expand(count,q,128)
    target=torch.tensor([int(r.split(':')[0])/int(r.split(':')[1]) for r in ratios for _ in range(2)]).log()[None,:,None].expand(count,q,128);mode=torch.arange(q).remainder(2).float()[None,:,None].expand(count,q,128);size=boxes[...,2:]-boxes[...,:2]
    return torch.cat([boxes,(size[...,0]*size[...,1])[...,None],aspect[...,None],target[...,None],mode[...,None]],-1)


def features(cache,arm,geometry=None):
    if arm not in ARMS:raise ValueError('registered classifier arm required')
    if arm=='geometry':return geometry if geometry is not None else geometry_features(cache['sources'],cache['ratios'],cache['query_boxes'])
    if arm=='hidden':return cache['hidden']
    extras=cache['permuted_extras'] if arm=='permuted-attention' else cache['extras']
    return torch.cat([cache['hidden'],extras[...,:16 if arm=='hidden-pose' else 32]],-1)


def balanced_source_loss(logits,positive,negative):
    """Per-source equal known-class means, then caller applies family weights."""
    if logits.shape!=positive.shape or logits.shape!=negative.shape or bool((positive&negative).any()):raise ValueError('disjoint matching known masks required')
    dims=tuple(range(1,logits.ndim));counts=torch.stack([positive.sum(dims),negative.sum(dims)],-1);valid=counts>0
    sums=torch.stack([(F.softplus(logits)*positive).sum(dims),(F.softplus(-logits)*negative).sum(dims)],-1)
    return ((sums/counts.clamp_min(1))*valid).sum(-1)/valid.sum(-1).clamp_min(1),valid.any(-1)


class CutHead(nn.Module):
    def __init__(self,mean,std):
        super().__init__();self.register_buffer('mean',mean.clone());self.register_buffer('std',std.clone());self.net=nn.Sequential(nn.Linear(len(mean),32),nn.ReLU(),nn.Linear(32,1))

    def forward(self,x):return self.net((x-self.mean)/self.std)[...,0]

    def compact(self,x):
        first,last=self.net[0],self.net[2];weight=first.weight/self.std;bias=first.bias-first.weight@(self.mean/self.std)
        return F.linear(torch.relu(F.linear(x,weight,bias)),last.weight,last.bias)[...,0]
