"""Private weighted linear controls for the continuous paired crop contract."""
import torch
from nested_crop_head import decode


def fractions(target,ctx):
    outer,inner=target.unbind(-2);outer_extent=outer[...,2:]-outer[...,:2];inner_extent=inner[...,2:]-inner[...,:2];slack=ctx['sizes'][:,None]-outer_extent;origin=torch.where(slack>0,outer[...,:2]/slack.clamp_min(1),torch.full_like(slack,.5));scale=torch.where(ctx['maximum']>2,(inner_extent[...,0]/ctx['pairs'][None,:,0]-1)/(ctx['maximum']-2).clamp_min(1),torch.full_like(ctx['maximum'],.5));offset=(inner[...,:2]-outer[...,:2])/(outer_extent-inner_extent).clamp_min(1)
    return torch.cat((origin,scale[...,None],offset),-1)


def native(values,ctx):return decode(torch.logit(values.clamp(1e-6,1-1e-6)),ctx,integer=True)


def pca(raw,weights,dimensions=8):
    mean=(raw*weights[:,None]).sum(0);scale=((raw-mean).square()*weights[:,None]).sum(0).sqrt().clamp_min(.001);z=(raw-mean)/scale;weighted=z*weights.sqrt()[:,None];eigenvectors=torch.linalg.eigh(weighted@weighted.T);values=eigenvectors.eigenvalues[-dimensions:].flip(0).clamp_min(1e-12);vectors=eigenvectors.eigenvectors[:,-dimensions:].flip(1);components=weighted.T@vectors/values.sqrt()[None]
    return dict(mean=mean,scale=scale,components=components)


def uncompressed(raw,weights):
    mean=(raw*weights[:,None]).sum(0)
    scale=((raw-mean).square()*weights[:,None]).sum(0).sqrt().clamp_min(.001)
    return dict(mean=mean,scale=scale,components=torch.eye(raw.shape[-1],dtype=raw.dtype,device=raw.device))


def design(geometry,raw,transform):
    if transform is None:return geometry
    pc=((raw-transform['mean'])/transform['scale'])@transform['components'];aspect=geometry[:,1:2]
    return torch.cat((geometry,pc,pc*aspect,pc*aspect.square()),-1)


def ridge(x,y,weights,lambdas):
    mean=(x*weights[:,None]).sum(0);scale=((x-mean).square()*weights[:,None]).sum(0).sqrt().clamp_min(.001);ym=(y*weights[:,None]).sum(0);weighted=((x-mean)/scale)*weights.sqrt()[:,None];yw=(y-ym)*weights.sqrt()[:,None];gram=weighted@weighted.T;matrix=gram[None]+lambdas[:,None,None]*torch.eye(len(x),dtype=x.dtype,device=x.device)[None];alpha=torch.linalg.solve(matrix,yw[None].expand(len(lambdas),-1,-1));coefficients=weighted.T[None]@alpha
    return dict(mean=mean,scale=scale,target_mean=ym,coefficients=coefficients)


def predict(x,model,index=0):return ((x-model['mean'])/model['scale'])@model['coefficients'][index]+model['target_mean']
