"""Experimental exact fractional region pooling; no trained composition weights."""
import torch
from torch import nn
from torch.nn import functional as F
from crop_ranker import CropRanker


def area_pool(features, rectangles):
    """Mean piecewise-constant feature cells inside normalized xyxy rectangles.

    Bilinear interpolation of the padded summed-area table integrates fractional
    cells exactly. Empty bands return zero. Work storage scales with feature maps
    and rectangle count, never rectangle-count times feature-map area.
    """
    with torch.autocast(device_type=features.device.type, enabled=False):
        f = features.float() if features.dtype in (torch.float16, torch.bfloat16) else features
        r = rectangles.to(f.dtype).clamp(0, 1)
        x0, y0, x1, y1 = r.unbind(-1)
        corners = torch.stack((torch.stack((x0,y0),-1),torch.stack((x1,y0),-1),
                               torch.stack((x0,y1),-1),torch.stack((x1,y1),-1)), -2)
        table = F.pad(f.cumsum(-1).cumsum(-2), (1,0,1,0))
        values = F.grid_sample(table, corners * 2 - 1, align_corners=True, padding_mode='border')
        total = values[...,0] - values[...,1] - values[...,2] + values[...,3]
        area = ((x1-x0)*(y1-y0)).clamp_min(0) * f.shape[-1] * f.shape[-2]
        return (total / area[:,None].clamp_min(1e-12)).transpose(1,2) * (area>0)[...,None]


def area_pool_matmul(features,rectangles):
    """Alternative GPU-training path: fractional overlap weights times features.

    Same piecewise-constant integral, with a denser temporary/retained matrix.
    Autocast may use tensor cores. Its numerical and memory tradeoffs require
    measurement; the default serving-oriented research path stays integral.
    """
    height,width=features.shape[-2:];r=rectangles.clamp(0,1)
    gx=torch.arange(width,device=features.device,dtype=r.dtype)/width
    gy=torch.arange(height,device=features.device,dtype=r.dtype)/height
    wx=(torch.minimum(gx[None,None]+1/width,r[...,2,None])-torch.maximum(gx[None,None],r[...,0,None])).clamp_min(0)
    wy=(torch.minimum(gy[None,None]+1/height,r[...,3,None])-torch.maximum(gy[None,None],r[...,1,None])).clamp_min(0)
    weights=(wy[...,None]*wx[...,None,:]).flatten(-2)
    weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
    return torch.bmm(weights.to(features.dtype),features.flatten(-2).transpose(1,2))


class RegionCropRanker(CropRanker):
    """Shared stride-8/16 encoder, pooled regions, rank and acceptability heads."""
    def __init__(self, width=24, channels=16, hidden=96, pool_backend='integral'):
        super().__init__(width,hidden)
        if pool_backend not in ['integral','matmul']:raise ValueError('unknown pooling backend')
        self.pool_backend=pool_backend
        self.project8=nn.Conv2d(width*2,channels,1)
        self.project16=nn.Conv2d(width*3,channels,1)
        # Nine retained bins, four discarded bands, eight boundary strips, global.
        self.scorer=nn.Sequential(nn.Linear(22*channels*2+14+44,hidden),nn.ReLU(),nn.Linear(hidden,2))
        starts=self.inside_grid-1/6
        self.register_buffer('bin_rectangles',torch.cat((starts,starts+1/3),-1))

    def encode(self,image):
        f8=self.enc2(self.enc1(self.stem(image)))
        f16=self.bottleneck(self.enc3(f8))
        return self.project8(f8),self.project16(f16)

    def score(self,features,boxes,source_aspect,target_aspect,mode,face_context,legal,valid_rect):
        lo,hi=boxes[...,:2],boxes[...,2:];size=hi-lo
        inside=torch.cat((lo[:,:,None]+size[:,:,None]*self.bin_rectangles[None,None,:,:2],
                          lo[:,:,None]+size[:,:,None]*self.bin_rectangles[None,None,:,2:]),-1)
        x0,y0,x1,y1=boxes.unbind(-1);z=torch.zeros_like(x0);one=torch.ones_like(x0)
        def rect(a,b,c,d):return torch.stack((a,b,c,d),-1)
        outside=torch.stack((rect(z,z,x0,one),rect(x1,z,one,one),rect(x0,z,x1,y0),rect(x0,y1,x1,one)),-2)
        dx=.02;dy=.02
        strips=torch.stack((rect(x0,y0,torch.minimum(x0+dx,x1),y1),rect(torch.maximum(x1-dx,x0),y0,x1,y1),
                            rect(x0,y0,x1,torch.minimum(y0+dy,y1)),rect(x0,torch.maximum(y1-dy,y0),x1,y1),
                            rect((x0-dx).clamp_min(0),y0,x0,y1),rect(x1,y0,(x1+dx).clamp_max(1),y1),
                            rect(x0,(y0-dy).clamp_min(0),x1,y0),rect(x0,y1,x1,(y1+dy).clamp_max(1))),-2)
        global_rect=rect(z,z,one,one)[...,None,:]
        regions=torch.cat((inside,outside,strips,global_rect),-2)
        area=(regions[...,2]-regions[...,0])*(regions[...,3]-regions[...,1])
        # Translate both xy corners, not just the lower corner.
        offset=valid_rect[:,:2].repeat(1,2)[:,None,None]
        scale=(valid_rect[:,2:]-valid_rect[:,:2]).repeat(1,2)[:,None,None]
        mapped=offset+regions*scale
        pool=area_pool if self.pool_backend=='integral' else area_pool_matmul
        descriptors=[pool(f,mapped.flatten(1,2)).reshape(boxes.shape[0],boxes.shape[1],-1) for f in features]
        geom=torch.stack((x0,y0,x1,y1,size[...,0]*size[...,1],source_aspect[:,None].expand_as(x0).log(),target_aspect.log(),mode),-1)
        vector=torch.cat((*descriptors,geom,face_context[:,None].expand(-1,boxes.shape[1],-1),area,(area>0).to(area.dtype)),-1)
        raw=self.scorer(vector)
        union=face_context[:,None,1:5]
        contains=((boxes[...,:2]<=union[...,:2]).all(-1)&(boxes[...,2:]>=union[...,2:]).all(-1))|(face_context[:,None,0]==0)
        valid=torch.isfinite(boxes).all(-1)&(boxes[...,:2]>=0).all(-1)&(boxes[...,2:]<=1).all(-1)&(boxes[...,2:]>boxes[...,:2]).all(-1)
        mask=legal&contains&valid&(face_context[:,None,5]==1)
        return raw[...,0].masked_fill(~mask,-torch.inf),raw[...,1].masked_fill(~mask,-torch.inf)
