"""Bounded corpus fitting primitives. Canonical holdouts never supply gradients."""
import hashlib,json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from crop_regions import RegionCropRanker

MAX_CANDIDATES=128
MAX_PAIRS=128

def load_manifest(path):
    data=json.loads(Path(path).read_text());families={};ids=set()
    if data['schema']!=1 or data['supervision']!='noisy_vlm_rendered_pixels':raise ValueError('admitted weak-label schema required')
    for s in data['sources']:
        if s['id'] in ids:raise ValueError('duplicate source ID')
        ids.add(s['id']);f=s['family_id'];use=s['use']
        if f in families and families[f]!=use:raise ValueError('source family crosses fitting/evaluation roles')
        families[f]=use
        if use in ['fit','dev'] and (s['bucket']!='train' or s['historical_eval_family']):raise ValueError('held-out source cannot enter fitting/development')
        if use=='fit' and not s['training_eligible']:raise ValueError('fitting source not admitted')
        if use!='fit' and s['training_eligible']:raise ValueError('held-out source marked training eligible')
        if s['human_rated'] or s['face_safety_certified']:raise ValueError('machine labels must not assert human quality/face certification')
    return data

class CropDataset(Dataset):
    def __init__(self,manifest,use,*,training=False):
        if training and use!='fit':raise ValueError('only fit sources may supply gradients')
        self.sources=[s for s in load_manifest(manifest)['sources'] if s['use']==use]
        self.features=[];self.images=OrderedDict();self.image_bytes=0;self.cache_limit=64*1024*1024
        if not self.sources:raise ValueError('requested admitted split is empty')
        for s in self.sources:
            n=len(s['candidates'])
            if not 1<=n<=MAX_CANDIDATES or len(s['pairs'])>MAX_PAIRS:raise ValueError('candidate/pair count exceeds fixed fitting bounds')
            if not any(c['weight']>0 for c in s['candidates']):raise ValueError('source has no admitted labels')
            for c in s['candidates']:
                x0,y0,x1,y1=c['box']
                if not 0<=x0<x1<=1 or not 0<=y0<y1<=1:raise ValueError('invalid normalized crop')
            for pair in s['pairs']:
                i,j=pair['preferred'],pair['other']
                if not 0<=i<n or not 0<=j<n or i==j or s['candidates'][i]['task_id']!=s['candidates'][j]['task_id']:raise ValueError('ranking pair crosses a task')
            b=Path(s['tensor_path']).read_bytes()
            if len(b)!=256*256*3 or hashlib.sha256(b).hexdigest()!=s['tensor_sha256']:raise ValueError('frozen tensor bytes changed')
            self.features.append(self.pack(s))
    def __len__(self):return len(self.sources)
    @staticmethod
    def pack(s):
        boxes=torch.tensor([[0.,0.,1.,1.]]*MAX_CANDIDATES);aspects=torch.ones(MAX_CANDIDATES);modes=torch.zeros(MAX_CANDIDATES);targets=torch.zeros(MAX_CANDIDATES);weights=torch.zeros(MAX_CANDIDATES)
        for i,c in enumerate(s['candidates']):
            boxes[i]=torch.tensor(c['box']);aspects[i]=c['aspect'];modes[i]=c['mode'];targets[i]=float(c['acceptable']);weights[i]=c['weight']
        pairs=torch.zeros(MAX_PAIRS,2,dtype=torch.long);pairweights=torch.zeros(MAX_PAIRS)
        for i,p in enumerate(s['pairs']):pairs[i]=torch.tensor([p['preferred'],p['other']]);pairweights[i]=p['weight']
        faces=torch.tensor(s['face_context'],dtype=torch.float32)
        if faces[5]!=0:raise ValueError('weak labels cannot certify face coverage')
        return {'boxes':boxes,'source_aspect':torch.tensor(s['width']/s['height']),'target_aspect':aspects,'mode':modes,'face_context':faces,'valid_rect':torch.tensor(s['valid_rect']),'targets':targets,'weights':weights,'pairs':pairs,'pairweights':pairweights}
    def __getitem__(self,index):
        if index in self.images:pixels=self.images.pop(index);self.images[index]=pixels
        else:
            b=bytearray(Path(self.sources[index]['tensor_path']).read_bytes())
            if len(b)!=256*256*3 or hashlib.sha256(b).hexdigest()!=self.sources[index]['tensor_sha256']:raise ValueError('frozen tensor bytes changed during fitting')
            pixels=torch.from_numpy(np.frombuffer(b,dtype=np.uint8).reshape(256,256,3)).permute(2,0,1).float()/255
            size=pixels.numel()*pixels.element_size()
            while self.image_bytes+size>self.cache_limit:
                _,old=self.images.popitem(last=False);self.image_bytes-=old.numel()*old.element_size()
            self.images[index]=pixels;self.image_bytes+=size
        return {**self.features[index],'image':pixels,'index':index}

class TrainingModel(nn.Module):
    def __init__(self):
        super().__init__();self.ranker=RegionCropRanker()
    def forward(self,image,boxes,source_aspect,target_aspect,mode,face_context,valid_rect):
        return self.ranker.raw_scores(self.ranker.encode(image),boxes,source_aspect,target_aspect,mode,face_context,valid_rect)

def loss(raw,batch):
    """Equal source contribution; reject-all trains BCE without invented winners."""
    bce=F.binary_cross_entropy_with_logits(raw[...,1].float(),batch['targets'],reduction='none')
    weight=batch['weights'];accept=(bce*weight).sum(-1)/weight.sum(-1).clamp_min(1e-12)
    scores=raw[...,0].float();pairs=batch['pairs'];pw=batch['pairweights']
    delta=scores.gather(1,pairs[...,0])-scores.gather(1,pairs[...,1])
    rank=(F.softplus(-delta)*pw).sum(-1)/pw.sum(-1).clamp_min(1e-12)
    return (accept+rank).mean(),accept.mean().detach(),rank.mean().detach()

MODEL_ARGS=['image','boxes','source_aspect','target_aspect','mode','face_context','valid_rect']
