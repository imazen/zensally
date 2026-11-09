"""Research source applicability and bounded soft cut-risk pair scoring."""
import torch


def person_count(source,image,detections):
    if source['id']!=image['id'] or source['rgb_sha256']!=image['rgb_sha256'] or source['working_size']!=[image['width'],image['height']]:raise ValueError('person detector source pixels differ')
    return sum(any(c['name']=='person' and c['score']>=.3 for c in d['categories']) for d in detections)


def adjusted_scores(scores,logits,applicable,strength):
    if strength not in [0.,.25,.5,1.,2.,4.,8.]:raise ValueError('registered risk strength required')
    if applicable.dtype!=torch.bool or applicable.shape!=(scores.shape[0],):raise ValueError('one applicability bit per source required')
    probability=logits.sigmoid().double()*applicable[:,None,None,None]
    penalty=probability[...,0,:,None]+probability[...,1,None,:]
    return scores-strength*penalty


def hard_logits(logits,applicable):
    return torch.where(applicable[:,None,None,None],logits,torch.full_like(logits,-1.))
