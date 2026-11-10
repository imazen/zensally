"""Source-only exact crop/region relationships for private risk-head research."""
import hashlib
import json
import torch
from refinement_inputs import local_heads
from roi_geometry import person_views
from candidate_cut_head import geometry_features

ARMS=['geometry-regions','hidden-regions','hidden-regions-attention','hidden-permuted-regions-attention']
GROUPS=['faces','pose','upper','person']


def tensor_sha(value):return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def relationships(boxes,regions,width,height):
    """Positive native integer crop rows only; empty region groups are measured."""
    if boxes.ndim!=2 or boxes.shape[-1]!=4 or not bool((boxes[:,2:]>boxes[:,:2]).all()):raise ValueError('positive native crops required')
    if not regions:return torch.zeros((len(boxes),8),dtype=torch.float32)
    r=torch.tensor(regions,dtype=torch.int64)
    if not bool((r[:,:2]>=0).all() and (r[:,2:]>r[:,:2]).all() and (r[:,2:]<=torch.tensor([width,height])).all()):raise ValueError('positive in-source regions required')
    b=boxes[:,None,:];area=((r[:,2]-r[:,0])*(r[:,3]-r[:,1])).double();contains=(b[...,:2]<=r[:,:2]).all(-1)&(b[...,2:]>=r[:,2:]).all(-1);extent=(torch.minimum(b[...,2:],r[:,2:])-torch.maximum(b[...,:2],r[:,:2])).clamp_min(0);overlap=extent[...,0]*extent[...,1];cut=(overlap>0)&~contains;excluded=overlap==0;fraction=overlap.double()/area;relative=area/(width*height)
    values=torch.stack([torch.full((len(boxes),),float(torch.log1p(torch.tensor(len(regions),dtype=torch.float64)))),contains.double().mean(-1),cut.double().mean(-1),excluded.double().mean(-1),fraction.mean(-1),fraction.masked_fill(~cut,0).amax(-1),(relative*cut).amax(-1),(relative*contains).amax(-1)],-1)
    return values.float()


def proposal_bundle(contexts,faces,pose,original,miap,unsplash):
    measured,identities=local_heads(contexts,faces,pose,original)
    poses={'new':json.loads((pose/'pose-heads.json').read_text())['source_boxes'],'old':{s['id']:[] for s in contexts['old']['data']['sources']}}
    for row in json.loads((original/'regions/pose-heads.json').read_text()):poses['old'][row['id']].append(row['native_box'])
    result={}
    for cohort,ctx in contexts.items():
        root=miap if cohort=='new' else unsplash;rows={r['id']:r['result']['detections'] for r in json.loads((root/'full/tf-efficientdet_lite2/results.json').read_text())['rows']};groups=[]
        for s in ctx['data']['sources']:
            sid=s['id'];one=dict(faces=measured[cohort][sid]['faces'],pose=[list(b) for b in sorted(set(map(tuple,poses[cohort][sid])))],upper=[],person=[])
            for detection in rows[sid]:
                if any(v['name']=='person' and v['score']>=.3 for v in detection['categories']):
                    for name,box in person_views(detection['xywh'],s['working_size'],s['native_size']):one[name].append(box)
            groups.append({name:[list(b) for b in sorted(set(map(tuple,one[name])))] for name in GROUPS})
        donors=list(range(len(groups)))
        for role in ['fit','dev']:
            indices=sorted([i for i,s in enumerate(ctx['data']['sources']) if s['role']==role],key=lambda i:ctx['data']['sources'][i]['id'])
            if len(indices)==1:raise ValueError('nontrivial within-role donors required')
            for index,i in enumerate(indices):donors[i]=indices[(index+1)%len(indices)]
        result[cohort]=dict(sources=ctx['data']['sources'],groups=groups,donors=donors)
    return result,identities


def build(ctx,proposal,*,permuted=False):
    c=ctx['data'];sources=c['sources'];result=torch.zeros((*c['eligible'].shape,32),dtype=torch.float32)
    if proposal['sources']!=sources:raise ValueError('proposal/source identities differ')
    for i,s in enumerate(sources):
        j=proposal['donors'][i] if permuted else i;sender=sources[j];w,h=s['native_size'];sw,sh=sender['native_size'];valid=c['eligible'][i];boxes=ctx['native'][i].reshape(22,128,4)[valid];columns=[]
        for name in GROUPS:
            regions=[[b[0]*w//sw,b[1]*h//sh,(b[2]*w+sw-1)//sw,(b[3]*h+sh-1)//sh] for b in proposal['groups'][j][name]]
            columns.append(relationships(boxes,regions,w,h))
        result[i][valid]=torch.cat(columns,-1)
    if not torch.isfinite(result).all():raise ValueError('nonfinite region relationship features')
    return result


def features(c,real,permuted,arm):
    if arm=='geometry-regions':return torch.cat([geometry_features(c['sources'],c['ratios'],c['query_boxes']),real],-1)
    if arm=='hidden-regions':return torch.cat([c['hidden'],real],-1)
    if arm=='hidden-regions-attention':return torch.cat([c['hidden'],real,c['extras']],-1)
    if arm=='hidden-permuted-regions-attention':return torch.cat([c['hidden'],permuted,c['permuted_extras']],-1)
    raise ValueError('registered relation-risk arm required')


def head_applicability(contexts,bundle):
    counts={}
    for cohort,ctx in contexts.items():
        row=bundle[cohort]
        if row['sources']!=ctx['data']['sources']:raise ValueError('head applicability source identity differs')
        head=torch.tensor([bool(g['faces'] or g['pose']) for g in row['groups']],dtype=torch.bool)
        before=ctx['person'].clone();ctx['person']=before&head
        counts[cohort]=dict(person_sources=int(before.sum()),measured_head_sources=int(head.sum()),active_sources=int(ctx['person'].sum()))
    return counts
