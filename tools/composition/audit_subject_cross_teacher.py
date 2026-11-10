#!/usr/bin/env python3
"""Cross-evaluate fixed heads and an explicit supplied-head guard, without APIs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import torch
from artifact_identity import sha
from fit_unsplash_ranks import encode
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from local_rank_distillation import LocalCropRanker
from subject_geometry import pixels
from subject_policy import face_relation

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','rank_cache','initializer','output']:p.add_argument(key,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(a.rank_cache)!='420b7fad28e64af4292f1c155d0447abd106b0fe190bbd07bb88d750d1df804d' or sha(a.initializer)!='c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df':raise ValueError('frozen model/native parents differ')
    panels={name:torch.load(a.run/folder/'cache.pt',map_location='cpu',weights_only=False) for name,folder in [('terra','set-cache'),('sol','sol-set-cache')]}
    left,right=panels['terra'],panels['sol']
    for key in ['image','content','boxes','eligible']:
        if not torch.equal(left[key],right[key]):raise ValueError('cross-panel model inputs differ')
    if [(s['id'],s['role'],s['family_id']) for s in left['sources']]!=[(s['id'],s['role'],s['family_id']) for s in right['sources']]:raise ValueError('cross-panel source/role order differs')
    native=torch.load(a.rank_cache,map_location='cpu',weights_only=False);native_lookup={s['id']:i for i,s in enumerate(native['sources'])}
    models=dict(initializer=a.initializer);selected={}
    for panel,folder in [('terra','fits'),('sol','sol-fits')]:
        reports=json.loads((a.run/folder/'summary.json').read_text())
        row=min((r for r in reports if r['old_nonregression']),key=lambda r:(r['new']['essential']['explicit_face_cut'],-r['new']['essential']['reference_match'],r['old']['essential']['teacher_regret'],r['seed']))
        selected[panel]=row;models[panel]=a.run/folder/(row['arm']+'-seed'+str(row['seed']))/'best.pt'
    dev=[i for i,s in enumerate(left['sources']) if s['role']=='dev'];image=left['image'][dev].cuda();content=left['content'][dev].cuda();boxes=left['boxes'][dev].cuda()
    aspect=torch.tensor([left['sources'][i]['native_size'][0]/left['sources'][i]['native_size'][1] for i in dev],device='cuda')
    initial=torch.load(a.initializer,map_location='cpu',weights_only=False)['model'];scores={};features=None
    for name,path in models.items():
        state=torch.load(path,map_location='cpu',weights_only=False)['model']
        for key in initial:
            if not key.startswith('ranker.scorer') and not torch.equal(initial[key],state[key]):raise ValueError('frozen encoders are not identical')
        model=LocalCropRanker().cuda();model.load_state_dict(state);model.eval()
        with torch.no_grad():
            if features is None:features=encode(model,image,list(range(len(dev))))
            matrices=[]
            for q in range(22):
                w,h=map(int,left['ratios'][q//2].split(':'))
                raw=model.scores(features,boxes[:,q//2],aspect,aspect.new_full((len(dev),),w/h),aspect.new_full((len(dev),),q%2),content)
                if not bool(torch.isfinite(raw).all()):raise ValueError('nonfinite cross-teacher scores')
                matrices.append(raw.cpu().numpy())
            scores[name]=matrices
        del model
    rows=[]
    for panel,cache in panels.items():
        for j,index in enumerate(dev):
            source=cache['sources'][index]
            if not source['annotation_valid']:continue
            entry=a.run/'vision-cache'/source['reference_cache_key'];zones=json.loads((entry/'zones.json').read_text());sw,sh=source['native_size']
            faces=[pixels(r['box'],sw,sh,outward=True) for r in zones['rois'] if r['kind']=='face_head']
            conditions={(r['ratio'],r['mode']):r['action'] for r in zones['crops']}
            for q in range(22):
                ratio=cache['ratios'][q//2];mode='minimal' if q%2==0 else 'maximal'
                if (ratio,mode) not in conditions:continue
                bank=native['native_boxes'][native_lookup[source['id']]][q//2];count=len(bank);eligible=cache['eligible'][index,q,:count].numpy()
                relations=[[face_relation(list(box),face) for face in faces] for box in bank]
                contains=np.array(['contained' in r for r in relations]);cut=np.array(['cut' in r for r in relations])
                if not np.array_equal(cut&eligible,cache['negative'][index,q,:count].numpy()):raise ValueError('frozen explicit face-cut mask differs')
                legal=eligible&~cut&contains
                for model_name,values in scores.items():
                    value=values[q][j,:count];raw=int(np.argmax(np.where(eligible,value,-np.inf)))
                    guarded=int(np.argmax(np.where(legal,value,-np.inf))) if bool(legal.any()) else None
                    for policy,choice in [('raw',raw),('supplied-head-guard',guarded)]:
                        rows.append(dict(panel=panel,model=model_name,policy=policy,id=source['id'],family_id=source['family_id'],query=q,ratio=ratio,mode=mode,
                            essential=q<18,source_has_nominated_heads=bool(faces),offered=choice is not None,
                            cut=bool(choice is not None and cut[choice]),retains_head=bool(choice is not None and contains[choice]),
                            reference_available=bool(cache['reference_valid'][index,q].any()),reference_match=bool(choice is not None and cache['positive'][index,q,choice]),
                            selected=choice,teacher_action=conditions[(ratio,mode)]))
    summaries=[]
    for panel in panels:
        for model_name in models:
            for policy in ['raw','supplied-head-guard']:
                chosen=[r for r in rows if r['panel']==panel and r['model']==model_name and r['policy']==policy and r['essential']]
                groups=defaultdict(list)
                for row in chosen:groups[row['family_id']].append(row)
                result=dict(panel=panel,model=model_name,policy=policy,families=len(groups),requests=len(chosen))
                for key in ['offered','cut','retains_head']:
                    result[key]=float(np.mean([np.mean([r[key] for r in group]) for group in groups.values()]))
                known=defaultdict(list)
                for row in chosen:
                    if row['reference_available']:known[row['family_id']].append(row)
                result['reference_families']=len(known);result['reference_match']=float(np.mean([np.mean([r['reference_match'] for r in group]) for group in known.values()]))
                summaries.append(result)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(dict(summary=summaries,selected=selected,weights_sha256={k:sha(v) for k,v in models.items()},
        input_panels_bit_identical=True,paid_model_api_calls=0,model_promoted=False,
        guard_scope='Provided-head experiment: require at least one nominated whole head and no partial nominated heads. This is not complete actual-face detection or a shipping guard change.'),indent=2)+'\n')
    (a.output/'predictions.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(summaries))

if __name__=='__main__':main()

