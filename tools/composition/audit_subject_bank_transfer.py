#!/usr/bin/env python3
"""Test fixed Terra-trained bank heads on cached Terra and Sol references."""
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


def family_vectors(rows,scope):
    groups=defaultdict(list)
    for row in rows:
        if row['essential'] and (scope=='all-requested' or row['reference_available']):groups[row['family_id']].append(row)
    return {family:np.array([np.mean([r[key] for r in group]) for key in ['reference_match','cut','offered']]) for family,group in groups.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','fits','transfer','initializer','output']:p.add_argument(key,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(a.initializer)!='c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df':raise ValueError('fixed initializer differs')
    initial=torch.load(a.initializer,map_location='cpu',weights_only=False)['model']
    rows=[];identities={};features=None;previous_input=None
    for bank in ['old-broad-maximal','hybrid']:
        paths=dict(terra=a.fits/bank/'cache/cache.pt',sol=a.transfer/bank/'sol-cache/cache.pt')
        panels={name:torch.load(path,map_location='cpu',weights_only=False) for name,path in paths.items()};left,right=panels['terra'],panels['sol']
        for key in ['image','content','query_boxes','eligible']:
            if not torch.equal(left[key],right[key]):raise ValueError('cross-teacher model inputs differ')
        source_ids=[(s['id'],s['family_id'],s['role']) for s in left['sources']]
        if source_ids!=[(s['id'],s['family_id'],s['role']) for s in right['sources']]:raise ValueError('cross-teacher source roles/order differ')
        if previous_input is not None and not torch.equal(previous_input,left['image']):raise ValueError('cross-bank images differ')
        previous_input=left['image']
        dev=[i for i,s in enumerate(left['sources']) if s['role']=='dev']
        image=left['image'][dev].cuda();content=left['content'][dev].cuda();boxes=left['query_boxes'][dev].cuda()
        aspect=torch.tensor([left['sources'][i]['native_size'][0]/left['sources'][i]['native_size'][1] for i in dev],device='cuda')
        model_paths=dict(initializer=a.initializer)
        reports=json.loads((a.fits/bank/'fits/summary.json').read_text())
        for seed in [31,137,911]:
            record=next(r for r in reports if r['arm']=='frozen-image' and r['seed']==seed)
            path=a.fits/bank/'fits'/f'frozen-image-seed{seed}'/'best.pt'
            if not record['old_nonregression'] or not record['replay_exact'] or sha(path)!=record['weights_sha256']:raise ValueError('fixed selected frozen-head gate or identity differs')
            model_paths['frozen-seed'+str(seed)]=path
        scores={}
        for name,path in model_paths.items():
            state=torch.load(path,map_location='cpu',weights_only=False)['model']
            if any(not torch.equal(initial[key],state[key]) for key in initial if not key.startswith('ranker.scorer')):raise ValueError('encoder differs')
            model=LocalCropRanker().cuda();model.load_state_dict(state);model.eval()
            with torch.no_grad():
                if features is None:features=encode(model,image,list(range(len(dev))))
                matrices=[]
                for q in range(22):
                    w,h=map(int,left['ratios'][q//2].split(':'))
                    raw=model.scores(features,boxes[:,q],aspect,aspect.new_full((len(dev),),w/h),aspect.new_full((len(dev),),q%2),content)
                    if not bool(torch.isfinite(raw).all()):raise ValueError('nonfinite cross-teacher scores')
                    matrices.append(raw.cpu().numpy())
                scores[name]=matrices
            del model
        identities[bank]=dict(caches={k:sha(v) for k,v in paths.items()},weights={k:sha(v) for k,v in model_paths.items()})
        for panel,cache in panels.items():
            for j,index in enumerate(dev):
                source=cache['sources'][index]
                if not source['annotation_valid']:continue
                zones=json.loads((a.run/'vision-cache'/source['reference_cache_key']/'zones.json').read_text());w,h=source['native_size']
                heads=[pixels(r['box'],w,h,outward=True) for r in zones['rois'] if r['kind']=='face_head']
                requested={(r['ratio'],r['mode']):r['action'] for r in zones['crops']}
                for q in range(22):
                    ratio=cache['ratios'][q//2];mode='minimal' if q%2==0 else 'maximal'
                    if (ratio,mode) not in requested:continue
                    native=np.rint(cache['query_boxes'][index,q].numpy()*np.array([w,h,w,h])).astype(int)
                    relation=[[face_relation(list(box),head) for head in heads] for box in native]
                    cut=np.array(['cut' in r for r in relation]);contains=np.array(['contained' in r for r in relation]);eligible=cache['eligible'][index,q].numpy()
                    if not np.array_equal(cut&eligible,cache['negative'][index,q].numpy()):raise ValueError('native geometry no longer reproduces cached negatives')
                    legal=eligible&~cut&contains
                    for model_name,matrices in scores.items():
                        values=matrices[q][j];raw=int(np.argmax(np.where(eligible,values,-np.inf)));guarded=int(np.argmax(np.where(legal,values,-np.inf))) if bool(legal.any()) else None
                        for policy,choice in [('raw',raw),('provided-head-guard',guarded)]:
                            rows.append(dict(bank=bank,panel=panel,model=model_name,policy=policy,id=source['id'],family_id=source['family_id'],query=q,
                                ratio=ratio,mode=mode,essential=q<18,teacher_action=requested[(ratio,mode)],reference_available=bool(cache['reference_valid'][index,q].any()),
                                selected=choice,reference_match=bool(choice is not None and cache['positive'][index,q,choice]),cut=bool(choice is not None and cut[choice]),offered=choice is not None))
    groups=defaultdict(list)
    for row in rows:groups[(row['bank'],row['panel'],row['policy'],row['model'])].append(row)
    summaries=[];paired=[]
    for (bank,panel,policy,model),values in groups.items():
        for scope in ['all-requested','accepted-references']:
            vectors=family_vectors(values,scope);matrix=np.array(list(vectors.values()))
            summaries.append(dict(bank=bank,panel=panel,policy=policy,model=model,scope=scope,families=len(vectors),
                requests=sum(r['essential'] and (scope=='all-requested' or r['reference_available']) for r in values),
                **{key:float(matrix[:,j].mean()) for j,key in enumerate(['reference_match','cut','offered'])}))
    for bank in ['old-broad-maximal','hybrid']:
        for panel in ['terra','sol']:
            for policy in ['raw','provided-head-guard']:
                for scope in ['all-requested','accepted-references']:
                    initial=family_vectors(groups[(bank,panel,policy,'initializer')],scope);families=sorted(initial)
                    learned=[family_vectors(groups[(bank,panel,policy,'frozen-seed'+str(seed))],scope) for seed in [31,137,911]]
                    if any(set(v)!=set(families) for v in learned):raise ValueError('paired family sets differ')
                    delta=np.mean([[v[f] for f in families] for v in learned],axis=0)-np.array([initial[f] for f in families])
                    indices=np.random.default_rng(31).integers(0,len(families),size=(5000,len(families)));sample=delta[indices].mean(1)
                    paired.append(dict(bank=bank,panel=panel,policy=policy,scope=scope,families=len(families),
                        **{key:dict(delta=float(delta[:,j].mean()),ci95=np.quantile(sample[:,j],[.025,.975]).tolist()) for j,key in enumerate(['reference_match','cut','offered'])}))
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'predictions.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    result=dict(summary=summaries,paired=paired,identities=identities,paid_api_calls=0,model_promoted=False,test_inference=False,
        cross_teacher_model_inputs_bit_identical=True,checkpoint_selection_changed=False,
        limitation='Cached Sol disagreement check on reused DEV images, not an independent human gold or new test. Provided-head masks do not certify actual-face coverage.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(paired=[r for r in paired if r['panel']=='sol' and r['policy']=='raw' and r['scope']=='accepted-references'],model_promoted=False)))


if __name__=='__main__':main()
