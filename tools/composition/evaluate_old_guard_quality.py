#!/usr/bin/env python3
"""Frozen original quality targets with explicit guard abstention and coverage."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from artifact_identity import sha
from export_unsplash_ranker import session
from audit_rehearsal_face_candidates import allowed
from subject_origin_projection import padded_heads
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from local_rank_distillation import family_weights


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','cache','control','serving']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.cache)!='2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031' or sha(a.control/'SHA256SUMS.json')!='941d64dca4c3f0cf5477fdd78ac332dee1a080818c825e44b7c8f17043bb6fc4':raise ValueError('frozen old inputs differ')
    c=torch.load(a.cache,map_location='cpu',weights_only=False);dev=[i for i,s in enumerate(c['sources']) if s['role']=='dev']
    if len(dev)!=25:raise ValueError('exact original25 DEV sources required')
    members=json.loads((a.control/'SHA256SUMS.json').read_text())
    def preserved(name):
        path=a.control/'relationship-rehearsal-oldgate'/name
        if sha(path)!=members[str(path.relative_to(a.control))]:raise ValueError('preserved quality evidence differs')
        return json.loads(path.read_text())
    baseline=preserved('summary.json')['baseline'];prior={(r['id'],r['query']):r for r in preserved('replay-contrastive-seed31-predictions.json')}
    manifest=json.loads((a.root/'manifest.json').read_text());sources={s['source_id']:s for s in manifest['images']}
    region_identity=json.loads((a.root/'regions/identity.json').read_text())
    if sha(a.root/'manifest.json')!=region_identity['manifest_sha256']:raise ValueError('region manifest changed')
    for path,identity in region_identity['inputs_sha256'].items():
        if sha(a.root/path)!=identity:raise ValueError('local detector output changed')
    local=json.loads((a.root/'regions/boxes.json').read_text());pose=json.loads((a.root/'regions/pose-heads.json').read_text())
    qualification=json.loads((a.serving/'qualification.json').read_text())
    for name in ['encoder','candidate-query']:
        if sha(a.serving/(name+'.onnx'))!=qualification['exports'][name]['sha256']:raise ValueError('serving graph changed')
    if '4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b' not in qualification['inputs'].values():raise ValueError('selected student differs')
    encoder=session(a.serving/'encoder.onnx');query=session(a.serving/'candidate-query.onnx')
    policies=['raw','faces-partial-only','faces-require-one','faces-pose-partial-only','faces-pose-require-one']
    rows=[];raw_changes=[];regrets={key:torch.full((25,22),float('nan')) for key in policies};offers={key:torch.zeros((25,22),dtype=torch.bool) for key in policies}
    for j,i in enumerate(dev):
        source=c['sources'][i];sid=source['id'];binding=sources[sid];w,h=source['native_size']
        if source['family_id']!=binding['family_id'] or binding['role']!='dev' or source['rgb_sha256']!=binding['rgb_sha256']:raise ValueError('source identity or role differs')
        heads=[]
        for r in local:
            if r['id']==sid:
                for b in r['boxes']:heads.append([max(0,math.floor(b[0]*w)),max(0,math.floor(b[1]*h)),min(w,math.ceil(b[2]*w)),min(h,math.ceil(b[3]*h))])
        with_pose=[*heads,*[r['native_box'] for r in pose if r['id']==sid and r['native_box'] is not None]]
        regions={'faces':padded_heads(w,h,heads,25),'faces-pose':padded_heads(w,h,with_pose,25)}
        f8,f16=encoder.run(None,dict(image=c['image'][i:i+1].numpy()))
        for q in range(22):
            boxes=c['boxes'][i:i+1,q//2].numpy();native=(c['boxes'][i,q//2]*torch.tensor([w,h,w,h])).round().int().tolist();p,r=map(int,c['ratios'][q//2].split(':'))
            scores=query.run(None,dict(f8=f8,f16=f16,boxes=boxes,source_aspect=np.array([w/h],np.float32),target_aspect=np.array([p/r],np.float32),mode=np.array([q%2],np.float32),content=c['content'][i:i+1].numpy()))[0][0]
            if not np.isfinite(scores).all():raise ValueError('nonfinite neural scores')
            target=c['targets'][i,q];eligible=c['eligible'][i,q];teacher=int(target.masked_fill(~eligible,-torch.inf).argmax())
            for policy in policies:
                mask=eligible.numpy().copy()
                if policy!='raw':
                    family='faces-pose' if policy.startswith('faces-pose') else 'faces';require=policy.endswith('require-one')
                    # Original banks use zero rectangles in unused tensor slots.
                    # Only the caller's existing eligible slots have crop geometry.
                    mask&=np.array([allowed(b,regions[family],require) if mask[k] else False for k,b in enumerate(native)])
                choice=int(np.where(mask,scores,-np.inf).argmax()) if mask.any() else None
                if policy=='raw' and choice!=prior[sid,q]['selected']:raw_changes.append(dict(id=sid,query=q,prior=prior[sid,q]['selected'],ort=choice))
                offered=choice is not None;offers[policy][j,q]=offered
                regret=float(target[teacher]-target[choice]) if offered else None
                if offered:regrets[policy][j,q]=regret
                rows.append(dict(id=sid,family_id=source['family_id'],query=q,ratio=c['ratios'][q//2],mode='minimal' if q%2==0 else 'maximal',policy=policy,query_valid=bool(c['query_valid'][i,q]),offered=offered,selected=choice,teacher=teacher,teacher_regret=regret,native_box=native[choice] if offered else None,face_regions=len(heads),pose_regions=len(with_pose)-len(heads),face_coverage_certified=False,source=source['source']))
    if raw_changes:raise ValueError('raw student choices differ from preserved full CUDA gate: '+json.dumps(raw_changes))
    summary=[];fw=family_weights(c['sources'],dev)
    for policy in policies:
        for scope,qs in [('all',list(range(22))),('essential',list(range(18))),('secondary',list(range(18,22)))]:
            valid=c['query_valid'][dev][:,qs];offered=offers[policy][:,qs]&valid;complete=bool(torch.equal(offered,valid))
            # Coverage is a ratio; normalize floating point family-weight mass.
            # Keep the original unnormalized float32 weights for quality scores.
            coverage=float(((offered.sum(-1).double()/valid.sum(-1))*fw.double()).sum()/fw.double().sum())
            present=offered.any(-1);indices=[dev[k] for k in range(25) if present[k]]
            conditional=None;matched_raw=None
            if indices:
                weights=family_weights(c['sources'],indices);denom=offered[present].sum(-1)
                values=regrets[policy][:,qs][present].masked_fill(~offered[present],0).sum(-1)/denom
                raw=regrets['raw'][:,qs][present].masked_fill(~offered[present],0).sum(-1)/denom
                conditional=float((values*weights).sum());matched_raw=float((raw*weights).sum())
            score=conditional if complete else None
            summary.append(dict(policy=policy,scope=scope,valid_requests=int(valid.sum()),offered_requests=int(offered.sum()),contains=int((valid&~offered).sum()),coverage_family_weighted=coverage,conditional_regret=conditional,raw_regret_on_same_offered_queries=matched_raw,conditional_sources=int(present.sum()),conditional_families=len({c['sources'][k]['family_id'] for k in indices}),full_panel_regret=score,frozen_original_baseline_regret=baseline[scope]['teacher_regret'],full_panel_nonregression_proven=complete and score<=baseline[scope]['teacher_regret']))
    output=a.root/'quality';output.mkdir(exist_ok=False)
    result=dict(summary=summary,raw_neural_choice_changes=len(raw_changes),decisions=len(rows),api_calls=0,test_inference=False,model_promoted=False,face_coverage_certified=False,scope='unchanged original target regret and family weighting; incomplete coverage has no full-panel score; local detector coverage is uncertified')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(output/'predictions.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    print(json.dumps([r for r in summary if r['scope']=='essential']))


if __name__=='__main__':main()
