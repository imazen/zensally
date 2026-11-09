#!/usr/bin/env python3
"""Compare finite-reference bank coverage without learning or new model API calls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import torch
from artifact_identity import sha
from subject_candidate_bank import generate
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from subject_geometry import audit,pixels,project

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','rank_cache','output']:p.add_argument(key,type=Path)
    a=p.parse_args()
    if sha(a.rank_cache)!='420b7fad28e64af4292f1c155d0447abd106b0fe190bbd07bb88d750d1df804d':raise ValueError('frozen native bank parent differs')
    frozen=torch.load(a.run/'set-cache/cache.pt',map_location='cpu',weights_only=False)
    original=torch.load(a.rank_cache,map_location='cpu',weights_only=False);lookup={s['id']:i for i,s in enumerate(original['sources'])}
    arms=['old-shared','mode128','mode256','oracle-roi128'];records=[];proposals=[]
    a.output.mkdir(parents=True,exist_ok=False)
    for index,source in enumerate(frozen['sources']):
        if not source['annotation_valid']:continue
        parent=lookup[source['id']];w,h=source['native_size'];scale=np.array([w,h,w,h],np.float64)
        result=json.loads((a.run/'real-terra'/(source['id'].replace(':','-')+'.json')).read_text());zones=json.loads((Path(result['entry'])/'zones.json').read_text())
        for request in zones['crops']:
            ratio=request['ratio'];mode=request['mode'];ri=frozen['ratios'].index(ratio);q=ri*2+(mode=='maximal')
            if not bool(frozen['reference_valid'][index,q].any()):continue
            good=[]
            for option in request['options']:
                if option['face_policy_check']!='whole_faces_or_excluded':continue
                raw=pixels(option['box'],w,h);exact=project(option['box'],w,h,ratio,mode)
                args=(zones['rois'],option['selected_subject_ids'],option['excluded_subject_ids'],w,h)
                if not audit(raw,*args,outward=False)['geometry_compatible'] or not audit(exact,*args)['geometry_compatible'] or max(np.abs(np.asarray(exact)-raw)/scale*128)>3:continue
                good.append((np.array(exact)/scale,option))
            if len(good)!=int(frozen['reference_valid'][index,q].sum()):raise ValueError('frozen weak-window admission differs')
            for slot,(target,_) in enumerate(good):
                if not np.array_equal(target.astype(np.float32),frozen['references'][index,q,slot].numpy()):raise ValueError('original weak reference geometry differs')
            old=original['native_boxes'][parent][ri];old=[box for j,box in enumerate(old) if bool(frozen['eligible'][index,q,j])]
            banks=dict(old_shared=old)
            p,r=map(int,ratio.split(':'))
            banks={'old-shared':old,'mode128':generate(w,h,(p,r),mode),
                'mode256':generate(w,h,(p,r),mode,capacity=256),
                'oracle-roi128':generate(w,h,(p,r),mode,rois=zones['rois'])}
            for arm in arms:
                bank=banks[arm];covered=[];winning=[]
                for j,box in enumerate(bank):
                    for target,option in good:
                        if max(np.abs(np.array(box)/scale-target)*128)>3:continue
                        if audit(list(box),zones['rois'],option['selected_subject_ids'],option['excluded_subject_ids'],w,h)['geometry_compatible']:
                            covered.append(j);winning.append(list(box));break
                if arm=='old-shared' and bool(covered)!=bool(frozen['positive'][index,q].any()):raise ValueError('old-bank coverage does not reproduce frozen oracle')
                records.append(dict(id=source['id'],role=source['role'],family_id=source['family_id'],arm=arm,ratio=ratio,mode=mode,
                    essential=max(p/r,r/p)<=3,covered=bool(covered),candidate_count=len(bank),matching_candidates=len(covered)))
                proposals.append(dict(id=source['id'],arm=arm,ratio=ratio,mode=mode,boxes=bank))
    groups=defaultdict(list)
    for row in records:
        for scope in ['all','essential' if row['essential'] else 'secondary']:
            groups[(row['arm'],scope,row['mode'])].append(row)
    summary=[]
    for (arm,scope,mode),rows in sorted(groups.items()):
        summary.append(dict(arm=arm,scope=scope,mode=mode,requests=len(rows),covered=sum(r['covered'] for r in rows),
            mean_candidates=float(np.mean([r['candidate_count'] for r in rows])),families=len({r['family_id'] for r in rows})))
    protocol=dict(reference_cache_sha256=sha(a.run/'set-cache/cache.pt'),native_parent_sha256=sha(a.rank_cache),source_roles_changed=False,
        training=False,model_api_calls=0,reference_matching_grid=3,all_old_bank_reference_coverage_reproduced=True,
        oracle_roi_warning='Same-call unreviewed teacher ROIs are an offline upper-bound input; no deployment localization or artistic-quality claim.',
        mode_policy='Research Maximal permits maximum-area choices; original shared-bank eligibility remains frozen.')
    (a.output/'summary.json').write_text(json.dumps(dict(protocol=protocol,summary=summary),indent=2)+'\n')
    (a.output/'queries.json').write_text(json.dumps(records,indent=2)+'\n')
    with (a.output/'candidates.jsonl').open('x') as stream:
        for row in proposals:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    print(json.dumps(summary))

if __name__=='__main__':main()

