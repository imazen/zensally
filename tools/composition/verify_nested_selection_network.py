#!/usr/bin/env python3
"""Full ORT encoder/query replay of all frozen-score nested-pair policies."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from artifact_identity import sha
from export_unsplash_ranker import session
from nested_pair_selection import select,strict_nested


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['old','new','audit','serving']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.old)!='2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031' or sha(a.new)!='bfd84ff0ea48a7c3a2d2d32082f3fd8a9bd4a6c59b576899a80c4a7812f873c9':raise ValueError('frozen image caches differ')
    qualification=json.loads((a.serving/'qualification.json').read_text());protocol=json.loads((a.audit/'protocol.json').read_text())
    if protocol['checkpoint_sha256']!='4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b' or protocol['checkpoint_sha256'] not in qualification['inputs'].values():raise ValueError('selected checkpoint lineage differs')
    for name in ['encoder','candidate-query']:
        if sha(a.serving/(name+'.onnx'))!=qualification['exports'][name]['sha256']:raise ValueError('fixed serving graph changed')
    records=json.loads((a.audit/'pairs.json').read_text());expected={(r['cohort'],r['id'],r['ratio_index'],r['policy']):r for r in records};metadata=json.loads((a.audit/'sources.json').read_text());policies=protocol['policies'];encoder=session(a.serving/'encoder.onnx');query=session(a.serving/'candidate-query.onnx');rows=[];images=0;queries=0
    for cohort,path in [('old',a.old),('new',a.new)]:
        c=torch.load(path,map_location='cpu',weights_only=False)
        if [(s['id'],s['role'],s['family_id']) for s in c['sources']]!=[(s['id'],s['role'],s['family_id']) for s in metadata[cohort]]:raise ValueError('source identity or role differs')
        for i,source in enumerate(c['sources']):
            if source['role'] not in ['fit','dev']:raise ValueError('TEST remains closed')
            w,h=source['native_size'];scale=torch.tensor([w,h,w,h]);f8,f16=encoder.run(None,dict(image=c['image'][i:i+1].numpy()));images+=1
            for ri,ratio in enumerate(c['ratios']):
                p,r=map(int,ratio.split(':'));scores=[];boxes=[];native=[];masks=[]
                for mode in [0,1]:
                    q=ri*2+mode;box=c['boxes'][i:i+1,ri] if cohort=='old' else c['query_boxes'][i:i+1,q]
                    output=query.run(None,dict(f8=f8,f16=f16,boxes=box.numpy(),source_aspect=np.array([w/h],np.float32),target_aspect=np.array([p/r],np.float32),mode=np.array([mode],np.float32),content=c['content'][i:i+1].numpy()))[0][0]
                    if not np.isfinite(output).all():raise ValueError('nonfinite neural score')
                    scores.append(output.tolist());boxes.append(box);native.append((box[0]*scale).round().int().tolist());masks.append(c['eligible'][i,q].tolist());queries+=1
                for policy in policies:
                    ref=expected[cohort,source['id'],ri,policy];oi,ii=select(*native,*scores,*masks,policy);same_presence=(ii is None)==(ref['inner'] is None);outer_edge=float((np.abs(np.asarray(native[0][oi])-ref['outer_box'])/np.array([w,h,w,h])*128).max());inner_edge=float((np.abs(np.asarray(native[1][ii])-ref['inner_box'])/np.array([w,h,w,h])*128).max()) if ii is not None and ref['inner'] is not None else 0.
                    if policy!='independent' and ii is not None and not strict_nested(native[1][ii],native[0][oi]):raise ValueError('serving pair violates nesting')
                    rows.append(dict(cohort=cohort,id=source['id'],role=source['role'],ratio=ratio,policy=policy,reference_outer=ref['outer'],reference_inner=ref['inner'],ort_outer=oi,ort_inner=ii,outer_changed=oi!=ref['outer'],inner_changed=ii!=ref['inner'],same_presence=same_presence,outer_edge_movement_128_grid=outer_edge,inner_edge_movement_128_grid=inner_edge,gate_passed=same_presence and max(outer_edge,inner_edge)<=3))
        del c
    if images!=191 or queries!=4202 or len(rows)!=len(expected):raise ValueError('complete fixed inference panel required')
    result=dict(rows=rows,images=images,neural_queries=queries,pair_decisions=len(rows),outer_index_changes=sum(r['outer_changed'] for r in rows),inner_index_changes=sum(r['inner_changed'] for r in rows),presence_changes=sum(not r['same_presence'] for r in rows),gate_violations=sum(not r['gate_passed'] for r in rows),max_edge_movement_128_grid=max(max(r['outer_edge_movement_128_grid'],r['inner_edge_movement_128_grid']) for r in rows),audit_pairs_sha256=sha(a.audit/'pairs.json'),serving_exports={k:qualification['exports'][k] for k in ['encoder','candidate-query']},api_calls=0,test_inference=False,model_promoted=False,scope='full ORT gather encoder/query plus existing native-bank selectors; excludes decode/resize, detector inference and artistic/face certification')
    with (a.audit/'full-neural-check.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','serving_exports']}))
    if result['gate_violations']:raise SystemExit(1)


if __name__=='__main__':main()
