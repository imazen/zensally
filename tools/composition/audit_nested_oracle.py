#!/usr/bin/env python3
"""Diagnostic best frozen-target nested pairs; never a runtime selector."""
import argparse
import json
from pathlib import Path
import sys
import torch
from artifact_identity import sha
from nested_pair_selection import select,strict_nested
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from local_rank_distillation import family_weights


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('output',type=Path);a=p.parse_args();torch.set_num_threads(2)
    if sha(a.cache)!='2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031':raise ValueError('frozen original target cache differs')
    c=torch.load(a.cache,map_location='cpu',weights_only=False);rows=[];regrets=torch.full((len(c['sources']),22),float('nan'));offered=torch.zeros((len(c['sources']),22),dtype=torch.bool)
    for i,source in enumerate(c['sources']):
        if source['role'] not in ['fit','dev']:raise ValueError('TEST remains closed')
        w,h=source['native_size']
        for ri,ratio in enumerate(c['ratios']):
            q=ri*2;boxes=(c['boxes'][i,ri]*torch.tensor([w,h,w,h])).round().int().tolist();masks=c['eligible'][i,q:q+2].tolist();targets=[c['targets'][i,query].tolist() if c['query_valid'][i,query] else [0.]*128 for query in [q,q+1]];oi,ii=select(boxes,boxes,*targets,*masks,'joint-sum');teacher=[int(c['targets'][i,query].masked_fill(~c['eligible'][i,query],-torch.inf).argmax()) for query in [q,q+1]]
            for query,index,top in [(q,oi,teacher[0]),(q+1,ii,teacher[1])]:
                offered[i,query]=index is not None
                if index is not None:regrets[i,query]=c['targets'][i,query,top]-c['targets'][i,query,index]
            rows.append(dict(id=source['id'],family_id=source['family_id'],role=source['role'],ratio=ratio,ratio_index=ri,outer=oi,inner=ii,paired_targets_valid=bool(c['query_valid'][i,q:q+2].all()),independent_target_pair_strict=strict_nested(boxes[teacher[1]],boxes[teacher[0]]),outer_regret=float(regrets[i,q]),inner_regret=float(regrets[i,q+1]) if ii is not None else None))
    summary=[]
    for role in ['fit','dev']:
        indices=[i for i,s in enumerate(c['sources']) if s['role']==role];weights=family_weights(c['sources'],indices)
        for scope,qs in [('all',list(range(22))),('essential',list(range(18)))]:
            valid=c['query_valid'][indices][:,qs];complete=bool(offered[indices][:,qs][valid].all());value=float(((regrets[indices][:,qs].masked_fill(~valid,0).sum(-1)/valid.sum(-1))*weights).sum()) if complete else None
            chosen=[r for r in rows if r['role']==role and (scope=='all' or r['ratio_index']<9)]
            summary.append(dict(role=role,scope=scope,valid_endpoints=int(valid.sum()),pair_requests=len(chosen),strict_pairs_available=sum(r['inner'] is not None for r in chosen),both_targets_valid_pairs=sum(r['paired_targets_valid'] for r in chosen),independent_target_pairs_strict=sum(r['independent_target_pair_strict'] for r in chosen),best_target_sum_nested_regret=value))
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'summary.json').write_text(json.dumps(dict(summary=summary,cache_sha256=sha(a.cache),scope='oracle uses held-out target values only as a diagnostic; not learned output, runtime, artistic guarantee or exact floating point global optimum proof',api_calls=0,test_inference=False,model_promoted=False),indent=2)+'\n');(a.output/'pairs.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
