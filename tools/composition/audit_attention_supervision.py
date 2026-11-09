#!/usr/bin/env python3
"""Audit cached weak labels and attention support without fitting or API calls."""
import argparse
import json
from pathlib import Path
import torch
from artifact_identity import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();torch.set_num_threads(2)
    identity=sha(args.cache)
    if identity!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3':raise ValueError('registered attention cache differs')
    cache=torch.load(args.cache,map_location='cpu',weights_only=False)
    rows=[]
    for role in ['fit','dev']:
        indices=[i for i,s in enumerate(cache['sources']) if s['role']==role]
        for scope in ['all-ratios','essential-1:3-through-3:1']:
            queries=[q for q in range(22) if scope=='all-ratios' or 1/3<=int(cache['ratios'][q//2].split(':')[0])/int(cache['ratios'][q//2].split(':')[1])<=3]
            positive=cache['positive'][indices][:,queries].bool()
            negative=cache['negative'][indices][:,queries].bool()
            eligible=cache['eligible'][indices][:,queries].bool()
            valid=cache['reference_valid'][indices][:,queries].any(-1).bool()
            scores=cache['initial_scores'][indices][:,queries].masked_fill(~eligible,-torch.inf)
            choice=scores.argmax(-1,keepdim=True)
            extras=cache['extras'][indices][:,queries]
            support=[]
            for name,start,end in [('pose',0,16),('gaze',16,24),('scanpath',24,32)]:
                values=extras[...,start:end]
                # Only eligible candidates participate; constants across a query
                # cannot change its ordering through this linear extension.
                low=values.masked_fill(~eligible[...,None],torch.inf).amin(-2)
                high=values.masked_fill(~eligible[...,None],-torch.inf).amax(-2)
                varies=(high>low).any(-1)&eligible.any(-1)
                nonzero=((values!=0)&eligible[...,None]).any(-1).any(-1)
                support.append(dict(block=name,sources_with_nonzero=int(nonzero.any(-1).sum()),
                    valid_queries_with_nonzero=int((nonzero&valid).sum()),valid_queries_with_candidate_variation=int((varies&valid).sum()),
                    varying_columns=[cache['feature_names'][start+j] for j in range(end-start) if bool(((high[...,j]>low[...,j])&valid).any())]))
            rows.append(dict(role=role,scope=scope,sources=len(indices),queries=len(indices)*len(queries),
                valid_reference_queries=int(valid.sum()),eligible_candidates=int(eligible.sum()),
                positive_candidates=int((positive&eligible).sum()),explicit_negative_candidates=int((negative&eligible).sum()),
                positive_negative_overlap=int((positive&negative&eligible).sum()),
                positive_negative_overlap_queries=int((positive&negative&eligible).any(-1).sum()),
                valid_queries_without_eligible_positive=int((valid&~(positive&eligible).any(-1)).sum()),
                selected_explicit_head_cut=int((negative.gather(-1,choice)[...,0]&valid).sum()),
                selected_reference_positive=int((positive.gather(-1,choice)[...,0]&valid).sum()),support=support))
    result=dict(cache_sha256=identity,rows=rows,api_calls=0,trained=False,test_inference=False,
        limitation='Cached weak references and nominated-head negatives, not human aesthetic labels or exhaustive face coverage. Feature variation is availability, not predictive benefit.')
    args.output.mkdir(exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
