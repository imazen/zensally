#!/usr/bin/env python3
"""Measure algebraic student fusion on the fixed candidate cache."""
import argparse
import json
from pathlib import Path
import torch
from artifact_identity import sha
from distill_attention_linear import Student


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fits','cache','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.cache)!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3':raise ValueError('fixed cache differs')
    cache=torch.load(a.cache,map_location='cpu',weights_only=False);rows=[]
    for fit in json.loads((a.fits/'summary.json').read_text())['fits']:
        path=a.fits/(fit['arm']+'-seed'+str(fit['seed']))/'selected.pt';checkpoint=torch.load(path,map_location='cpu',weights_only=False);state=checkpoint['model']
        model=Student(cache['scorer_state'],state['teacher'],state['mean'],state['std']);model.load_state_dict(state)
        with torch.no_grad():
            raw=model(cache['hidden'],None);fused=torch.nn.functional.linear(cache['hidden'],**checkpoint['fused'])[...,0]
            if not torch.isfinite(raw).all() or not torch.isfinite(fused).all():raise ValueError('nonfinite fusion')
            x=raw.masked_fill(~cache['eligible'],-torch.inf).argmax(-1);y=fused.masked_fill(~cache['eligible'],-torch.inf).argmax(-1)
            boxes=cache['query_boxes'];ix=x[...,None,None].expand(-1,-1,1,4);iy=y[...,None,None].expand(-1,-1,1,4)
            movement=(boxes.gather(2,ix)-boxes.gather(2,iy)).abs().amax()*128
            rows.append(dict(arm=fit['arm'],seed=fit['seed'],checkpoint_sha256=sha(path),max_abs_score_difference=float((raw-fused).abs().max()),eligible_choice_changes=int((x!=y).sum()),max_edge_movement_128_grid=float(movement),requests=x.numel(),within_existing_three_grid_gate=bool(movement<=3)))
    a.output.mkdir(parents=True,exist_ok=False);result=dict(rows=rows,api_calls=0,test_inference=False,model_promoted=False,limitation='CPU fixed-cache arithmetic comparison, not Rust export parity or runtime timing.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
