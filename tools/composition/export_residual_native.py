#!/usr/bin/env python3
"""Export first registered contrastive seed for runtime study, not model promotion."""
import argparse
import json
from pathlib import Path
import struct
import torch
from artifact_identity import sha


def bake(layers):
    return dict(schema_hash=0x5a535248303031,flags=0,scaler_mean=[0.]*96,scaler_scale=[1.]*96,layers=layers,zerobias_tau=0.,compressed=False,optimize=False)


def layer(weight,bias,activation):
    return dict(in_dim=weight.shape[1],out_dim=weight.shape[0],activation=activation,dtype='f32',weights=weight.flatten().tolist(),biases=bias.flatten().tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['cache','fits','output']:p.add_argument(key,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.cache)!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3':raise ValueError('fixed hidden cache differs')
    path=a.fits/'replay-contrastive-seed31/selected.pt';checkpoint=torch.load(path,map_location='cpu',weights_only=False);state=checkpoint['model'];cache=torch.load(a.cache,map_location='cpu',weights_only=False)
    if any(not torch.equal(state['base.'+k],v) for k,v in cache['scorer_state'].items()):raise ValueError('original base differs')
    first=state['probe.0.weight']/state['std'];bias=state['probe.0.bias']-state['probe.0.weight']@(state['mean']/state['std'])
    last=state['teacher']@state['probe.2.weight'];last_bias=state['teacher']@state['probe.2.bias']
    a.output.mkdir(parents=True,exist_ok=False)
    model=dict(format='zensally-compact-residual-v1',base=bake([layer(state['base.weight'],state['base.bias'],'identity')]),residual=bake([layer(first,bias,'relu'),layer(last,last_bias,'identity')]))
    (a.output/'model.json').write_text(json.dumps(model,indent=2)+'\n')
    hidden=cache['hidden'].contiguous().numpy().astype('<f4');fixture=a.output/'hidden.bin'
    with fixture.open('xb') as stream:stream.write(b'ZSRH001\0'+struct.pack('<I',hidden.shape[0]*hidden.shape[1]));stream.write(hidden.tobytes())
    offset=next(i for i,s in enumerate(cache['sources']) if s['role']=='dev')*22
    report=dict(checkpoint=str(path),checkpoint_sha256=sha(path),selected_seed=31,selection='first registered seed of contrastive-only arm; runtime study only',fixture_sha256=sha(fixture),fixture_bytes=fixture.stat().st_size,queries=hidden.shape[0]*hidden.shape[1],model_json_sha256=sha(a.output/'model.json'),benchmark_query_offset=offset,api_calls=0,test_inference=False,model_promoted=False)
    (a.output/'export.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
