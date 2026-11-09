#!/usr/bin/env python3
"""Qualify embedded zenpredict residual output against its exact student."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from artifact_identity import sha
from distill_attention_linear import Student


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','cache','binary']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);export=json.loads((a.root/'export.json').read_text())
    if sha(a.cache)!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3' or sha(Path(export['checkpoint']))!=export['checkpoint_sha256'] or sha(a.root/'hidden.bin')!=export['fixture_sha256']:raise ValueError('frozen qualification input differs')
    output=a.root/'native-scores.bin';subprocess.run([str(a.binary),'gate',str(a.root/'hidden.bin'),str(output)],check=True)
    cache=torch.load(a.cache,map_location='cpu',weights_only=False);checkpoint=torch.load(export['checkpoint'],map_location='cpu',weights_only=False);s=checkpoint['model'];model=Student(cache['scorer_state'],s['teacher'],s['mean'],s['std'],nonlinear=True);model.load_state_dict(s)
    with torch.no_grad():reference=model(cache['hidden'],None).numpy()
    native=np.fromfile(output,dtype='<f4').reshape(reference.shape)
    if not np.isfinite(native).all():raise ValueError('nonfinite native scores')
    mask=cache['eligible'].numpy();left=np.where(mask,reference,-np.inf).argmax(-1);right=np.where(mask,native,-np.inf).argmax(-1);boxes=cache['query_boxes'].numpy();rows=[];maximum=0.
    for i,q in np.argwhere(left!=right):
        movement=float(np.abs(boxes[i,q,left[i,q]]-boxes[i,q,right[i,q]]).max()*128);maximum=max(maximum,movement)
        rows.append(dict(id=cache['sources'][i]['id'],query=int(q),reference=int(left[i,q]),native=int(right[i,q]),edge_movement_128_grid=movement))
    report=dict(binary_sha256=sha(a.binary),binary_bytes=a.binary.stat().st_size,queries=int(left.size),changed_choices=rows,max_abs_score_difference=float(np.abs(reference-native).max()),max_edge_movement_128_grid=maximum,gate_violations=sum(r['edge_movement_128_grid']>3 for r in rows),reference='same selected uncompressed Torch CPU student on frozen candidate hidden values; no encoder timing or Rust whole-pipeline claim',api_calls=0,test_inference=False,model_promoted=False)
    with (a.root/'qualification.json').open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report,changed_choices=len(rows))))
    if report['gate_violations']:raise SystemExit(1)


if __name__=='__main__':main()
