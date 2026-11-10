#!/usr/bin/env python3
"""Cache explicitly admitted person-auxiliary targets using owned letterboxing."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from artifact_identity import sha
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from person_relationships import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['acquired','review','resizer','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    for path,digest in [(a.acquired/'SHA256SUMS.json','93f63d69180d82f197e34a733005118de260eda728e18998ced3194959581bb1'),
                        (a.review/'SHA256SUMS.json','75a75ee2df2c85ecac6b18cf6c875d3bfb0c6d625bf26542efc57fcbb4d2bca9'),
                        (a.review/'source-mask.json','e443d102d0069d89197c81e4c0032eff2335e4d9b85f01503a2248ed48f88f80'),
                        (a.resizer,'ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041')]:
        if sha(path)!=digest:raise ValueError('frozen auxiliary source/preprocessing identity differs')
    manifest=json.loads((a.acquired/'manifest.json').read_text())
    members={r['path']:r['sha256'] for r in json.loads((a.acquired/'SHA256SUMS.json').read_text())}
    if sha(a.acquired/'manifest.json')!=members['manifest.json']:raise ValueError('acquisition rows differ')
    mask={r['id']:r for r in json.loads((a.review/'source-mask.json').read_text())}
    rows=[r for r in manifest['rows'] if mask[r['id']]['person_auxiliary_source_eligible']]
    if Counter(r['source']['role'] for r in rows)!=Counter(fit=275,dev=91) or len({r['source']['family_id'] for r in rows})!=366:
        raise ValueError('fixed366 candidate sources required')
    a.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(schema='person-relationship-transfer-input-v1',acquisition_packet_sha256=sha(a.acquired/'SHA256SUMS.json'),
        review_packet_sha256=sha(a.review/'SHA256SUMS.json'),resizer_sha256=sha(a.resizer),
        build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        sources=366,fit_sources=275,development_sources=91,person_auxiliary_targets_admitted=True,
        artistic_labels_admitted=False,face_coverage_certified=False,test_pixels_opened=False,paid_model_api_calls=0)
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    images=[];sources=[];content=[];targets=[];valid=[]
    for row in rows:
        key=row['id'].split(':')[1];path=a.acquired/key/'preview.rgb';w,h=row['geometry']['width'],row['geometry']['height']
        if sha(path)!=row['preview.rgb_sha256']:raise ValueError('owned source pixels differ')
        tw,th=max(1,round(w*256/max(w,h))),max(1,round(h*256/max(w,h)))
        with path.open('rb') as stream:
            raw=subprocess.check_output([str(a.resizer),str(w),str(h),str(w*3),','.join(map(str,[0,0,w,h,tw,th]))],stdin=stream)
        if len(raw)!=tw*th*3:raise ValueError('owned resized length differs')
        image=np.zeros((256,256,3),np.uint8);ox,oy=(256-tw)//2,(256-th)//2
        image[oy:oy+th,ox:ox+tw]=np.frombuffer(raw,np.uint8).reshape(th,tw,3)
        tensor=a.output/(key+'.rgb');tensor.write_bytes(image.tobytes())
        target,known=statistics(row['boxes'])
        images.append(image);targets.append(target);valid.append(known);content.append([ox/256,oy/256,(ox+tw)/256,(oy+th)/256])
        sources.append(dict(id=row['id'],role=row['source']['role'],family_id=row['source']['family_id'],
            native_size=row['header']['size'],rgb_sha256=row['preview.rgb_sha256'],input_sha256=sha(tensor),
            individual_records=sum(not b['flags']['IsGroupOf'] for b in row['boxes']),group_records=sum(b['flags']['IsGroupOf'] for b in row['boxes']),
            training_eligible=row['source']['role']=='fit',face_coverage_certified=False))
        (a.output/'progress.json').write_text(json.dumps(dict(pid=os.getpid(),completed=len(sources),required=366))+'\n')
    payload=dict(image=torch.from_numpy(np.stack(images)),content=torch.tensor(content),target=torch.from_numpy(np.stack(targets)),
        valid=torch.from_numpy(np.stack(valid)),sources=sources,protocol=protocol)
    torch.save(payload,a.output/'cache.pt')
    total=sum(f.stat().st_size for f in a.output.rglob('*') if f.is_file())
    if total>1024**3:raise ValueError('1GiB auxiliary cache budget exceeded')
    (a.output/'progress.json').write_text(json.dumps(dict(pid=None,stage='complete',completed=366,required=366))+'\n')
    (a.output/'summary.json').write_text(json.dumps(dict(sources=366,fit=275,development=91,cache_sha256=sha(a.output/'cache.pt'),payload_bytes=total),indent=2)+'\n')
    print((a.output/'summary.json').read_text())


if __name__=='__main__':main()
