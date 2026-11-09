#!/usr/bin/env python3
"""Verify native nested-pair geometry and independently read every owned PNG."""
import argparse
import json
from pathlib import Path
from PIL import Image
from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();variants=[a.root,a.root/'tighter'];pairs=0;images={};raw_nested=0
    for root in variants:
        selection=json.loads((root/'selection.json').read_text());sources={s['id']:s for s in selection['sources']}
        if len(sources)!=4 or len({s['family_id'] for s in sources.values()})!=4 or any(s['role']!='fit' for s in sources.values()):raise ValueError('four independent FIT families required')
        summary=json.loads((root/'summary.json').read_text())
        if len(summary['cases'])!=8 or summary['api_calls'] or summary['training_labels_admitted']:raise ValueError('unadmitted eight-case review required')
        for case in summary['cases']:
            source=sources[case['id']];w,h=source['native_size'];p,r=map(int,case['ratio'].split(':'));maximum=min(w//p,h//r)
            if case['role']!='fit' or case['family_id']!=source['family_id']:raise ValueError('case role/family differs')
            path=Path(case['source_png'])
            if sha(path)!=case['source_png_sha256']:raise ValueError('source PNG differs')
            images[str(path)]=case['source_png_sha256'];raw_nested+=bool(case['raw_nested'])
            for option in case['options']:
                outer,inner=[v['native_box'] for v in option['images']]
                for box in [outer,inner]:
                    x0,y0,x1,y1=box
                    if not(0<=x0<x1<=w and 0<=y0<y1<=h) or (x1-x0)*r!=(y1-y0)*p:raise ValueError('native source bounds or exact aspect differs')
                if outer[2]-outer[0]!=p*maximum or outer[3]-outer[1]!=r*maximum:raise ValueError('outer not maximum-area Minimal')
                if not(outer[0]<=inner[0] and outer[1]<=inner[1] and outer[2]>=inner[2] and outer[3]>=inner[3]):raise ValueError('inner is not nested')
                area=lambda b:(b[2]-b[0])*(b[3]-b[1])
                if area(inner)>=area(outer):raise ValueError('inner must be strictly smaller')
                if option['strategy']=='tighter' and not .36<=area(inner)/area(outer)<=.64:raise ValueError('registered tighter area differs')
                pairs+=1
            for pane in case['panes']:
                path=Path(pane['path'])
                if sha(path)!=pane['sha256']:raise ValueError('crop PNG changed')
                with Image.open(path) as im:
                    im.load()
                    if im.size!=(pane['width'],pane['height']) or im.width*r!=im.height*p or im.convert('RGB').tobytes()!=path.with_suffix('.rgb').read_bytes():raise ValueError('independent owned PNG readback differs')
                images[str(path)]=pane['sha256']
    for path in (a.root).glob('case*/*.png'):
        if path.read_bytes()!=(a.root/'tighter'/path.relative_to(a.root)).read_bytes():raise ValueError('original controls changed in tighter packet')
    result=dict(variants=2,curated_pairs=pairs,png_files=len(images),distinct_png_hashes=len(set(images.values())),original_controls_pixel_exact=True,api_calls=0,test_inference=False,automatic_pair_label_admission=False,excellent_intervals_admitted=0)
    with (a.root/'verification.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
