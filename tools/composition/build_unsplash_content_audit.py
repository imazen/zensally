#!/usr/bin/env python3
"""Build bounded owned source previews for a fit/development content audit."""
import argparse
import json
from pathlib import Path
import subprocess

from PIL import Image

from artifact_identity import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['registry','resizer','output']:
        p.add_argument(name,type=Path)
    args = p.parse_args()
    if sha(args.registry) != '6f8d318cc870248dcc2400a235fcbaa5f7f0edf47d575d965bf5c72b5684d2f6':
        raise ValueError('fixed source/family roles differ')
    if sha(args.resizer) != 'ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041':
        raise ValueError('owned source resizer differs')
    sources = [s for s in json.loads(args.registry.read_text())['sources'] if s['role'] in ['fit','dev']]
    if len(sources) != 119 or sum(s['role']=='fit' for s in sources) != 94:
        raise ValueError('registered94/25 source coverage required')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'previews').mkdir()
    rows,asset_bytes = [],0
    for index,source in enumerate(sources):
        raw = Path(source['rgb'])
        if sha(raw) != source['rgb_sha256']:
            raise ValueError('owned source RGB differs')
        w,h = source['working_size']
        scale = min(1,320/max(w,h))
        tw,th = max(1,round(w*scale)),max(1,round(h*scale))
        with raw.open('rb') as stream:
            resized = subprocess.check_output([str(args.resizer),str(w),str(h),str(w*3),
                f'0,0,{w},{h},{tw},{th}'],stdin=stream)
        if len(resized) != tw*th*3:
            raise ValueError('owned resize length differs')
        path = args.output/'previews'/(source['source']['provider_id']+'.png')
        Image.frombytes('RGB',(tw,th),resized).save(path,format='PNG')
        with Image.open(path) as image:
            if image.size != (tw,th) or image.convert('RGB').tobytes() != resized:
                raise ValueError('PNG readback differs from owned RGB')
        asset_bytes += path.stat().st_size
        if asset_bytes > 64*1024*1024:
            raise ValueError('registered preview storage budget exceeded; preserve existing files')
        rows.append(dict(source,index=index,sheet=index//10,preview=str(path.relative_to(args.output)),
            preview_size=[tw,th],preview_sha256=sha(path),preview_rgb_sha256=__import__('hashlib').sha256(resized).hexdigest(),
            content_reviewed=False,face_coverage_certified=False,artistic_quality_rated=False))
    data = dict(sources=rows,registry_sha256=sha(args.registry),resizer_sha256=sha(args.resizer),
        build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        fit_sources=94,dev_sources=25,test_sources=0,preview_asset_bytes=asset_bytes,
        source_roles_changed=False,training_admission_changed=False,paid_model_api_calls=0)
    with (args.output/'sources.json').open('x') as stream:
        stream.write(json.dumps(data,indent=2)+'\n')
    template = Path(__file__).with_name('unsplash_content_audit.html').read_text()
    with (args.output/'index.html').open('x') as stream:
        stream.write(template.replace('/*DATA*/null',json.dumps(data).replace('</','<\\/')))
    print(json.dumps(dict(sources=len(rows),sheets=12,preview_asset_bytes=asset_bytes,test_sources=0)))


if __name__ == '__main__':
    main()
