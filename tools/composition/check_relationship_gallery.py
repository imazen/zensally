#!/usr/bin/env python3
"""Verify the advertised gallery and all rendered image bytes over HTTP."""
import argparse
import io
import json
from pathlib import Path
import urllib.request
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('url')
    p.add_argument('--face-candidates',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    report=dict(url=a.url,images=0,image_bytes=0,status=None)
    pattern='case*.png' if a.face_candidates else 'case*/*.png'
    for path in [a.root/'index.html',*sorted(a.root.glob(pattern))]:
        expected=path.read_bytes();url=a.url.rsplit('/',1)[0]+'/'+str(path.relative_to(a.root))
        with urllib.request.urlopen(url,timeout=10) as response:
            actual=response.read(len(expected)+1)
            if response.status!=200 or actual!=expected:raise ValueError('advertised gallery bytes differ: '+url)
        if path.suffix=='.png':
            with Image.open(io.BytesIO(actual)) as image:image.load()
            report['images']+=1;report['image_bytes']+=len(actual)
        else:report['status']=200
    summary=json.loads((a.root/'summary.json').read_text())
    required=sum(len(w['panes']) for w in summary['cases']) if a.face_candidates else sum(1+len(w['images']) for w in summary['witnesses'])
    if report['images']!=required:raise ValueError('missing witness images')
    with (a.output or a.root/'http-check.json').open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
