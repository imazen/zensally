#!/usr/bin/env python3
"""Independent read-only PNG and retained RGB/layout checks; writes a report only."""
import argparse,hashlib,json
from pathlib import Path
from PIL import Image

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('inputs',type=Path);p.add_argument('output',type=Path);p.add_argument('--jxl-oracle',type=Path);a=p.parse_args();checks=[]
    for s in json.loads((a.inputs/'sources.json').read_text()):
        d=Path(s['directory']);raw=(d/'source.rgb').read_bytes()
        with Image.open(d/'source.png') as im:
            if im.size!=(s['width'],s['height']) or im.convert('RGB').tobytes()!=raw:raise ValueError('owned PNG changed source RGB')
        with Image.open(d/'preview.png') as im:
            if im.convert('RGB').tobytes()!=(d/'preview.rgb').read_bytes():raise ValueError('owned PNG changed preview RGB')
        tensor=(d/'tensor.rgb').read_bytes()
        if len(tensor)!=256*256*3 or hashlib.sha256(tensor).hexdigest()!=s['tensor_sha256']:raise ValueError('model tensor identity mismatch')
        x0,y0,x1,y1=[round(v*256) for v in s['valid_rect']]
        if x1-x0<1 or y1-y0<1:raise ValueError('empty fitted content')
        for y in range(256):
            row=tensor[y*768:(y+1)*768]
            if y<y0 or y>=y1:
                if any(row):raise ValueError('nonzero model padding')
            elif any(row[:x0*3]) or any(row[x1*3:]):raise ValueError('nonzero model padding')
        checks.append({'id':s['id'],'source_png_rgb_exact':True,'preview_png_rgb_exact':True,'tensor_hash_and_padding':True})
    independent=None
    if a.jxl_oracle:
        raw=(a.inputs/'19030/source.rgb').read_bytes()
        with Image.open(a.jxl_oracle) as im:
            decoded=im.convert('RGB').tobytes()
            if decoded!=raw:raise ValueError('lossless JXL independent decoder disagreement')
        independent={'id':'19030','lossless_jxl_rgb_exact':True,'oracle_file':str(a.jxl_oracle)}
    with a.output.open('x') as f:json.dump({'checks':checks,'independent_jxl':independent},f,indent=2);f.write('\n')
    print(json.dumps({'sources_checked':len(checks),'independent_jxl':independent}))
if __name__=='__main__':main()
