#!/usr/bin/env python3
"""Cross-check every retained PNG using owned decode and independent readback."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from PIL import Image

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('renderer',type=Path);p.add_argument('output',type=Path);a=p.parse_args();rows=[]
    for path in sorted(a.root.rglob('*.png')):
        owned=json.loads(subprocess.check_output([str(a.renderer),'--inspect',str(path)]))
        with Image.open(path) as image:
            image.load();actual={'width':image.width,'height':image.height,'rgb_sha256':hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()}
        if owned!=actual:raise ValueError('independent PNG pixel disagreement: '+str(path))
        rows.append({'file':str(path.relative_to(a.root)),**actual})
    if not rows:raise ValueError('no PNGs inspected')
    with a.output.open('x') as f:f.write(json.dumps({'pngs_checked':len(rows),'all_rgb_hashes_match':True,'records':rows},indent=2)+'\n')
    print(json.dumps({'pngs_checked':len(rows),'all_rgb_hashes_match':True}))

if __name__=='__main__':main()
