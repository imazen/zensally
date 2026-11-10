#!/usr/bin/env python3
"""Read-only independent RGB verification of the frozen source PNGs."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('inputs',type=Path);p.add_argument('executable');p.add_argument('output',type=Path);a=p.parse_args();records=[]
    for source in json.loads((a.inputs/'sources.json').read_text()):
        path=a.inputs/f'{source["id"]}.png';assert hashlib.sha256(path.read_bytes()).hexdigest()==source['png_sha256']
        own=json.loads(subprocess.run([a.executable,'--inspect',str(path)],check=True,capture_output=True,text=True).stdout)
        with Image.open(path) as im:
            im.load();rgb=im.convert('RGB').tobytes();digest=hashlib.sha256(rgb).hexdigest()
            assert list(im.size)==[own['width'],own['height']]==[source['width'],source['height']]
            assert digest==own['rgb_sha256']
            records.append({'source_id':source['id'],'role':source['role'],'dimensions':list(im.size),'rgb_sha256':digest})
    a.output.write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({'independently_verified_source_RGB':len(records)}))


if __name__=='__main__':main()
