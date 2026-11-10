#!/usr/bin/env python3
"""Verify frozen PNG byte identity against the pinned tracked variant objects."""
import argparse,hashlib,json,subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('variant_root',type=Path);p.add_argument('resolution',type=Path);p.add_argument('sources',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    resolution=json.loads(a.resolution.read_text());sources={r['id']:r for r in json.loads(a.sources.read_text())};rows=[]
    for r in resolution['sources']:
        blob=subprocess.run(['git','-C',str(a.variant_root),'show',resolution['variant_commit']+':'+r['variant_path']],check=True,capture_output=True).stdout
        if blob.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            expected=next(line.removeprefix('oid sha256:') for line in blob.decode().splitlines() if line.startswith('oid sha256:'))
        else:expected=hashlib.sha256(blob).hexdigest()
        if expected!=sources[r['id']]['png_sha256']:raise ValueError('canonical variant byte identity differs for '+r['id'])
        rows.append({'source_id':r['id'],'expected_png_sha256':expected,'matches':True})
    a.output.write_text(json.dumps({'variant_commit':resolution['variant_commit'],'sources':rows},indent=2)+'\n');print(json.dumps({'canonical_variant_pngs_verified':len(rows)}))


if __name__=='__main__':main()
