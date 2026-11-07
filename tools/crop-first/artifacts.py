#!/usr/bin/env python3
"""Freeze/verify complete artifact membership and streamed SHA256 hashes."""
import argparse,hashlib,json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','verify']);p.add_argument('directory',type=Path);a=p.parse_args()
    root=a.directory;manifest=root/'SHA256SUMS.json'
    files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p!=manifest}
    if a.action=='freeze':
        with manifest.open('x') as f:f.write(json.dumps(files,indent=2,sort_keys=True)+'\n')
    elif json.loads(manifest.read_text())!=files:raise ValueError('artifact membership or SHA256 differs')
    print(json.dumps({'directory':str(root),'files':len(files),'manifest_sha256':sha(manifest),'verified':a.action=='verify'}))


if __name__=='__main__':main()
