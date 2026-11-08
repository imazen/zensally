#!/usr/bin/env python3
"""Freeze/verify complete artifact membership and streamed SHA256 hashes."""
import argparse,hashlib,json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify(root, expected):
    paths=[p for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'SHA256SUMS.json']
    if isinstance(expected, list):
        # The richer successor packet also binds exact sizes. Never collapse
        # duplicate paths into a mapping or silently ignore unexpected fields.
        actual=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]
        if expected!=actual:raise ValueError('artifact membership, size or SHA256 differs')
    elif isinstance(expected, dict):
        actual={str(p.relative_to(root)):sha(p) for p in paths}
        if expected!=actual:raise ValueError('artifact membership or SHA256 differs')
    else:raise ValueError('unsupported artifact manifest schema')
    return len(paths)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','verify']);p.add_argument('directory',type=Path);a=p.parse_args()
    root=a.directory;manifest=root/'SHA256SUMS.json'
    if a.action=='freeze':
        files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p!=manifest}
        with manifest.open('x') as f:f.write(json.dumps(files,indent=2,sort_keys=True)+'\n')
        count=len(files)
    else:count=verify(root,json.loads(manifest.read_text()))
    print(json.dumps({'directory':str(root),'files':count,'manifest_sha256':sha(manifest),'verified':a.action=='verify'}))


if __name__=='__main__':main()
