#!/usr/bin/env python3
"""Verify a pushed dataset tree, ordinary blobs and every stored image hash."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('repository',type=Path);p.add_argument('revision');p.add_argument('expected_images',type=int);p.add_argument('output',type=Path);a=p.parse_args()
    def git(*args):return subprocess.check_output(['git','-C',str(a.repository),*args])
    revision=git('rev-parse',a.revision).decode().strip()
    subprocess.run(['git','-C',str(a.repository),'merge-base','--is-ancestor',revision,'origin/ai-originals'],check=True)
    paths=[v.decode() for v in git('ls-tree','-rz','--name-only',revision).split(b'\0') if v.endswith(b'.jxl')]
    records=[]
    for path in paths:
        raw=git('show',revision+':'+path)
        if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):raise ValueError('LFS pointer instead of original blob')
        metadata=json.loads(git('show',revision+':'+path[:-4]+'.json'))
        digest=hashlib.sha256(raw).hexdigest()
        if digest!=metadata['storage']['sha256'] or len(raw)!=metadata['storage']['bytes']:raise ValueError('stored original differs from verified metadata')
        records.append(dict(path=path,bytes=len(raw),sha256=digest))
    if len(records)!=a.expected_images:raise ValueError('remote image count differs')
    result=dict(revision=revision,images=len(records),bytes=sum(r['bytes'] for r in records),lfs_pointers=0,all_stored_hashes_match=True,records=records)
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
