#!/usr/bin/env python3
"""Verify a pushed dataset tree, ordinary blobs and every stored image hash."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def verify_review(review,metadata):
    for row in review['records']:
        old=metadata[row['stored_source_id']];new=metadata[row['replacement_stored_source_id']]
        if (row['positive_training_reference_eligible'] or row['positive_evaluation_reference_eligible'] or
                row['png_sha256']!=old['png_sha256'] or old['scenario_family']!=new['scenario_family'] or
                old['registered_role']!=new['registered_role'] or new['replaces_source_id']!=row['source_id']):
            raise ValueError('rejection identity or replacement family/role differs')
    return len(review['records'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('repository',type=Path);p.add_argument('revision');p.add_argument('expected_images',type=int);p.add_argument('output',type=Path)
    p.add_argument('--require-source-review',action='store_true');a=p.parse_args()
    def git(*args):return subprocess.check_output(['git','-C',str(a.repository),*args])
    revision=git('rev-parse',a.revision).decode().strip()
    subprocess.run(['git','-C',str(a.repository),'merge-base','--is-ancestor',revision,'origin/ai-originals'],check=True)
    paths=[v.decode() for v in git('ls-tree','-rz','--name-only',revision).split(b'\0') if v.endswith(b'.jxl')]
    records=[];sources={};legacy_sources=0
    for path in paths:
        raw=git('show',revision+':'+path)
        if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):raise ValueError('LFS pointer instead of original blob')
        metadata=json.loads(git('show',revision+':'+path[:-4]+'.json'))
        identity=metadata.get('source_id','legacy-file:'+path)
        legacy_sources+='source_id' not in metadata
        if identity in sources:raise ValueError('stored source ID duplicated')
        sources[identity]=metadata
        digest=hashlib.sha256(raw).hexdigest()
        if digest!=metadata['storage']['sha256'] or len(raw)!=metadata['storage']['bytes']:raise ValueError('stored original differs from verified metadata')
        records.append(dict(path=path,bytes=len(raw),sha256=digest))
    if len(records)!=a.expected_images:raise ValueError('remote image count differs')
    result=dict(revision=revision,images=len(records),bytes=sum(r['bytes'] for r in records),lfs_pointers=0,all_stored_hashes_match=True,legacy_sources_with_path_identity=legacy_sources,records=records)
    if a.require_source_review:result['verified_source_rejections']=verify_review(json.loads(git('show',revision+':SOURCE_REVIEW.json')),sources)
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
