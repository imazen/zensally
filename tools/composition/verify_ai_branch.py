#!/usr/bin/env python3
"""Bind all generated local JXL bytes to the pinned remote ordinary Git blobs."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from inventory_unsplash import git_blob_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('jxl_root', type=Path)
    args = parser.parse_args()
    tree = json.loads((args.capture/'remote-tree.json').read_text())
    files = json.loads((args.capture/'remote-files.json').read_text())
    if tree['sha'] != 'a3606f90964885d641c3a79f5098e7170609667a' or tree['truncated']:
        raise ValueError('complete pinned AI branch tree required')
    blobs = {r['path']: r for r in tree['tree'] if r['type'] == 'blob'}
    for remote in files:
        data = remote['content'].encode()
        digest = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if digest != remote['sha']:
            raise ValueError('remote metadata blob differs')
    attributes = files[0]['content']
    if '*.jxl -filter -diff -merge -text' not in attributes.splitlines() or 'filter=lfs' in attributes:
        raise ValueError('JXL filter is not explicitly disabled')
    groups = {'sunburst-pilot-v1': '2026-11-08-pilot', 'sunburst-commerce-v2': '2026-11-08-commerce-v2',
              'sunburst-products-selfies-v3': '2026-11-08-products-selfies-v3'}
    rows = []
    for remote in files[1:]:
        for row in csv.DictReader(io.StringIO(remote['content']), delimiter='\t'):
            group = Path(row['file']).parts[1]
            path = args.jxl_root/groups[group]/row['id']/'lossless.jxl'
            blob = blobs[row['file']]
            git_sha, raw_sha = git_blob_sha(path)
            with path.open('rb') as stream:
                prefix = stream.read(16)
            valid_jxl = prefix.startswith(b'\xff\x0a') or prefix.startswith(b'\0\0\0\x0cJXL \r\n\x87\n')
            if (not valid_jxl or git_sha != blob['sha'] or raw_sha != row['sha256']
                    or path.stat().st_size != blob['size'] or blob['size'] != int(row['bytes'])):
                raise ValueError('JXL bytes do not match remote Git object/manifest: '+row['id'])
            rows.append(dict(id=row['id'], file=row['file'], bytes=blob['size'],
                sha256=raw_sha, git_blob_sha1=git_sha, valid_jxl_signature=True,
                lfs_pointer=False, local_jxl=str(path)))
    if len(rows) != 56 or set(r['file'] for r in rows) != {p for p in blobs if p.endswith('.jxl')}:
        raise ValueError('56JXL branch coverage differs')
    report = dict(branch='ai-originals', commit=tree['sha'], files=len(rows),
        bytes=sum(r['bytes'] for r in rows), attributes=attributes, all_remote_git_blobs_match=True,
        all_manifests_match=True, lfs_pointers=0, rows=rows, source_checkout_modified=False)
    with (args.capture/'verification.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
