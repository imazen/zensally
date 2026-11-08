#!/usr/bin/env python3
"""Verify local corpus bytes against pinned remote Git blobs without editing it."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse


def git_blob_sha(path):
    digest = hashlib.sha1(b'blob '+str(path.stat().st_size).encode()+b'\0')
    raw_sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
            raw_sha.update(block)
    return digest.hexdigest(), raw_sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['checkout', 'remote', 'corpus', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = json.loads((args.remote/'remote-tree.json').read_text())
    files = json.loads((args.remote/'remote-files.json').read_text())
    if tree['sha'] != 'b8d7cd87562c271a7725bd446d656587a5886c01' or tree['truncated']:
        raise ValueError('pinned complete remote tree required')
    blobs = {r['path']: r for r in tree['tree'] if r['type'] == 'blob'}
    for remote in files:
        name = remote['display_title']
        actual, _ = git_blob_sha(args.checkout/name)
        if actual != remote['sha'] or actual != blobs[name]['sha']:
            raise ValueError('local metadata differs from remote pinned blob: '+name)
    text = next(r['content'] for r in files if r['display_title'] == 'manifest.tsv')
    rows = list(csv.DictReader(io.StringIO(text), delimiter='\t'))
    if len(rows) != 144 or len({r['file'] for r in rows}) != 144:
        raise ValueError('registered144source inventory differs')
    prior = []
    for pattern in ['jxl-*.tsv', 'historical-*.tsv']:
        for path in sorted(args.corpus.glob(pattern)):
            with path.open(newline='') as stream:
                prior.extend(dict(row, metadata_manifest=path.name) for row in csv.DictReader(stream, delimiter='\t'))
    args.output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(args.output).free < 8*1024**3:
        raise ValueError('need8GiB disk headroom')
    original_root = args.output/'originals'
    original_root.mkdir()
    records = []
    links = [dict(text=r['display_title'], href=r['display_url']) for r in files]
    for row in rows:
        path = args.checkout/row['file']
        if not path.resolve().is_relative_to((args.checkout/'originals').resolve()):
            raise ValueError('source path escapes originals')
        blob = blobs[row['file']]
        if path.stat().st_size != int(row['bytes']) or blob['size'] != int(row['bytes']):
            raise ValueError('source byte count differs')
        git_sha, raw_sha = git_blob_sha(path)
        if raw_sha != row['sha256'] or git_sha != blob['sha']:
            raise ValueError('source differs from pinned remote/manifest: '+row['file'])
        photo_id = urlparse(row['photo_url']).path.rsplit('/', 1)[-1]
        suffix = '-'+photo_id+'-unsplash.jpg'
        if not path.name.endswith(suffix) or len(photo_id) != 11:
            raise ValueError('filename/provider ID identity differs')
        photographer = path.name[:-len(suffix)]
        overlaps = [dict(role=p.get('split') or p.get('bucket') or p['metadata_manifest'].removeprefix(
                            'jxl-').removeprefix('historical-').removesuffix('.tsv'), metadata_manifest=p['metadata_manifest'],
                        source_id=p.get('id', p.get('source_id', 'unknown')))
                    for p in prior if raw_sha in [p.get(k) for k in ['sha256', 'original_sha256']]
                    or photo_id in p.get('path', '') or photo_id in p.get('url', '') or photo_id in p.get('source_url', '')]
        destination = original_root/path.name
        with path.open('rb') as source, destination.open('xb') as target:
            shutil.copyfileobj(source, target, length=1024*1024)
        copied_git, copied_sha = git_blob_sha(destination)
        if copied_git != git_sha or copied_sha != raw_sha:
            raise ValueError('copied source identity differs')
        records.append(dict(id='unsplash:'+photo_id, provider_id=photo_id, photographer=photographer,
            provisional_group='unsplash-photographer:'+photographer, source_url=row['photo_url'],
            source_repository='https://github.com/imazen/zensally-datasets', branch='unsplash-originals',
            commit=tree['sha'], git_blob_sha1=git_sha, sha256=raw_sha, bytes=int(row['bytes']),
            file=str(destination), canonical_overlap=overlaps, use='screening', family_role=None,
            training_eligible=False, human_rated=False, face_coverage_certified=False,
            licence_review='source repository states applicable Unsplash terms; detailed admission unresolved'))
        links.append(dict(text='Unsplash photo '+photo_id, href=row['photo_url']))
    if len({r['provider_id'] for r in records}) != 144:
        raise ValueError('provider ID duplicated')
    report = dict(commit=tree['sha'], sources=len(records), bytes=sum(r['bytes'] for r in records),
        photographer_groups=len({r['photographer'] for r in records}),
        exact_duplicate_hash_groups=[dict(sha256=h, count=n) for h, n in Counter(r['sha256'] for r in records).items() if n > 1],
        sources_with_canonical_identity_matches=sum(bool(r['canonical_overlap']) for r in records),
        metadata_only_split_audit=True, perceptual_duplicate_audit_complete=False,
        source_checkout_modified=False, roles_assigned=False, paid_model_api_calls=0)
    for name, data in [('sources.json', records), ('summary.json', report)]:
        (args.output/name).write_text(json.dumps(data, indent=2)+'\n')
    (args.output/'links.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in links))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
