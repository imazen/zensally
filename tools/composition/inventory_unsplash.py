#!/usr/bin/env python3
"""Verify local corpus bytes against pinned remote Git blobs without editing it."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import time
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
    parser.add_argument('--download-missing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    tree = json.loads((args.remote/'remote-tree.json').read_text())
    files = json.loads((args.remote/'remote-files.json').read_text())
    if tree['sha'] != 'b8d7cd87562c271a7725bd446d656587a5886c01' or tree['truncated']:
        raise ValueError('pinned complete remote tree required')
    blobs = {r['path']: r for r in tree['tree'] if r['type'] == 'blob'}
    local_metadata_matches = {}
    for remote in files:
        name = remote['display_title']
        body = remote['content'].encode('utf-8')
        actual = hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
        if actual != remote['sha'] or actual != blobs[name]['sha']:
            raise ValueError('captured remote metadata differs from pinned blob: '+name)
        local = args.checkout/name
        local_metadata_matches[name] = local.is_file() and git_blob_sha(local)[0] == actual
    text = next(r['content'] for r in files if r['display_title'] == 'manifest.tsv')
    rows = list(csv.DictReader(io.StringIO(text), delimiter='\t'))
    if len(rows) != 144 or len({r['file'] for r in rows}) != 144:
        raise ValueError('registered144source inventory differs')
    prior = []
    for pattern in ['jxl-*.tsv', 'historical-*.tsv']:
        for path in sorted(args.corpus.glob(pattern)):
            with path.open(newline='') as stream:
                prior.extend(dict(row, metadata_manifest=path.name) for row in csv.DictReader(stream, delimiter='\t'))
    args.output.mkdir(parents=True, exist_ok=args.resume)
    protocol = dict(commit=tree['sha'], tree_sha256=hashlib.sha256((args.remote/'remote-tree.json').read_bytes()).hexdigest(),
                    remote_files_sha256=hashlib.sha256((args.remote/'remote-files.json').read_bytes()).hexdigest())
    if (args.output/'protocol.json').exists():
        if json.loads((args.output/'protocol.json').read_text()) != protocol or (args.output/'summary.json').exists():
            raise ValueError('resume protocol differs or complete cache is already closed')
    else:
        (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    if shutil.disk_usage(args.output).free < 8*1024**3:
        raise ValueError('need8GiB disk headroom')
    original_root = args.output/'originals'
    original_root.mkdir(exist_ok=args.resume)
    metadata_root = args.output/'remote-metadata'
    metadata_root.mkdir(exist_ok=args.resume)
    for remote in files:
        target = metadata_root/remote['display_title']
        data = remote['content'].encode('utf-8')
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError('retained remote metadata changed')
        else:
            target.write_bytes(data)
    records = []
    links = [dict(text=r['display_title'], href=r['display_url']) for r in files]
    for row in rows:
        path = args.checkout/row['file']
        if not path.resolve().is_relative_to((args.checkout/'originals').resolve()):
            raise ValueError('source path escapes originals')
        blob = blobs[row['file']]
        git_sha, raw_sha = blob['sha'], row['sha256']
        if blob['size'] != int(row['bytes']) or int(row['bytes']) > 64*1024**2:
            raise ValueError('remote source byte count/bound differs')
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
        local_matches = path.is_file() and path.stat().st_size == int(row['bytes']) and git_blob_sha(path) == (blob['sha'], row['sha256'])
        origin = 'matching local bytes' if local_matches else 'pinned remote Git blob'
        if destination.exists():
            if not args.resume:
                raise ValueError('output already exists')
            origin = 'resumed verified source'
        elif local_matches:
            with path.open('rb') as source, destination.open('xb') as target:
                shutil.copyfileobj(source, target, length=1024*1024)
        elif args.download_missing:
            partial = destination.with_name(destination.name+'.partial-'+str(time.time_ns()))
            with partial.open('xb') as stream:
                subprocess.run(['gh', 'api', 'repos/imazen/zensally-datasets/git/blobs/'+blob['sha'],
                    '-H', 'Accept: application/vnd.github.raw+json'], stdout=stream, check=True, timeout=180)
            if partial.stat().st_size != int(row['bytes']) or git_blob_sha(partial) != (blob['sha'], row['sha256']):
                raise ValueError('downloaded Git blob differs; partial retained')
            partial.rename(destination)
        else:
            raise ValueError('local bytes differ; explicit remote-download flag required: '+row['file'])
        copied_git, copied_sha = git_blob_sha(destination)
        git_sha, raw_sha = blob['sha'], row['sha256']
        if copied_git != git_sha or copied_sha != raw_sha:
            raise ValueError('copied source identity differs')
        records.append(dict(id='unsplash:'+photo_id, provider_id=photo_id, photographer=photographer,
            provisional_group='unsplash-photographer:'+photographer, source_url=row['photo_url'],
            source_repository='https://github.com/imazen/zensally-datasets', branch='unsplash-originals',
            commit=tree['sha'], git_blob_sha1=git_sha, sha256=raw_sha, bytes=int(row['bytes']), acquisition_origin=origin,
            file=str(destination), canonical_overlap=overlaps, use='screening', family_role=None,
            training_eligible=False, human_rated=False, face_coverage_certified=False,
            licence_review='source repository states applicable Unsplash terms; detailed admission unresolved'))
        links.append(dict(text='Unsplash photo '+photo_id, href=row['photo_url']))
        (args.output/'progress.json').write_text(json.dumps(dict(pid=os.getpid(), stage='acquiring',
            completed=len(records), required=144, last_source_id=photo_id, last_checked_unix=time.time()))+'\n')
        print(json.dumps(dict(completed=len(records), required=144, id=photo_id, origin=origin)), flush=True)
    if len({r['provider_id'] for r in records}) != 144:
        raise ValueError('provider ID duplicated')
    report = dict(commit=tree['sha'], sources=len(records), bytes=sum(r['bytes'] for r in records),
        photographer_groups=len({r['photographer'] for r in records}),
        exact_duplicate_hash_groups=[dict(sha256=h, count=n) for h, n in Counter(r['sha256'] for r in records).items() if n > 1],
        sources_with_canonical_identity_matches=sum(bool(r['canonical_overlap']) for r in records),
        metadata_only_split_audit=True, perceptual_duplicate_audit_complete=False,
        local_metadata_matches_remote=local_metadata_matches,
        acquisition_origins=dict(Counter(r['acquisition_origin'] for r in records)),
        source_checkout_modified=False, roles_assigned=False, paid_model_api_calls=0)
    for name, data in [('sources.json', records), ('summary.json', report)]:
        (args.output/name).write_text(json.dumps(data, indent=2)+'\n')
    (args.output/'links.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in links))
    (args.output/'progress.json').write_text(json.dumps(dict(pid=None, stage='complete',
        completed=144, required=144, last_checked_unix=time.time()))+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
