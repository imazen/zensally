#!/usr/bin/env python3
"""Freeze validation-only representative sources for rectangle comparisons.

Uses canonical imazen-26 validation URLs, keeps source links and hashes, and
supplements representative images to at least 50 per broad content class.
Originals stay in the explicitly supplied cache. No images are generated.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import shutil
import threading
import urllib.request


def content_class(name):
    number = int(name.split('-')[0])
    if number <= 2000:
        return 'photo'
    if number in (8000, 8100):
        return 'screen'
    if number in (6000, 6800, 7000):
        return 'line-and-document'
    return 'mixed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('validation_tsv', type=Path)
    parser.add_argument('representatives_tsv', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--max-bytes', type=int, required=True)
    args = parser.parse_args()
    with args.validation_tsv.open() as f:
        valid = {r['png_v3_sdr_url']: r for r in csv.DictReader(f, delimiter='\t') if r['png_v3_sdr_url']}
    selected = {}
    with args.representatives_tsv.open() as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['url'] in valid and r['url'] not in selected:
                selected[r['url']] = {**valid[r['url']], 'cluster': r['cluster_id']}
    for label in ('photo', 'screen', 'line-and-document', 'mixed'):
        count = sum(content_class(r['content_class']) == label for r in selected.values())
        for url, r in sorted(valid.items(), key=lambda pair: int(pair[1]['id'])):
            if count >= 50:
                break
            if content_class(r['content_class']) == label and url not in selected:
                selected[url] = {**r, 'cluster': 'supplement'}
                count += 1
        if count < 50:
            raise RuntimeError(f'{label}: only {count} validation sources')
    args.output.mkdir(parents=True, exist_ok=True)
    source = {'source_commit': args.source_commit, 'validation_sha256': hashlib.sha256(args.validation_tsv.read_bytes()).hexdigest(),
              'representatives_sha256': hashlib.sha256(args.representatives_tsv.read_bytes()).hexdigest(),
              'selection': 'whole validation source images represented by K1000; supplement each broad class to 50', 'count': len(selected)}
    source_path = args.output / 'source.json'
    if source_path.exists() and json.loads(source_path.read_text()) != source:
        raise RuntimeError('cache belongs to different frozen inputs')
    if not source_path.exists():
        source_path.write_text(json.dumps(source, indent=2) + '\n')
    lock = threading.Lock()
    used = sum(p.stat().st_size for p in args.output.glob('*.png'))
    def fetch(item):
        nonlocal used
        url, row = item
        path = args.output / (row['id'] + '.png')
        checksum = path.with_suffix('.sha256')
        if path.exists():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if not checksum.exists() or checksum.read_text().strip() != actual:
                raise RuntimeError(f'cache hash mismatch for id {row["id"]}')
        else:
            partial = path.with_suffix('.partial')
            digest = hashlib.sha256()
            request = urllib.request.Request(url, headers={'User-Agent': 'zensally-eval/0.1 (+https://github.com/imazen/zensally)'})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open('xb') as output:
                while chunk := response.read(1024 * 1024):
                    with lock:
                        if used + len(chunk) > args.max_bytes or shutil.disk_usage(args.output).free < 2 * 1024**3:
                            raise RuntimeError('corpus byte/disk budget exceeded; partial input preserved')
                        used += len(chunk)
                    output.write(chunk)
                    digest.update(chunk)
            partial.rename(path)
            actual = digest.hexdigest()
            checksum.write_text(actual + '\n')
        return [row['id'], content_class(row['content_class']), path.name, actual, url, row['cluster']]
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(fetch, sorted(selected.items(), key=lambda pair: int(pair[1]['id']))):
            rows.append(row)
            print(f'fetched {len(rows)}/{len(selected)} id={row[0]} bytes={used}', flush=True)
    manifest = args.output / 'inputs.tsv'
    with manifest.open('x', newline='') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(['id', 'class', 'path', 'sha256', 'url', 'cluster'])
        writer.writerows(rows)
    print(json.dumps({'images': len(rows), 'bytes': used, 'manifest': str(manifest)}))


if __name__ == '__main__':
    main()
