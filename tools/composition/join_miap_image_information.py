#!/usr/bin/env python3
"""Pin image-information bytes and join TRAIN attribution/rotation without pixels."""
import argparse
import atexit
import base64
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlsplit,urlunsplit
from urllib.request import Request,urlopen

from artifact_identity import sha

LIMIT=768*1024**2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('metadata',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if sha(a.metadata/'SHA256SUMS.json')!='77fd61540c7748b066383a4940c504358b9e48f6bc45982cb2a53536eb573bb6':raise ValueError('frozen TRAIN annotation inventory differs')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.output.parent).free<4*1024**3:raise ValueError('image-information stage needs four GiB disk headroom')
    a.output.mkdir(exist_ok=True)
    if (a.output/'summary.json').exists():raise ValueError('image-information join already complete; inspect cached results')
    progress=a.output/'progress.json';finished=False
    def final_status():
        if not finished:progress.write_text(json.dumps(dict(pid=None,stage='failed-source-bytes-retained',checked_utc=datetime.now(timezone.utc).isoformat()))+'\n')
    atexit.register(final_status)
    head=json.loads((a.metadata/'image-information-head.json').read_text());headers=head['headers']
    length=int(headers['Content-Length']);generation=headers['x-goog-generation'];etag=headers['ETag']
    if length!=638407721 or generation!='1526567626302755' or etag!='"a4ac0bcedb5c2df4d1b1230dc6b41b8f"':raise ValueError('probed source object identity differs')
    expected_md5=dict(v.strip().split('=',1) for v in headers['x-goog-hash'].split(','))['md5']
    url=head['url']+'?generation='+generation;raw=a.output/'train-image-information.csv';record=a.output/'download.json'
    if raw.exists():
        cached=json.loads(record.read_text())
        if cached['url']!=url or sha(raw)!=cached['sha256'] or raw.stat().st_size!=length:raise ValueError('cached information source differs or is incomplete')
    else:
        digest=hashlib.sha256();md5=hashlib.md5(usedforsecurity=False);size=0
        with urlopen(Request(url,headers={'User-Agent':'zensally-source-research','Accept-Encoding':'identity','If-Match':etag}),timeout=60) as response:
            if response.headers.get('ETag')!=etag or response.headers.get('x-goog-generation')!=generation:raise ValueError('source changed after size/identity probe')
            response_headers=dict(response.headers.items())
            with raw.open('xb') as stream:
                while block:=response.read(1024*1024):
                    size+=len(block)
                    if size>length or size>LIMIT:raise ValueError('declared image-information size exceeded; preserve partial data')
                    stream.write(block);digest.update(block);md5.update(block)
                    progress.write_text(json.dumps(dict(pid=os.getpid(),stage='download-information',bytes=size,required_bytes=length,checked_utc=datetime.now(timezone.utc).isoformat()))+'\n')
            if size!=length or base64.b64encode(md5.digest()).decode()!=expected_md5:raise ValueError('download size or independent provider MD5 differs')
            record.write_text(json.dumps(dict(url=url,resolved_url=response.url,headers=response_headers,bytes=size,
                sha256=digest.hexdigest(),provider_md5_base64=expected_md5,provider_md5_exact=True,captured_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
    wanted={line.split('/')[1] for line in (a.metadata/'train-keys.lst').read_text().splitlines()}
    quarantine={r['id'].split(':')[1] for r in json.loads((a.metadata/'invalid-geometry.json').read_text())['records']}
    matched=set();author_keys=set();licenses=Counter();rotations=Counter();issues=Counter();row_count=0
    csv.field_size_limit(1024*1024)
    with raw.open(newline='',encoding='utf-8-sig') as stream,(a.output/'miap-training-image-information.jsonl').open('x') as out:
        reader=csv.DictReader(stream)
        required=['ImageID','Subset','OriginalURL','OriginalLandingURL','License','AuthorProfileURL','Author','OriginalMD5','Rotation']
        if not set(required)<=set(reader.fieldnames):raise ValueError('required attribution/source/rotation fields missing')
        for row in reader:
            row_count+=1;identifier=row['ImageID']
            if identifier not in wanted:continue
            if identifier in matched or row['Subset']!='train':raise ValueError('duplicate or cross-official-split source ID')
            matched.add(identifier);reason=[];licenses[row['License']]+=1;rotations[row['Rotation']]+=1
            author=urlsplit(row['AuthorProfileURL']);canonical=urlunsplit(('https',author.netloc.lower(),author.path.rstrip('/'),'','')) if author.scheme in ['http','https'] and author.netloc and author.path.strip('/') else None
            if not canonical:reason.append('unknown_author_profile')
            else:author_keys.add(canonical)
            if not row['OriginalURL'] or not row['OriginalLandingURL']:reason.append('missing_original_source_link')
            if not row['License']:reason.append('missing_listed_license')
            if row['Rotation'] not in ['0','90','180','270']:reason.append('unknown_rotation')
            try:
                if len(base64.b64decode(row['OriginalMD5'],validate=True))!=16:reason.append('unknown_original_md5')
            except ValueError:reason.append('unknown_original_md5')
            if identifier in quarantine:reason.append('quarantined_annotation_geometry')
            issues.update(reason)
            out.write(json.dumps(dict(id='openimages:'+identifier,official_split='train',role=None,training_eligible=False,
                source=row,canonical_author_profile=canonical,source_metadata_issues=reason,
                per_image_live_license_verified=False,face_coverage_certified=False))+'\n')
            if len(matched)%1000==0:progress.write_text(json.dumps(dict(pid=os.getpid(),stage='join-attribution',matched=len(matched),required=70000,checked_utc=datetime.now(timezone.utc).isoformat()))+'\n')
    missing=sorted(wanted-matched)
    (a.output/'unmatched-train-ids.json').write_text(json.dumps(missing,indent=2)+'\n')
    summary=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        metadata_packet_sha256=sha(a.metadata/'SHA256SUMS.json'),image_information_sha256=sha(raw),
        source_rows_scanned=row_count,matched_training_images=len(matched),unmatched_training_images=len(missing),
        listed_licenses=dict(licenses),rotations=dict(rotations),distinct_known_author_profiles=len(author_keys),
        source_metadata_issues=dict(issues),original_source_hyperlinks_retained=True,
        images_acquired=0,source_roles_assigned=False,training_eligible=False,
        per_image_live_license_verified=False,existing_holdout_reconciliation_complete=False,
        demographic_attributes_used=False,face_coverage_certified=False,human_artistic_labels=False,paid_model_api_calls=0)
    if sum(p.stat().st_size for p in a.output.iterdir() if p.is_file())>LIMIT:raise ValueError('768MiB metadata allowance exceeded; preserve output')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    progress.write_text(json.dumps(dict(pid=None,stage='complete',matched=len(matched),required=70000,checked_utc=datetime.now(timezone.utc).isoformat()))+'\n');finished=True
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
