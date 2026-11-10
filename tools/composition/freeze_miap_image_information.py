#!/usr/bin/env python3
"""Freeze the attribution join, original observations and exact source producers."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    if (a.root/'SHA256SUMS.json').exists():raise ValueError('join packet already frozen')
    summary=json.loads((a.root/'summary.json').read_text());progress=json.loads((a.root/'progress.json').read_text())
    download=json.loads((a.root/'download.json').read_text());raw=a.root/'train-image-information.csv'
    if (progress['pid'] is not None or progress['stage']!='complete' or summary['matched_training_images']!=70000
        or summary['unmatched_training_images'] or summary['images_acquired'] or summary['training_eligible']
        or summary['source_metadata_issues']!={'unknown_rotation':9541,'quarantined_annotation_geometry':5}):
        raise ValueError('complete corrected metadata-only join required')
    if raw.stat().st_size!=638407721 or sha(raw)!=download['sha256'] or download['sha256']!=summary['image_information_sha256']:
        raise ValueError('pinned source bytes differ')
    with raw.open('rb') as stream:md5=hashlib.file_digest(stream,lambda:hashlib.md5(usedforsecurity=False)).digest()
    if base64.b64encode(md5).decode()!=download['provider_md5_base64']:raise ValueError('provider MD5 differs')
    count=0;unknown=0;ids=set()
    with (a.root/'miap-training-image-information.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);count+=1
            if row['id'] in ids or row['role'] is not None or row['training_eligible']:raise ValueError('duplicate or admitted source')
            ids.add(row['id']);rotation=row['rotation_degrees']
            if rotation is None:unknown+=1
            elif type(rotation) is not int or rotation not in (0,90,180,270):raise ValueError('invalid normalized rotation')
    if count!=70000 or unknown!=9541:raise ValueError('joined record counts differ')
    paths=['justfile','docs/experiments/MIAP_METADATA.md','benchmarks/miap_image_information_2026-11-09.md',
           'tools/composition/join_miap_image_information.py','tools/composition/freeze_miap_image_information.py','tools/composition/artifact_identity.py']
    producers={}
    for name in paths:
        target=a.root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
        producers[name]=subprocess.check_output(['git','log','-1','--format=%H','--',name],text=True).strip()
    initial=a.root/'source/initial-joiner.py'
    initial.write_bytes(subprocess.check_output(['git','show','4c7d51fa:tools/composition/join_miap_image_information.py']))
    (a.root/'logs').mkdir()
    for pattern in ['zensally-miap-image-information*20261109.log','zensally-miap-image-information*20261109.time']:
        for path in Path('/home/lilith/tmp').glob(pattern):shutil.copy2(path,a.root/'logs'/path.name)
    manifest=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        source_producers=producers,parent_packet_sha256=summary['metadata_packet_sha256'],
        metadata_only=True,images_acquired=0,source_roles_assigned=False,training_eligible=False,
        original_observations_retained=True,face_coverage_certified=False,human_artistic_labels=False,paid_model_api_calls=0)
    (a.root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(a.root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(a.root.rglob('*')) if p.is_file()]
    total=sum(r['bytes'] for r in entries)
    if total>768*1024**2:raise ValueError('768MiB storage allowance exceeded; preserve output')
    sums=a.root/'SHA256SUMS.json'
    with sums.open('x') as stream:stream.write(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(members=len(entries),payload_bytes=total,manifest_sha256=sha(sums))))


if __name__=='__main__':main()
