#!/usr/bin/env python3
"""Freeze source metadata, quarantines and hyperlink-preserving documentation."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    if (a.root/'SHA256SUMS.json').exists():raise ValueError('metadata packet already frozen')
    summary=json.loads((a.root/'summary.json').read_text());progress=json.loads((a.root/'progress.json').read_text())
    quarantine=json.loads((a.root/'invalid-geometry.json').read_text())
    if (progress['pid'] is not None or progress['stage']!='complete' or summary['official_training_images']!=70000
        or summary['person_box_records']!=373386 or summary['invalid_geometry_records']!=5
        or summary['quarantined_images']!=5 or not quarantine['affected_images_quarantined']
        or quarantine['strict_bounds_relaxed'] or summary['images_acquired']):raise ValueError('complete metadata-only validation required')
    for record in summary['source_records']:
        name=next(p for p in a.root.iterdir() if p.is_file() and p.name.endswith('.http.json') and json.loads(p.read_text()).get('sha256')==record['sha256'])
        raw=a.root/name.name.removesuffix('.http.json')
        if sha(raw)!=record['sha256'] or raw.stat().st_size!=record['bytes']:raise ValueError('captured source bytes differ')
    paths=[Path('justfile'),Path('docs/experiments/MIAP_METADATA.md'),Path('benchmarks/miap_metadata_2026-11-09.md'),Path('tools/composition/acquire_miap_metadata.py'),Path('tools/composition/freeze_miap_metadata.py'),Path('tools/composition/artifact_identity.py')]
    producers={}
    for path in paths:
        target=a.root/'source'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        producers[str(path)]=subprocess.check_output(['git','log','-1','--format=%H','--',str(path)],text=True).strip()
    initial=a.root/'source/initial-acquirer.py';initial.write_bytes(subprocess.check_output(['git','show','c96babf7:tools/composition/acquire_miap_metadata.py']))
    (a.root/'logs').mkdir()
    for pattern in ['zensally-miap-metadata*20261109.log','zensally-miap-metadata*20261109.time']:
        for path in Path('/home/lilith/tmp').glob(pattern):shutil.copy2(path,a.root/'logs'/path.name)
    manifest=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),source_producers=producers,
        official_training_images=70000,person_box_records=373386,invalid_geometry_records=5,quarantined_images=5,
        metadata_only=True,images_acquired=0,source_roles_assigned=False,training_eligible=False,
        demographic_attributes_used=False,paid_model_api_calls=0,canonical_holdouts_opened=False,
        face_coverage_certified=False,human_artistic_labels=False,diagnostic_maps_saved=False)
    (a.root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(a.root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(a.root.rglob('*')) if p.is_file()]
    if sum(r['bytes'] for r in entries)>128*1024**2:raise ValueError('metadata storage allowance exceeded; preserve files')
    sums=a.root/'SHA256SUMS.json'
    with sums.open('x') as stream:stream.write(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(members=len(entries),payload_bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha(sums))))


if __name__=='__main__':main()
