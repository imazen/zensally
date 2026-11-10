#!/usr/bin/env python3
"""Freeze metadata roles and sampling before pixel acquisition."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    if (a.root/'SHA256SUMS.json').exists():raise ValueError('pilot plan already frozen')
    summary=json.loads((a.root/'summary.json').read_text());pilot=json.loads((a.root/'pilot.json').read_text())
    if (summary['pilot_sources']!={'dev':96,'fit':288,'test':96} or len(pilot)!=480
        or len({r['family_id'] for r in pilot})!=480 or summary['images_acquired'] or summary['training_eligible']):
        raise ValueError('complete unadmitted metadata roles required')
    for row in pilot:
        if row['acquisition_exclusions'] or row['training_eligible'] or row['source_information']['rotation_degrees']!=0:
            raise ValueError('pilot selection includes excluded source')
    paths=['justfile','docs/experiments/MIAP_PEOPLE_PILOT.md','benchmarks/miap_people_pilot_2026-11-09.md',
        'tools/composition/plan_miap_people_pilot.py','tools/composition/test_miap_source_groups.py',
        'tools/composition/freeze_miap_pilot.py','tools/composition/artifact_identity.py']
    producers={}
    for name in paths:
        target=a.root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
        producers[name]=subprocess.check_output(['git','log','-1','--format=%H','--',name],text=True).strip()
    (a.root/'logs').mkdir()
    for suffix in ['log','time']:
        path=Path('/home/lilith/tmp/zensally-miap-pilot-plan-20261109.'+suffix);shutil.copy2(path,a.root/'logs'/path.name)
    manifest=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        source_producers=producers,roles_frozen_before_pixels=True,images_acquired=0,training_eligible=False,
        original_roles_changed=False,pilot_sha256=sha(a.root/'pilot.json'),complete_duplicate_recall_certified=False,paid_model_api_calls=0)
    (a.root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(a.root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(a.root.rglob('*')) if p.is_file()]
    total=sum(r['bytes'] for r in entries)
    if total>256*1024**2:raise ValueError('planning allowance exceeded; preserve artifacts')
    sums=a.root/'SHA256SUMS.json'
    with sums.open('x') as stream:stream.write(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(members=len(entries),payload_bytes=total,manifest_sha256=sha(sums),pilot_sha256=manifest['pilot_sha256'])))


if __name__=='__main__':main()
