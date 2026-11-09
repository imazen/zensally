#!/usr/bin/env python3
"""Verify inherited source roles and freeze the corrected group-aware sampling."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('parent',type=Path);a=p.parse_args()
    if (a.root/'SHA256SUMS.json').exists():raise ValueError('group plan already frozen')
    protocol=json.loads((a.root/'protocol.json').read_text());summary=json.loads((a.root/'summary.json').read_text())
    if sha(a.parent/'SHA256SUMS.json')!=protocol['parent_packet_sha256']:raise ValueError('parent manifest differs')
    members={r['path']:r for r in json.loads((a.parent/'SHA256SUMS.json').read_text())}
    if sha(a.parent/'registry.jsonl')!=members['registry.jsonl']['sha256']:raise ValueError('inherited source registry payload differs')
    with (a.parent/'registry.jsonl').open() as stream:bindings={r['id']:(r['role'],r['family_id']) for r in map(json.loads,stream)}
    with (a.root/'cells.jsonl').open() as stream:cells=list(map(json.loads,stream))
    if len(cells)!=70000 or {r['id'] for r in cells}!=bindings.keys():raise ValueError('incomplete inherited registry')
    if any((r['role'],r['family_id'])!=bindings[r['id']] for r in cells):raise ValueError('existing source role/component changed')
    pilot=json.loads((a.root/'pilot.json').read_text())
    if len(pilot)!=480 or len({r['family_id'] for r in pilot})!=480 or summary['pilot_sources']!={'dev':96,'fit':288,'test':96}:raise ValueError('balanced group-aware pilot incomplete')
    if any(len([r for r in pilot if r['role']==role and r['group_cell']==cell])!=quota for role,quota in [('fit',18),('dev',6),('test',6)] for cell in {r['group_cell'] for r in pilot}):raise ValueError('cell quotas differ')
    if len({r['group_cell'] for r in pilot})!=16 or summary['shortfalls'] or protocol['previous_source_roles_changed'] or protocol['training_eligible']:raise ValueError('sampling contract differs')
    if any(r['id'] in {'openimages:7df0a4df4df67865','openimages:46953247dc4e523a'} or r['acquisition_exclusions'] or r['training_eligible'] for r in pilot):raise ValueError('excluded source entered pilot')
    paths=['justfile','docs/experiments/MIAP_PEOPLE_PILOT.md','benchmarks/miap_group_pilot_2026-11-09.md',
        'tools/composition/plan_miap_group_pilot.py','tools/composition/test_miap_source_groups.py',
        'tools/composition/freeze_miap_group_pilot.py','tools/composition/artifact_identity.py']
    producers={}
    for name in paths:
        target=a.root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
        producers[name]=subprocess.check_output(['git','log','-1','--format=%H','--',name],text=True).strip()
    (a.root/'logs').mkdir()
    for suffix in ['log','time']:
        path=Path('/home/lilith/tmp/zensally-miap-group-plan-20261109.'+suffix);shutil.copy2(path,a.root/'logs'/path.name)
    manifest=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        source_producers=producers,parent_packet_sha256=protocol['parent_packet_sha256'],witness_packet_sha256=protocol['witness_packet_sha256'],
        all_70000_source_roles_and_components_exact=True,training_eligible=False,images_acquired=0,
        human_artistic_labels=False,complete_face_coverage_certified=False,paid_model_api_calls=0)
    (a.root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(a.root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(a.root.rglob('*')) if p.is_file()]
    total=sum(r['bytes'] for r in entries)
    if total>64*1024**2:raise ValueError('64MiB derived planning allowance exceeded')
    sums=a.root/'SHA256SUMS.json'
    with sums.open('x') as stream:stream.write(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(members=len(entries),payload_bytes=total,manifest_sha256=sha(sums),pilot_sha256=sha(a.root/'pilot.json'))))


if __name__=='__main__':main()
