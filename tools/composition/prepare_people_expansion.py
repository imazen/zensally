#!/usr/bin/env python3
"""Prepare a bounded screening-only acquisition from verified unused TRAIN metadata."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['inventory','corpus','output']:
        p.add_argument(name,type=Path)
    args = p.parse_args()
    if sha(args.inventory) != '9e0ccc5e65b60a66584f2442a6197f880fe2c44a0a4b04caa58055b7ac80fd7d':
        raise ValueError('registered unused-source metadata inventory differs')
    inventory = json.loads(args.inventory.read_text())
    people = [s for s in inventory['candidates'] if s['candidate_kind']=='people']
    if len(people)!=10 or any(s['bucket']!='train' or s['historical_eval_family']
            or s['metadata_status']!='unassigned_metadata_candidate' or s['new_role'] is not None for s in people):
        raise ValueError('all ten unassigned non-holdout people candidates required')
    selected,excluded = [],[]
    for source in people:
        if int(source['width'])*int(source['height'])>32_000_000:
            excluded.append(dict(id=source['id'],reason='above unchanged32MP decoder bound',
                                 native_pixels=int(source['width'])*int(source['height'])))
        else:
            selected.append(dict(source,use='screening',training_eligible=False,
                human_rated=False,face_safety_certified=False,screening_state='pending_visual_and_relationship_review'))
    if len(selected)!=8 or len(excluded)!=2:
        raise ValueError('registered eight bounded/two unacquired sources differ')
    if shutil.disk_usage(args.output.parent).free<12*1024**3:
        raise ValueError('registered disk headroom unavailable')
    args.output.mkdir(parents=True,exist_ok=False)
    protocol = json.loads((args.corpus/'protocol.json').read_text())
    protocol.update(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        expansion_inventory_sha256=sha(args.inventory),selected_uses=dict(screening=8),
        admission='screening only; no new fitting/development/TEST assignment',
        excluded_unacquired=excluded,maximum_native_pixels=32_000_000,
        artifact_allowance_bytes=8*1024**3,host_cap_gib=6,worker_threads=2,
        prior_roles_changed=False,new_training_labels=False,paid_model_api_calls=0)
    for name,value in [('sources.json',selected),('protocol.json',protocol)]:
        with (args.output/name).open('x') as stream:
            stream.write(json.dumps(value,indent=2)+'\n')
    (args.output/'links.jsonl').write_text(''.join(json.dumps(link)+'\n' for link in protocol['links']))
    print(json.dumps(dict(screening_sources=8,source_families=len({s['family_id'] for s in selected}),
        excluded_unacquired=excluded,declared_encoded_bytes=sum(int(s['bytes']) for s in selected),
        plan_sha256=sha(args.output/'sources.json'),new_fitting_or_development_assignments=0)))


if __name__ == '__main__':
    main()
