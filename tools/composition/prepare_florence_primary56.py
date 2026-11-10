#!/usr/bin/env python3
"""Bind the existing56 primary queries to unchanged synthetic family/source roles."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['admission', 'runtime', 'queries', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    admission = json.loads(args.admission.read_text())
    runtime = json.loads(args.runtime.read_text())
    queries = json.loads(args.queries.read_text())['queries']
    by_id = {s['id']:s for s in runtime['images']}
    if len(admission['sources']) != 56 or set(queries) != {s['id'] for s in admission['sources']}:
        raise ValueError('exact legacy56/query coverage required')
    rows = []
    for source in admission['sources']:
        original = by_id[source['id']]
        if (not original['synthetic'] or source['rgb_sha256'] != original['rgb_sha256']
                or source['use'] not in ['fit','dev'] or sha(Path(original['rgb'])) != source['rgb_sha256']
                or Path(original['rgb']).stat().st_size != original['width']*original['height']*3):
            raise ValueError('legacy source role, owned RGB identity or layout differs')
        rows.append(dict(id=source['id'], role=source['use'], family_id=source['family_id'],
            rgb=original['rgb'], rgb_sha256=source['rgb_sha256'],
            native_size=[original['width'],original['height']], working_size=[original['width'],original['height']],
            source=dict(provider_id=source['id'], namespace='synthetic-generated'),
            query=queries[source['id']], legacy_source=source, runtime_source=original,
            face_coverage_certified=False, human_rated=False, training_labels_admitted=False))
    roles = Counter(r['role'] for r in rows)
    families = {role:{r['family_id'] for r in rows if r['role']==role} for role in roles}
    if roles != Counter(fit=48,dev=8) or len(families['fit']) != 27 or len(families['dev']) != 8 or families['fit'] & families['dev']:
        raise ValueError('unchanged48/8 sources and27/8 disjoint families required')
    result = dict(schema='zensally-florence-primary56-v1', sources=rows,
        build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        legacy_admission_sha256=sha(args.admission), runtime_manifest_sha256=sha(args.runtime),
        query_plan_sha256=sha(args.queries), source_roles_changed=False, canonical_holdouts_opened=False,
        teacher='florence-community/Florence-2-base@00921df66db728a9ceb750f5eca43e5c203a2051',
        training_labels_admitted=False, paid_model_api_calls=0)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(sources=56,fit=48,dev=8,fit_families=27,dev_families=8,plan_sha256=sha(args.output))))


if __name__ == '__main__':
    main()
