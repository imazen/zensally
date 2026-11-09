#!/usr/bin/env python3
"""Bind frozen retention states and a metadata-only supplementary null mapping."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['heads','cohort','references','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    if sha(a.heads/'SHA256SUMS.json')!='3e603f65c94814b360fe8bee64484cffe07c39024ef32a4dda6e062c85a1698a':raise ValueError('frozen retention heads differ')
    if sha(a.cohort)!='a52144833a5824386577f63626e599f8dc602f8b77ee9d4d7f0c183574cebf70':raise ValueError('supplementary source roles differ')
    if sha(a.references/'SHA256SUMS.json')!='9fe44aa3645d379ab1ab925731f4710674f84b03171908026b1b3210a553e9f4':raise ValueError('cached supplementary references differ')
    old=json.loads((a.heads/'supplementary-selection-plan.json').read_text())
    sources=json.loads(a.cohort.read_text())['sources']
    if len(sources)!=15 or len({s['family_id'] for s in sources})!=15 or any(s['fit_eligible'] or s['role']!='supplementary_dev' for s in sources):raise ValueError('fifteen supplementary development-only families required')
    for binding in old['fixed_models']:
        if sha(Path(binding['path']))!=binding['sha256']:raise ValueError('preselected head changed')
    groups=defaultdict(list)
    for i,s in enumerate(sources):groups[tuple(s['valid_rect'])].append(i)
    donors=list(range(15))
    for indices in groups.values():
        ordered=sorted(indices,key=lambda i:hashlib.sha256(('retention-supplement-null-v1\0'+sources[i]['id']).encode()).hexdigest())
        for i,j in zip(ordered,ordered[1:]+ordered[:1]):donors[i]=j
    a.output.mkdir(parents=True,exist_ok=False)
    value=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        fixed_models=old['fixed_models'],original_selection_plan_sha256=sha(a.heads/'supplementary-selection-plan.json'),
        cohort_sha256=sha(a.cohort),reference_packet_sha256=sha(a.references/'SHA256SUMS.json'),sources=sources,
        source_permutation=donors,moved_sources=sum(i!=j for i,j in enumerate(donors)),
        permutation='SHA256 order then rotate within exact letterbox shape; metadata only, before new predictions',
        offer_threshold=.5,model_selection_changed=False,new_gradients=False,new_test_inference=False,
        paid_model_api_calls=0,face_coverage_certified=False)
    path=a.output/'plan.json';path.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(models=len(old['fixed_models']),sources=15,moved_sources=value['moved_sources'],plan_sha256=sha(path))))


if __name__=='__main__':main()
