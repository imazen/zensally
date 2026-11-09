#!/usr/bin/env python3
"""Compare all acquired people sources with available frozen fingerprints."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import subprocess

from artifact_identity import sha
from audit_unsplash_relationships import fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['acquired','binary','photographic','supplementary','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    for path,digest in [(a.binary,'3113e25a9ac4f037eac705445c598223771796491ad6b266889445abaeb0aef0'),
        (a.photographic,'e7220dd1db356ea3c1b00db864dfaf7baa896130a303a38e717fc678068f828d'),
        (a.supplementary,'0cb5e4abba51cec990062e2a18032723c1ed518671d6a4b38405376cab97d827')]:
        if sha(path)!=digest:raise ValueError('qualified fingerprint binary/reference differs')
    manifest=json.loads((a.acquired/'manifest.json').read_text());rows=manifest['rows'];summary=json.loads((a.acquired/'summary.json').read_text())
    if len(rows)!=384 or summary['independent_png_readbacks']!=384 or manifest['protocol']['plan_packet_sha256']!='e5baf11ae1673af8b1feb98e790cbd17f93976f5c37ebe3888af3d508bd0a5db':raise ValueError('complete source-bound pilot required')
    a.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        acquisition_manifest_sha256=sha(a.acquired/'manifest.json'),binary_sha256=sha(a.binary),
        reference_bindings={str(p):sha(p) for p in [a.photographic,a.supplementary]},threshold=10,
        pair_payload_budget_bytes=64*1024**2,training_eligible=False,paid_model_api_calls=0)
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    references=json.loads(a.photographic.read_text())+json.loads(a.supplementary.read_text())
    if len(references)!=253:raise ValueError('all available stored references required')
    hashes=[];exact=[]
    for row in rows:
        path=a.acquired/row['id'].split(':')[1]/'preview.rgb';rgb_sha=sha(path)
        if rgb_sha!=row['preview.rgb_sha256'] or row['source']['role'] not in {'fit','dev'} or row['training_eligible']:raise ValueError('source role/pixels/admission differs')
        g=row['geometry'];value=fingerprint(a.binary,path,g['width'],g['height'],g['width']*3)
        hashes.append(dict(id=row['id'],role=row['source']['role'],family_id=row['source']['family_id'],**value,rgb_sha256=rgb_sha,cvdf_sha256=row['download']['sha256']))
        for ref in references:
            if row['download']['sha256'] in {ref.get('original_sha256'),ref.get('encoded_sha256')} or rgb_sha in {ref.get('rgb_sha256'),ref.get('working_rgb_sha256')}:exact.append(dict(left=row['id'],right=ref['id']))
    pairs=[dict(left=left['id'],right=right['id'],kind='new_vs_existing',distance=(int(left['hash'],16)^int(right['hash'],16)).bit_count()) for left in hashes for right in references]
    pairs.extend(dict(left=left['id'],right=right['id'],kind='new_internal',distance=(int(left['hash'],16)^int(right['hash'],16)).bit_count(),roles=[left['role'],right['role']]) for left,right in combinations(hashes,2))
    if len(pairs)!=170688:raise ValueError('complete 384x253 plus internal pair coverage required')
    flags=[r for r in pairs if r['distance']<=10]
    affected=sorted({p[k] for p in flags+exact for k in ['left','right'] if p[k] in {r['id'] for r in hashes}})
    result=dict(sources=384,existing_references=253,pairs=len(pairs),cross_reference_pairs=97152,internal_pairs=73536,
        flags=flags,exact_matches=exact,flagged_new_sources=affected,threshold=10,
        possible_relationships_require_quarantine_or_review=True,complete_duplicate_recall_certified=False,
        canonical_holdout_pixels_opened=False,source_roles_changed=False,training_eligible=False,paid_model_api_calls=0)
    for name,value in [('fingerprints.json',hashes),('pair-distances.json',pairs),('summary.json',result)]:
        (a.output/name).write_text(json.dumps(value,indent=2)+'\n')
    if sum(p.stat().st_size for p in a.output.iterdir())>64*1024**2:raise ValueError('64MiB scalar-pair budget exceeded; preserve results')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
