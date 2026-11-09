#!/usr/bin/env python3
"""Freeze bounded people witnesses, independent readbacks and negative findings."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    if (a.root/'SHA256SUMS.json').exists():raise ValueError('witness packet already frozen')
    protocol=json.loads((a.root/'protocol.json').read_text());review=json.loads((a.root/'review-and-relationships.json').read_text())
    browser=json.loads((a.root/'browser-miap-witness.json').read_text());summary=json.loads((a.root/'summary.json').read_text())
    if (summary['sources']!=12 or summary['independent_png_readbacks']!=12 or summary['training_eligible']
        or browser['decoded']!=12 or browser['boxes']!=46 or browser['exceptions'] or review['pairs']!=3102
        or review['labels_admitted'] or review['out_of_scope_document_sources']!=1 or review['depiction_target_quarantines']!=1):raise ValueError('complete unadmitted witness review required')
    for identifier in protocol['selected_ids']:
        folder=a.root/identifier.split(':')[1];row=json.loads((folder/'record.json').read_text())
        if not row['independent_png_rgb_exact'] or row['source']['role']!='fit' or row['training_eligible']:raise ValueError('witness eligibility/readback differs')
        if sha(folder/'source.jpg')!=row['download']['sha256'] or any(sha(folder/name)!=row[name+'_sha256'] for name in ['preview.png','preview.rgb']):raise ValueError('witness bytes differ')
    paths=['justfile','docs/experiments/MIAP_PEOPLE_PILOT.md','benchmarks/miap_witnesses_2026-11-09.md',
        'tools/composition/acquire_miap_witnesses.py','tools/composition/review_miap_witnesses.py',
        'tools/composition/miap_witness_browser_check.mjs','tools/crop-first/browser_check.py',
        'tools/composition/freeze_miap_witnesses.py','tools/composition/artifact_identity.py','tools/composition/audit_unsplash_relationships.py']
    producers={}
    for name in paths:
        target=a.root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
        producers[name]=subprocess.check_output(['git','log','-1','--format=%H','--',name],text=True).strip()
    (a.root/'logs').mkdir()
    for pattern in ['zensally-miap-witness*20261109.log','zensally-miap-witness*20261109.time']:
        for path in Path('/home/lilith/tmp').glob(pattern):shutil.copy2(path,a.root/'logs'/path.name)
    manifest=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        source_producers=producers,source_role='fit',source_roles_changed=False,training_eligible=False,
        complete_face_coverage_certified=False,human_artistic_labels=False,paid_model_api_calls=0,
        diagnostic_maps_saved=False,body_region_overlays_saved=True,screenshot_budget_bytes=8*1024**2,
        parent_plan_packet_sha256='133daebb418f83587f4f1077c59ba7a54a80e088626ba9c1c896a416b4d880ab')
    (a.root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(a.root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(a.root.rglob('*')) if p.is_file()]
    total=sum(r['bytes'] for r in entries)
    if total>128*1024**2:raise ValueError('128MiB witness allowance exceeded; preserve output')
    sums=a.root/'SHA256SUMS.json'
    with sums.open('x') as stream:stream.write(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(members=len(entries),payload_bytes=total,manifest_sha256=sha(sums))))


if __name__=='__main__':main()
