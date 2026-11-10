#!/usr/bin/env python3
"""Freeze both phrase variants, known-box audits and complete source-bound views."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    args = p.parse_args()
    if (args.root/'SHA256SUMS.json').exists():
        raise ValueError('packet already frozen')
    for folder,stage,count,plan in [('references','photo87',87,'source-plan.json'),
        ('people-references','photo-people20',20,'people-source-plan.json')]:
        root = args.root/folder
        summary = json.loads((root/(stage+'-summary.json')).read_text())
        progress = json.loads((root/'progress.json').read_text())
        sources = {s['id']:s for s in json.loads((args.root/plan).read_text())['sources']}
        if len(summary['calls'])!=count or summary['invalid_calls'] or progress['pid'] is not None or progress['completed']!=count:
            raise ValueError('completed cached generation required')
        for call in summary['calls']:
            path = root/'responses'/call['response']
            row = json.loads(path.read_text())
            contract = row['contract']
            digest = hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            source = sources[contract['source_id']]
            if (digest!=path.stem or digest!=row['contract_sha256'] or row['postprocess_error']
                    or row['token_cap_reached'] or not row['ended_with_eos']
                    or contract['source_plan_sha256']!=sha(args.root/plan)
                    or sha(root/'inputs'/(source['source']['provider_id']+'.rgb'))!=contract['owned_input_sha256']):
                raise ValueError('complete cached source-bound generation required')
    for folder,boxes in [('review',160),('review-generic',143)]:
        browser = json.loads((args.root/folder/'browser-photo-subjects.json').read_text())
        if browser['decoded']!=87 or browser['region_boxes']!=boxes or browser['exceptions']:
            raise ValueError('all source/region browser checks required')
    initial = json.loads((args.root/'reference-audit.json').read_text())
    generic = json.loads((args.root/'reference-audit-generic.json').read_text())
    if initial['counts']['all']['valid_boxes']!=160 or generic['counts']['all']['valid_boxes']!=143:
        raise ValueError('reviewed weak reference totals differ')
    assets = sum(f.stat().st_size for d in ['references','people-references'] for f in (args.root/d).rglob('*') if f.is_file())
    shots = sum(f.stat().st_size for d in ['review','review-generic'] for f in (args.root/d/'screenshots').glob('*.png'))
    if assets>256*1024**2 or shots>24*1024**2:
        raise ValueError('declared reference/screenshot storage allowance exceeded')
    paths = [Path('justfile'),Path('docs/experiments/PHOTO_SUBJECT_RETENTION.md'),
        Path('benchmarks/photo_subject_retention_2026-11-09.md'),Path('tools/crop-first/browser_check.py')]
    paths += [Path('tools/composition')/n for n in ['artifact_identity.py','prepare_photo_subject_queries.py',
        'probe_florence_regions.py','audit_photo_subject_references.py','freeze_photo_subject_references.py',
        'region_retention.py','audit_crop_feasibility.py','unsplash_content_audit.html','unsplash_content_browser_check.mjs']]
    producers = {}
    for path in paths:
        target = args.root/'source'/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        producers[str(path)] = subprocess.check_output(['git','log','-1','--format=%H','--',str(path)],text=True).strip()
    # The first query plan/probe and audit used earlier versions. Preserve those
    # exact sources as well, rather than attributing old bytes to current code.
    for commit,names in [('5727d090',['prepare_photo_subject_queries.py']),
        ('70f78b1b',['probe_florence_regions.py']),('f0b4b96a',['audit_photo_subject_references.py'])]:
        for name in names:
            relative='tools/composition/'+name
            target=args.root/'producer-sources'/commit/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(subprocess.check_output(['git','show',commit+':'+relative]))
    (args.root/'logs').mkdir()
    for pattern in ['zensally-photo-subject*20261109.log','zensally-photo-people*20261109.log']:
        for path in Path('/home/lilith/tmp').glob(pattern):
            shutil.copy2(path,args.root/'logs'/path.name)
    for name in ['http.log','chromium.log']:
        path=Path('/home/lilith/tmp/zensally-photo-subject-generic-browser-20261109')/name
        shutil.copy2(path,args.root/'logs'/('failed-browser-'+name))
    manifest=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        source_producers=producers,unique_generations=107,primary_sources=87,primary_fit_sources=72,
        primary_development_sources=15,initial_boxes=160,generic_people_substitution_boxes=143,
        reference_storage_bytes=assets,screenshot_bytes=shots,
        rank_packet_sha256='d2a9e3c1b2f67a0db01088c52b865895467b1e86c6fbc5588f9e2a15a49b1b8e',
        teacher_packet_sha256='462488a1a7893e6653a1c82e602aeb3362c5a0c8c87cf1a9315adefc66644a9c',
        content_packet_sha256='2440fb879fe7bd8df6dc0e6fb740459627bf5c221962a8fc2e07132f0e02ac6a',
        source_roles_changed=False,reference_labels_admitted=False,new_test_inference=False,
        paid_model_api_calls=0,face_coverage_certified=False,human_artistic_quality_measured=False,
        diagnostic_maps_saved=False,model_promoted=False)
    (args.root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(f.relative_to(args.root)),bytes=f.stat().st_size,sha256=sha(f))
             for f in sorted(args.root.rglob('*')) if f.is_file()]
    sums=args.root/'SHA256SUMS.json'
    with sums.open('x') as stream:
        stream.write(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(members=len(entries),payload_bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha(sums))))


if __name__=='__main__':
    main()
