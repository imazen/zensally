#!/usr/bin/env python3
"""Freeze completed local region scouting and matched weak-target fits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def audit_responses(root, count):
    files = sorted((root/'evaluation/responses').glob('*.json'))
    if len(files) != count:
        raise ValueError('unique generation count differs')
    boxes = 0
    for path in files:
        row = json.loads(path.read_text())
        contract = row['contract']
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',',':')).encode()).hexdigest()
        key = contract['source_id'].removeprefix('unsplash:')
        if (digest != path.stem or digest != row['contract_sha256'] or row['postprocess_error']
                or row['token_cap_reached'] or not row['ended_with_eos']
                or sha(root/'evaluation/inputs'/(key+'.rgb')) != contract['owned_input_sha256']):
            raise ValueError('cached generation is not complete and source bound')
        w,h = contract['native_size']
        for value in row['parsed'].values():
            if not isinstance(value,dict):
                continue
            if len(value.get('bboxes',[])) != len(value.get('labels',[])):
                raise ValueError('parsed label/geometry count differs')
            for b in value.get('bboxes',[]):
                if not (len(b)==4 and all(math.isfinite(v) for v in b)
                        and 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):
                    raise ValueError('invalid native region geometry')
                boxes += 1
    return boxes


def freeze(root, manifest):
    if (root/'SHA256SUMS.json').exists():
        raise ValueError('packet already frozen')
    paths = [Path('justfile'),Path('training/composition_extent.py'),Path('tools/crop-first/browser_check.py')]
    paths += [Path('tools/composition')/n for n in ['artifact_identity.py','acquire_florence_regions.py',
        'probe_florence_regions.py','analyze_florence_regions.py','prepare_florence_primary56.py',
        'cache_florence_extents.py','rebind_florence_features.py','fit_extents.py','fit_heads.py',
        'analyze_florence_extents.py','build_florence_extent_gallery.py','florence_extent_gallery.html',
        'florence_extent_browser_check.mjs','freeze_florence_evidence.py','analyze_efficientdet_family.py',
        'region_retention.py']]
    paths += [Path('docs/experiments')/n for n in ['FLORENCE_REGION_SUITABILITY.md','FLORENCE_EXTENT_FIT.md']]
    paths += [Path('benchmarks')/n for n in ['florence_regions_2026-11-09.md','florence_extents_2026-11-09.md']]
    producers = {}
    for path in paths:
        target = root/'source'/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        producers[str(path)] = subprocess.check_output(['git','log','-1','--format=%H','--',str(path)],text=True).strip()
    (root/'logs').mkdir()
    for path in Path('/home/lilith/tmp').glob('zensally-*florence*20261109.log'):
        shutil.copy2(path,root/'logs'/path.name)
    manifest.update(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        source_producers=producers, paid_model_api_calls=0, source_roles_changed=False,
        canonical_holdouts_opened=False, model_promoted=False, face_coverage_certified=False,
        human_artistic_quality_measured=False, diagnostic_maps_saved=False)
    (root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries = [dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p))
               for p in sorted(root.rglob('*')) if p.is_file()]
    sums = root/'SHA256SUMS.json'
    sums.write_text(json.dumps(entries,indent=2)+'\n')
    result = dict(root=str(root),members=len(entries),payload_bytes=sum(r['bytes'] for r in entries),
                  manifest_sha256=sha(sums))
    print(json.dumps(result),flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('scout',type=Path)
    p.add_argument('fits',type=Path)
    p.add_argument('original_cache',type=Path)
    p.add_argument('focal_cache',type=Path)
    p.add_argument('resizer',type=Path)
    args = p.parse_args()
    if any((r/'SHA256SUMS.json').exists() for r in [args.scout,args.fits]):
        raise ValueError('both input packets must be unfrozen')
    if audit_responses(args.scout,24) != 58 or audit_responses(args.fits,56) != 101:
        raise ValueError('registered output box totals differ')
    for stage,count in [('probe',1),('scout',18),('features',6)]:
        summary = json.loads((args.scout/'evaluation'/(stage+'-summary.json')).read_text())
        if summary['completed'] != count or summary['invalid_calls']:
            raise ValueError('incomplete region scouting')
    for directory in [args.fits/'extent-fits',args.fits/'focalnet/extent-fits']:
        progress = json.loads((directory/'progress.json').read_text())
        if progress['pid'] is not None or progress['completed_runs'] != 9 or progress['stage'] != 'complete':
            raise ValueError('fit process is not demonstrably complete')
    assets = json.loads((args.scout/'asset-identities.json').read_text())
    for file in assets['files']:
        path = args.scout/file['path']
        if sha(path) != file['sha256'] or path.stat().st_size != file['bytes']:
            raise ValueError('local model asset differs')
    browser = json.loads((args.fits/'review/browser-florence-extent.json').read_text())
    if browser['checks'] != 336 or browser['decoded'] != 56 or browser['failures'] or browser['exceptions']:
        raise ValueError('source-bound review check is not complete')
    (args.scout/'binaries').mkdir()
    shutil.copy2(args.resizer,args.scout/'binaries'/args.resizer.name)
    original = args.fits/'original-descriptors'
    original.mkdir()
    for name,path in [('small',args.original_cache),('focal',args.focal_cache)]:
        shutil.copy2(path,original/(name+'.pt'))
        shutil.copy2(path.parent/'admission.json',original/(name+'-admission.json'))
    scout = freeze(args.scout,dict(unique_calls=24,valid_native_boxes=58,
        assets_sha256=sha(args.scout/'asset-identities.json'),
        supplied_region_packet_sha256='dfce5f1dc0cbfc0ce3113bd327f780628c8c6ad5a77b6364e3a02099b6be9699'))
    freeze(args.fits,dict(unique_calls=56,valid_native_boxes=101,completed_fits=18,
        source_plan_sha256=sha(args.fits/'source-plan.json'),scout_packet_sha256=scout['manifest_sha256'],
        training_admission='48 fit-only noisy phrase-grounded primary unions; eight development targets supply selection only'))


if __name__ == '__main__':
    main()
