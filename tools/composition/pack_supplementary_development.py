#!/usr/bin/env python3
"""Preserve new acquired people, role admission and relationship evidence together."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['people','review','inventory','relationships','admission','binaries','output']:
        p.add_argument(name,type=Path)
    args = p.parse_args()
    source = json.loads((args.people/'sources.json').read_text())
    cohort = json.loads((args.admission/'sources.json').read_text())
    relationships = json.loads((args.relationships/'summary.json').read_text())
    if (len(source)!=8 or cohort['new_development_sources']!=15 or cohort['fitting_sources']!=0
            or relationships['exact_conflicts'] or relationships['pairs']!=21582):
        raise ValueError('complete bounded source/admission evidence required')
    if sum(p.stat().st_size for p in args.people.rglob('*') if p.is_file())>8*1024**3:
        raise ValueError('declared input storage allowance exceeded')
    args.output.mkdir(parents=True,exist_ok=False)
    for name,root in [('inputs',args.people),('review',args.review),('metadata-inventory',args.inventory),
                      ('relationships',args.relationships),('admission',args.admission)]:
        shutil.copytree(root,args.output/name)
    (args.output/'binaries').mkdir()
    for name in ['zensally-crop-decode','crop_label_pairs','zensally-crop-preview-reference-20261108']:
        shutil.copy2(args.binaries/name,args.output/'binaries'/name)
    files = [Path('justfile')]
    files += [Path('tools/composition')/n for n in ['artifact_identity.py','inventory_crop_source_expansion.py',
        'prepare_people_expansion.py','verify_owned_pixels.py','screen_supplementary_relationships.py',
        'audit_unsplash_relationships.py','freeze_supplementary_development.py','pack_supplementary_development.py']]
    files += [Path('tools/crop-first')/n for n in ['acquire_current.py','acquisition_oracle.py','corpus.py']]
    files += [Path('docs/experiments')/n for n in ['CROP_SOURCE_EXPANSION.md','PEOPLE_EXPANSION_SCREEN.md','SUPPLEMENTARY_DEVELOPMENT.md']]
    files += [Path('benchmarks/supplementary_development_2026-11-09.md'),Path('tools/crop-decode/src/main.rs'),
              Path('tools/crop-decode/Cargo.toml'),Path('tools/crop-decode/Cargo.lock'),Path('tools/crop-preview/src/main.rs')]
    for path in files:
        target = args.output/'source'/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    (args.output/'logs').mkdir()
    for pattern in ['zensally-people-expansion*20261109.*','zensally-source-expansion*20261109.log',
                    'zensally-supplementary*20261109.log']:
        for path in Path('/home/lilith/tmp').glob(pattern):
            shutil.copy2(path,args.output/'logs'/path.name)
    identity = dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        newly_acquired_sources=8,new_development_sources=15,new_fitting_sources=0,labels_admitted=False,
        input_root_mapping={str(args.people):'inputs'},
        prior_owned_input_packet_sha256='1d884d295ed9e958a74d18aaf105aedb896aa493b88eb3542dc60004ccc5e16f',
        prior_owned_content_packet_sha256='d11ec3aa607f2ab92250465115d46a919142864c2dc2a86d062f6a364ed52d71',
        photographic_registry_sha256='6f8d318cc870248dcc2400a235fcbaa5f7f0edf47d575d965bf5c72b5684d2f6',
        development_manifest_sha256=sha(args.admission/'sources.json'),
        prior_roles_changed=False,canonical_buckets_changed=False,canonical_holdout_pixels_opened=False,
        paid_model_api_calls=0,diagnostic_maps_saved=False,face_coverage_certified=False)
    (args.output/'_MANIFEST.json').write_text(json.dumps(identity,indent=2)+'\n')
    entries = [dict(path=str(p.relative_to(args.output)),bytes=p.stat().st_size,sha256=sha(p))
               for p in sorted(args.output.rglob('*')) if p.is_file()]
    sums = args.output/'SHA256SUMS.json'
    sums.write_text(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(members=len(entries),payload_bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha(sums))))


if __name__ == '__main__':
    main()
