#!/usr/bin/env python3
"""Freeze failed person-to-crop transfer with complete input/state provenance."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    if (root/'SHA256SUMS.json').exists():raise ValueError('transfer packet already frozen')
    report=json.loads((root/'analysis/summary.json').read_text());runs=json.loads((root/'fits/summary.json').read_text())
    if len(runs)!=12 or not report['all12_selected_native_replays_exact'] or report['registered_crop_transfer_gate_passed']:
        raise ValueError('all twelve negative controls required')
    for run in runs:
        if run['best_epoch']!=0 or sha(root/'fits'/(run['arm']+'-seed'+str(run['seed']))/'best.pt')!=run['best_weights_sha256']:
            raise ValueError('selected negative result differs')
    paths=['justfile','DATA_PROVENANCE.md','docs/experiments/MIAP_COMPOSITION_TRANSFER.md','benchmarks/miap_composition_transfer_2026-11-09.md',
        'training/person_relationships.py','training/test_person_relationships.py','training/local_rank_distillation.py',
        'training/crop_regions.py','training/crop_ranker.py','training/model_v3ds.py','training/crop_fit.py','training/composition_extent.py',
        'tools/composition/fit_unsplash_ranks.py','tools/composition/cache_miap_transfer.py','tools/composition/fit_miap_transfer.py',
        'tools/composition/analyze_miap_transfer.py','tools/composition/analyze_unsplash_ranks.py','tools/composition/artifact_identity.py',
        'tools/composition/fit_extents.py','tools/composition/freeze_miap_transfer.py']
    producers={}
    for name in paths:
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
        producers[name]=subprocess.check_output(['git','log','-1','--format=%H','--',name],text=True).strip()
    (root/'logs').mkdir()
    for prefix in ['unit','unit-two-way','cache','help','fits','analysis']:
        for source in Path('/home/lilith/tmp').glob('zensally-miap-transfer-'+prefix+'-20261109.*'):
            shutil.copy2(source,root/'logs'/source.name)
    manifest=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        producers=producers,fit_protocol_sha256=sha(root/'fits/protocol.json'),auxiliary_cache_sha256=sha(root/'cache/cache.pt'),
        parent_acquisition_sha256='93f63d69180d82f197e34a733005118de260eda728e18998ced3194959581bb1',
        parent_review_sha256='75a75ee2df2c85ecac6b18cf6c875d3bfb0c6d625bf26542efc57fcbb4d2bca9',
        original_rank_cache_sha256='2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031',
        original_initializer_sha256='c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df',
        all_fits_complete=True,crop_transfer_qualified=False,new_test_inference=False,face_coverage_certified=False,paid_model_api_calls=0)
    (root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file()]
    total=sum(r['bytes'] for r in entries)
    if total>3*1024**3:raise ValueError('cache plus retained-fit budget exceeded')
    with (root/'SHA256SUMS.json').open('x') as stream:stream.write(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(members=len(entries),payload_bytes=total,manifest_sha256=sha(root/'SHA256SUMS.json'))))


if __name__=='__main__':main()
