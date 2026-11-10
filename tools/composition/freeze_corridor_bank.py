#!/usr/bin/env python3
"""Seal scalar bank-support followup separately from the paid pilot snapshot."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from artifact_identity import sha
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from artifacts import verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    if (a.root/'SHA256SUMS.json').exists():raise ValueError('sealed followup must not be rewritten')
    summary=json.loads((a.root/'summary.json').read_text())
    if summary['trained'] or summary['api_calls'] or summary['model_promoted'] or summary['test_inference']:raise ValueError('fixed scalar followup required')
    for name in ['justfile','tools/composition/audit_corridor_bank.py','tools/composition/freeze_corridor_bank.py',
        'tools/composition/artifact_identity.py','tools/crop-first/crop_corridors.py','tools/crop-first/test_crop_corridors.py',
        'tools/crop-first/artifacts.py','benchmarks/crop_label_quality_2026-11-09.md']:
        target=a.root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    (a.root/'logs').mkdir()
    for path in (Path.home()/'tmp').glob('zensally-corridor-bank-support*-20261109.log'):shutil.copy2(path,a.root/'logs'/path.name)
    manifest=dict(source_revision=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        parent_packets=dict(pilot='92838a0d3c83aa18b0a843bdbefc4087709d1350c2b10e662ca5563577938396',
            attention='86aacbfb10ce299649029e4805efc73073fb542feb4b2f2401cc0d9cb8c32ef3'),
        api_calls=0,trained=False,model_promoted=False,test_inference=False,excellent_intervals_admitted=0,
        usable_intervals_are_diagnostic_only=True,quality_outside_intervals_is_unknown=True)
    (a.root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(a.root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(a.root.rglob('*')) if p.is_file()]
    (a.root/'SHA256SUMS.json').write_text(json.dumps(entries,indent=2)+'\n');verify(a.root,entries)
    print(json.dumps(dict(files=len(entries),bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha(a.root/'SHA256SUMS.json'))))


if __name__=='__main__':main()
