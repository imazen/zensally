#!/usr/bin/env python3
"""Archive completed generated originals independently of live crop-reference work."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from sunburst_generate import sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    verification=json.loads((a.run/'expansion-native-verification.json').read_text())
    compression=json.loads((a.run/'expansion-jxl/summary.json').read_text())
    review=json.loads((a.run/'expansion/review.json').read_text())
    if verification['images']!=28 or verification['pending_attempts'] or len(review['notes'])!=28 or review.get('stage')=='pending-visual-review':raise ValueError('complete verified and inspected28 originals required')
    if compression['images']!=28 or any(not r['variants']['lossless']['lossless_exact'] or not r['variants']['lossless']['independent_exact'] for r in compression['records']):raise ValueError('all28 independent lossless pixel checks required')
    a.output.mkdir(parents=True,exist_ok=False)
    for name in ['expansion','expansion-jxl']:
        shutil.copytree(a.run/name,a.output/name)
    for name in ['authorization.json','api-models.json','brand-search.json','expansion-plan.json','plan-invalid-pixel-count.bak.json','expansion-native-verification.json','expansion-task-policy.json']:
        shutil.copy2(a.run/name,a.output/name)
    (a.output/'carry').mkdir();shutil.copy2(a.run/'pilot/budget.jsonl',a.output/'carry/budget.jsonl')
    for job in json.loads((a.run/'expansion/plan.json').read_text()):
        record=json.loads((a.run/'expansion'/(job['id']+'.json')).read_text());entry=a.run/'image-cache'/record['key']
        shutil.copytree(entry,a.output/'image-cache'/entry.name)
    producers=['justfile','tools/crop-first/sunburst_generate.py','tools/crop-first/sunburst_verify.py','tools/crop-first/sunburst_costs.py',
        'tools/crop-first/sunburst_subject_expansion.py','tools/crop-first/sunburst_subject_pilot.py','tools/crop-first/sunburst_commerce_v2.py',
        'tools/crop-first/sunburst_recompress.py','tools/crop-first/sunburst_dataset.py','tools/crop-first/sunburst_gallery.py',
        'tools/crop-first/sunburst_browser_check.mjs','tools/crop-first/browser_check.py','tools/crop-first/freeze_subject_expansion.py']
    for name in producers:
        target=a.output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    (a.output/'source/original-generation-plan-57e1ea5c.py').write_bytes(subprocess.check_output(['git','show','57e1ea5c:tools/crop-first/sunburst_subject_expansion.py']))
    (a.output/'logs').mkdir()
    for prefix in ['expansion','expansion-valid','expansion-jxl','expansion-browser']:
        for path in (Path.home()/'tmp').glob('zensally-subject-v4-'+prefix+'-20261109.*'):shutil.copy2(path,a.output/'logs'/path.name)
    manifest=dict(sources=28,generation_cost_usd=verification['new_published_rate_usd'],generation_total_with_pilot_usd=verification['total_published_rate_usd'],
        prior_pilot_packet_sha256='5c4494e41afee6f79c3ca7394576b7e8e27978fdf72ceca2af6ea657aa7f4947',
        original_generation_commit='57e1ea5c',source_revision=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        labels_admitted=False,face_safety_certified=False,includes_later_vision_calls_or_training=False)
    (a.output/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(a.output)),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(a.output.rglob('*')) if p.is_file()]
    (a.output/'SHA256SUMS.json').write_text(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(members=len(entries),bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha((a.output/'SHA256SUMS.json').read_bytes()))))

if __name__=='__main__':main()

