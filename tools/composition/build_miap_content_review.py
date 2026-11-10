#!/usr/bin/env python3
"""Expose every acquired source and person box, with caption/relationship warnings."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['acquired','captions','relationships','output']:p.add_argument(name,type=Path)
    a=p.parse_args();manifest=json.loads((a.acquired/'manifest.json').read_text());rows=manifest['rows']
    captions=json.loads((a.captions/'pilot-summary.json').read_text());by_id={r['id']:r for r in captions['calls']}
    relationships=json.loads((a.relationships/'summary.json').read_text());flagged=set(relationships['flagged_new_sources'])
    if len(rows)!=384 or captions['completed']!=384 or len(by_id)!=384 or relationships['pairs']!=170688 or Counter(r['source']['role'] for r in rows)!=Counter(fit=288,dev=96):raise ValueError('complete fixed fit/development screen required')
    if {r['id'] for r in rows}!=by_id.keys():raise ValueError('source/caption join differs')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'previews').mkdir()
    inputs={r['id']:i for i,r in enumerate(rows)}
    rows=sorted(rows,key=lambda r:(r['id'] not in flagged,not by_id[r['id']]['invalid'],not bool(by_id[r['id']]['review_hints']['document']),not bool(by_id[r['id']]['review_hints']['depiction']),inputs[r['id']]))
    sources=[];payload=0
    for index,row in enumerate(rows):
        key=row['id'].split(':')[1];original=a.acquired/key/'preview.png';preview=a.output/'previews'/(key+'.png')
        if sha(original)!=row['preview.png_sha256'] or row['training_eligible']:raise ValueError('owned preview/admission differs')
        shutil.copy2(original,preview);payload+=preview.stat().st_size
        if payload>512*1024**2:raise ValueError('512MiB copied-preview budget exceeded')
        raw=row['source']['source_information']['source'];width,height=row['header']['size'];call=by_id[row['id']]
        warnings=[]
        if row['id'] in flagged:warnings.append('RELATIONSHIP FLAG: quarantine')
        if call['invalid']:warnings.append('CAPTION INVALID: manual review')
        if call['review_hints']['document']:warnings.append('DOCUMENT HINT')
        if call['review_hints']['depiction']:warnings.append('DEPICTION HINT')
        sources.append(dict(id=row['id'],index=index,sheet=index//24,original_acquisition_index=inputs[row['id']],
            role=row['source']['role'],family_id=row['source']['family_id'],native_size=[width,height],
            preview=str(preview.relative_to(a.output)),preview_size=[row['geometry']['width'],row['geometry']['height']],
            preview_sha256=sha(preview),rgb_sha256=row['preview.rgb_sha256'],
            source=dict(provider_id=key,source_url=raw['OriginalLandingURL'],photographer=raw['Author'],author_url=raw['AuthorProfileURL'],listed_license=raw['License']),
            regions=[[b['box'][0]*width,b['box'][1]*height,b['box'][2]*width,b['box'][3]*height] for b in row['boxes']],
            region_flags=[b['flags'] for b in row['boxes']],group_cell=row['source']['group_cell'],
            query=' · '.join(warnings),caption=call['caption'],caption_invalid=call['invalid'],
            relationship_flagged=row['id'] in flagged,content_reviewed=False,training_eligible=False,face_coverage_certified=False))
    data=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        dataset_kind='miap-fit-dev',sources=sources,sheet_size=24,columns=4,preview_display_height=210,
        screenshot_budget_bytes=96*1024**2,preview_asset_bytes=payload,
        acquisition_manifest_sha256=sha(a.acquired/'manifest.json'),caption_summary_sha256=sha(a.captions/'pilot-summary.json'),
        relationship_summary_sha256=sha(a.relationships/'summary.json'),fit_sources=288,dev_sources=96,test_sources=0,
        source_roles_changed=False,training_admission_changed=False,paid_model_api_calls=0)
    (a.output/'sources.json').write_text(json.dumps(data,indent=2)+'\n')
    template=Path(__file__).with_name('unsplash_content_audit.html').read_text()
    (a.output/'index.html').write_text(template.replace('/*DATA*/null',json.dumps(data).replace('</','<\\/')))
    print(json.dumps(dict(sources=len(sources),sheets=16,person_box_records=sum(len(s['regions']) for s in sources),copied_preview_bytes=payload,caption_invalid_sources=sum(s['caption_invalid'] for s in sources),relationship_flagged_sources=len(flagged))))


if __name__=='__main__':main()
