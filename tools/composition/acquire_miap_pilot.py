#!/usr/bin/env python3
"""Acquire the frozen 384-source fit/development people pilot with resumable bytes."""
import argparse
import atexit
from collections import Counter
import csv
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha
from acquire_miap_witnesses import fetch
from miap_owned_preview import prepare_preview

LIMIT=2*1024**3


def member(root,name):
    entries=json.loads((root/'SHA256SUMS.json').read_text());matches=[r for r in entries if r['path']==name]
    if len(matches)!=1 or sha(root/name)!=matches[0]['sha256']:raise ValueError('frozen packet member differs: '+name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','annotations','witnesses','pipeline','png_reader','parity','output']:p.add_argument(name,type=Path)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    for root,digest in [(a.plan,'e5baf11ae1673af8b1feb98e790cbd17f93976f5c37ebe3888af3d508bd0a5db'),
        (a.annotations,'77fd61540c7748b066383a4940c504358b9e48f6bc45982cb2a53536eb573bb6'),
        (a.witnesses,'8468972acdcd5edc34bf2f7042631d83bc104071cf2bd99e9147dd0f280316ad')]:
        if sha(root/'SHA256SUMS.json')!=digest:raise ValueError('frozen acquisition input differs')
    member(a.plan,'pilot.json');member(a.annotations,'train-boxes.csv')
    if sha(a.pipeline)!='6797c4f8e2e627921bd1fe61122e229325b8757538b9e554bcc28a9b39a9ab9a' or sha(a.png_reader)!='3346ccde58afc455194c22397ff3a5b025e4ceede6115f6795a0f8ad0c694c16':raise ValueError('qualified preview binaries differ')
    parity=json.loads(a.parity.read_text())
    if parity['sources']!=12 or parity['exact_sources']!=12:raise ValueError('complete shared-preview parity required')
    chosen=[r for r in json.loads((a.plan/'pilot.json').read_text()) if r['role'] in {'fit','dev'}]
    if Counter(r['role'] for r in chosen)!=Counter(fit=288,dev=96) or len({r['family_id'] for r in chosen})!=384:raise ValueError('complete fixed fit/development cohort required')
    if any(r['acquisition_exclusions'] or r['training_eligible'] for r in chosen):raise ValueError('excluded or already admitted source')
    protocol=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        plan_packet_sha256=sha(a.plan/'SHA256SUMS.json'),pilot_sha256=sha(a.plan/'pilot.json'),
        annotation_packet_sha256=sha(a.annotations/'SHA256SUMS.json'),witness_packet_sha256=sha(a.witnesses/'SHA256SUMS.json'),
        preview_parity_sha256=sha(a.parity),pipeline_sha256=sha(a.pipeline),png_reader_sha256=sha(a.png_reader),
        selected_ids=[r['id'] for r in chosen],required=384,roles={'fit':288,'dev':96},
        source_max_pixels=32000000,source_mode='RGB',source_orientation=1,long_edge=768,
        encoded_per_source_limit=8*1024**2,total_payload_limit=LIMIT,host_cap_gib=2,threads=1,
        source_roles_changed=False,training_eligible=False,test_pixels_opened=False,paid_model_api_calls=0)
    a.output.mkdir(parents=True,exist_ok=a.resume)
    if a.resume:
        cached=json.loads((a.output/'protocol.json').read_text())
        if {k:v for k,v in cached.items() if k!='build_commit'}!={k:v for k,v in protocol.items() if k!='build_commit'}:raise ValueError('resume protocol differs')
        protocol=cached
        if (a.output/'summary.json').exists():raise ValueError('acquisition already complete; inspect cached results')
    else:(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    if shutil.disk_usage(a.output).free<4*1024**3:raise ValueError('fourGiB disk headroom required')
    progress=a.output/'progress.json';finished=False;rows=[]
    def final_status():
        if not finished:progress.write_text(json.dumps(dict(pid=None,stage='failed-bytes-retained',completed=len(rows),required=384,checked_utc=datetime.now(timezone.utc).isoformat()))+'\n')
    atexit.register(final_status)
    wanted={r['id'].split(':')[1]:r for r in chosen};boxes={key:[] for key in wanted}
    with (a.annotations/'train-boxes.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['ImageID'] in wanted:
                box=[float(row[k]) for k in ['XMin','YMin','XMax','YMax']]
                if not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):raise ValueError('invalid selected geometry')
                boxes[row['ImageID']].append(dict(box=box,flags={k:int(row[k]) for k in ['IsOccluded','IsTruncated','IsGroupOf','IsDepictionOf','IsInsideOf']}))
    for source in chosen:
        key=source['id'].split(':')[1];directory=a.output/key;directory.mkdir(exist_ok=a.resume)
        progress.write_text(json.dumps(dict(pid=os.getpid(),stage='acquiring-owned-previews',completed=len(rows),required=384,current_id=source['id'],checked_utc=datetime.now(timezone.utc).isoformat()))+'\n')
        record=directory/'record.json';url='https://open-images-dataset.s3.amazonaws.com/train/'+key+'.jpg'
        if a.resume and record.exists():
            row=json.loads(record.read_text())
            if row['source']!=source or row['boxes']!=boxes[key] or sha(directory/'source.jpg')!=row['download']['sha256'] or any(sha(directory/name)!=row[name+'_sha256'] for name in ['preview.png','preview.rgb']):raise ValueError('resumed source/preview/annotation differs')
        elif (a.witnesses/key/'record.json').exists():
            old=a.witnesses/key
            for name in ['record.json','source.jpg','source.jpg.http.json','preview.png','preview.rgb','pipeline.log','reader.log']:member(a.witnesses,key+'/'+name)
            old_row=json.loads((old/'record.json').read_text())
            if old_row['source']['role']!=source['role'] or old_row['source']['family_id']!=source['family_id'] or old_row['boxes']!=boxes[key]:raise ValueError('cached witness role/component/annotations differ')
            for name in ['source.jpg','source.jpg.http.json','preview.png','preview.rgb','pipeline.log','reader.log']:shutil.copy2(old/name,directory/name)
            shutil.copy2(old/'record.json',directory/'original-witness-record.json')
            row=dict(old_row,source=source,cache_origin='frozen-twelve-witness-packet',witness_record_sha256=sha(old/'record.json'))
            record.write_text(json.dumps(row,indent=2)+'\n')
        else:
            download=fetch(directory,'source.jpg',url)
            row=prepare_preview(directory,a.pipeline,a.png_reader,source,download,boxes[key]);row['cache_origin']='new-cvdf-download'
            record.write_text(json.dumps(row,indent=2)+'\n')
        rows.append(row)
        total=sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())
        if total>LIMIT:raise ValueError('twoGiB acquisition allowance exceeded; preserve outputs')
        print(json.dumps(dict(completed=len(rows),required=384,id=source['id'],role=source['role'],origin=row['cache_origin'])),flush=True)
    manifest=dict(protocol=protocol,rows=rows)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    links=[]
    for row in rows:
        raw=row['source']['source_information']['source']
        links.extend([dict(text=row['id']+' '+label,href=value) for label,value in [('CVDF JPEG',row['download']['url']),('original source',raw['OriginalLandingURL']),('author',raw['AuthorProfileURL']),('listed license',raw['License'])]])
    (a.output/'links.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in links))
    summary=dict(build_commit=protocol['build_commit'],sources=len(rows),roles=dict(Counter(r['source']['role'] for r in rows)),
        independent_png_readbacks=sum(r['independent_png_rgb_exact'] for r in rows),
        person_box_records=sum(len(r['boxes']) for r in rows),cache_origins=dict(Counter(r['cache_origin'] for r in rows)),
        encoded_bytes=sum(r['download']['bytes'] for r in rows),payload_bytes=sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file()),
        training_eligible=False,content_review_complete=False,source_relationship_screen_complete=False,
        per_image_live_license_verified=False,source_roles_changed=False,test_pixels_opened=False,paid_model_api_calls=0)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    progress.write_text(json.dumps(dict(pid=None,stage='complete-awaiting-content-and-relationship-screen',completed=384,required=384,checked_utc=datetime.now(timezone.utc).isoformat()))+'\n');finished=True
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
