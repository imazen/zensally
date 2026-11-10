#!/usr/bin/env python3
"""Acquire twelve fixed fitting witnesses, using owned previews and PNG readback."""
import argparse
import atexit
import csv
from datetime import datetime,timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.request import Request,urlopen

from PIL import Image
from artifact_identity import sha

LIMIT=128*1024**2


def fetch(root,name,url):
    path=root/name;record=path.with_suffix(path.suffix+'.http.json')
    if path.exists():
        cached=json.loads(record.read_text())
        if cached['url']!=url or cached['sha256']!=sha(path):raise ValueError('cached download identity differs')
        return cached
    with urlopen(Request(url,headers={'User-Agent':'zensally-people-pilot','Accept-Encoding':'identity'}),timeout=45) as response:
        headers=dict(response.headers.items());declared=response.headers.get('Content-Length');count=0
        if declared and int(declared)>8*1024**2:raise ValueError('source exceeds8MiB encoded cap')
        with path.open('xb') as stream:
            while block:=response.read(1024*1024):
                count+=len(block)
                if count>8*1024**2:raise ValueError('source exceeds8MiB encoded cap; preserve partial')
                stream.write(block)
        if declared and count!=int(declared):raise ValueError('declared source length differs')
        etag=headers.get('ETag','');provider_md5_exact=False
        if re.fullmatch(r'"[0-9a-f]{32}"',etag):
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,lambda:hashlib.md5(usedforsecurity=False)).hexdigest()
            if digest!=etag.strip('"'):raise ValueError('source/provider MD5 differs')
            provider_md5_exact=True
        result=dict(url=url,resolved_url=response.url,headers=headers,bytes=count,sha256=sha(path),
            provider_md5_exact=provider_md5_exact,captured_utc=datetime.now(timezone.utc).isoformat())
    record.write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','annotations','pipeline','png_reader','output']:p.add_argument(name,type=Path)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if sha(a.plan/'SHA256SUMS.json')!='133daebb418f83587f4f1077c59ba7a54a80e088626ba9c1c896a416b4d880ab':raise ValueError('frozen pilot plan differs')
    if sha(a.pipeline)!='6797c4f8e2e627921bd1fe61122e229325b8757538b9e554bcc28a9b39a9ab9a' or sha(a.png_reader)!='3346ccde58afc455194c22397ff3a5b025e4ceede6115f6795a0f8ad0c694c16':raise ValueError('qualified owned preview binaries differ')
    pilot=json.loads((a.plan/'pilot.json').read_text());selected={}
    for row in pilot:
        if row['role']=='fit':selected.setdefault(row['cell'],row)
    chosen=[selected[cell] for cell in sorted(selected)]
    if len(chosen)!=12 or any(row['acquisition_exclusions'] or row['training_eligible'] for row in chosen):raise ValueError('twelve unadmitted fitting-only cells required')
    protocol=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        pilot_sha256=sha(a.plan/'pilot.json'),pipeline_sha256=sha(a.pipeline),png_reader_sha256=sha(a.png_reader),
        selected_ids=[r['id'] for r in chosen],selection='first frozen fit source in each of twelve cells',
        source_max_pixels=32000000,encoded_per_source_limit=8*1024**2,total_payload_limit=LIMIT,
        long_edge=768,training_eligible=False,test_pixels_opened=False,paid_model_api_calls=0)
    a.output.mkdir(parents=True,exist_ok=a.resume)
    if a.resume:
        cached=json.loads((a.output/'protocol.json').read_text())
        if {k:v for k,v in cached.items() if k!='build_commit'}!={k:v for k,v in protocol.items() if k!='build_commit'}:raise ValueError('resume protocol differs')
        protocol=cached
        if (a.output/'summary.json').exists():raise ValueError('completed witness stage already closed')
    else:(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    finished=False;progress=a.output/'progress.json'
    def final_status():
        if not finished:progress.write_text(json.dumps(dict(pid=None,stage='failed-bytes-retained',checked_utc=datetime.now(timezone.utc).isoformat()))+'\n')
    atexit.register(final_status)
    if shutil.disk_usage(a.output).free<4*1024**3:raise ValueError('fourGiB disk headroom required')
    wanted={r['id'].split(':')[1]:r for r in chosen};boxes={identifier:[] for identifier in wanted}
    with (a.annotations/'train-boxes.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['ImageID'] in wanted:
                box=[float(row[k]) for k in ['XMin','YMin','XMax','YMax']]
                if not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):raise ValueError('invalid geometry in selected source')
                boxes[row['ImageID']].append(dict(box=box,flags={k:int(row[k]) for k in ['IsOccluded','IsTruncated','IsGroupOf','IsDepictionOf','IsInsideOf']}))
    # Save official download guidance as provenance; do not execute its decoder.
    fetch(a.output,'official-downloader.py','https://raw.githubusercontent.com/openimages/dataset/master/downloader.py')
    rows=[];links=[];cards=[]
    for source in chosen:
        identifier=source['id'].split(':')[1];directory=a.output/identifier;directory.mkdir(exist_ok=a.resume)
        progress.write_text(json.dumps(dict(pid=os.getpid(),stage='acquire-owned-preview',completed=len(rows),required=12,current_id=source['id'],checked_utc=datetime.now(timezone.utc).isoformat()))+'\n')
        original=directory/'source.jpg';url='https://open-images-dataset.s3.amazonaws.com/train/'+identifier+'.jpg'
        download=fetch(directory,'source.jpg',url);record=directory/'record.json'
        if a.resume and record.exists():
            row=json.loads(record.read_text())
            if row['source']!=source or row['download']!=download or any(sha(directory/name)!=row[name+'_sha256'] for name in ['preview.png','preview.rgb']):raise ValueError('cached witness differs')
        else:
            with Image.open(original) as image:
                header=dict(size=list(image.size),orientation=image.getexif().get(274,1),mode=image.mode)
                if image.format!='JPEG' or image.mode!='RGB' or header['orientation']!=1 or image.width*image.height>32000000:raise ValueError('witness source header outside registered orientation/format/size scope')
            result=subprocess.run([str(a.pipeline),str(original),str(directory/'preview.png'),'768'],capture_output=True,text=True)
            (directory/'pipeline.log').write_text(result.stdout+result.stderr);result.check_returncode();geometry=json.loads(result.stdout)
            result=subprocess.run([str(a.png_reader),str(directory/'preview.png'),str(directory/'preview.rgb')],capture_output=True,text=True)
            (directory/'reader.log').write_text(result.stdout+result.stderr);result.check_returncode()
            with Image.open(directory/'preview.png') as image:
                if list(image.size)!=[geometry['width'],geometry['height']] or image.convert('RGB').tobytes()!=(directory/'preview.rgb').read_bytes():raise ValueError('independent PNG samples/dimensions differ')
            if header['size']!=[geometry['source_width'],geometry['source_height']]:raise ValueError('independent JPEG and owned source dimensions differ')
            if len(boxes[identifier])!=source['box_inventory']['box_records']:raise ValueError('source-bound annotation count differs')
            row=dict(id=source['id'],source=source,download=download,header=header,geometry=geometry,boxes=boxes[identifier],
                **{name+'_sha256':sha(directory/name) for name in ['preview.png','preview.rgb']},
                independent_png_rgb_exact=True,training_eligible=False,face_coverage_certified=False,
                original_md5_not_applicable_to_cvdf_derivative=True,annotation_alignment_review_complete=False)
            record.write_text(json.dumps(row,indent=2)+'\n')
        rows.append(row);w,h=row['geometry']['width'],row['geometry']['height']
        rects=''.join('<rect x="{}" y="{}" width="{}" height="{}"/>'.format(b['box'][0]*w,b['box'][1]*h,(b['box'][2]-b['box'][0])*w,(b['box'][3]-b['box'][1])*h) for b in row['boxes'])
        cards.append('<article><h2>{} · {} · {} records</h2><svg viewBox="0 0 {} {}"><image href="{}/preview.png" width="{}" height="{}"/>{}</svg><a href="{}">Original source</a></article>'.format(identifier,source['cell'],len(row['boxes']),w,h,identifier,w,h,rects,html.escape(source['source_information']['source']['OriginalLandingURL'],quote=True)))
        links.extend([dict(text=source['id']+' CVDF JPEG',href=url),dict(text=source['id']+' original source',href=source['source_information']['source']['OriginalLandingURL']),dict(text=source['id']+' attribution',href=source['source_information']['source']['AuthorProfileURL'])])
        if sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())>LIMIT:raise ValueError('128MiB witness allowance exceeded; preserve bytes')
        print(json.dumps(dict(completed=len(rows),required=12,id=source['id'])),flush=True)
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MIAP fitting witnesses</title><style>body{font:14px sans-serif;background:#222;color:#eee}main{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}h2{font-size:13px}svg{width:100%;height:275px;background:#111}rect{fill:none;stroke:#ffcc00;stroke-width:2}a{color:#aef}article{overflow:hidden}</style><h1>Twelve fixed fitting-only witnesses</h1><p>Human person-box records; yellow overlays. No face/artistic labels or training admission. Box counts include group records.</p><main>'+''.join(cards)+'</main>')
    (a.output/'links.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in links))
    summary=dict(build_commit=protocol['build_commit'],sources=len(rows),roles={'fit':12},native_sizes=[r['header']['size'] for r in rows],
        person_box_records=sum(len(r['boxes']) for r in rows),independent_png_readbacks=len(rows),
        encoded_bytes=sum(r['download']['bytes'] for r in rows),payload_bytes=sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file()),
        annotation_alignment_review_complete=False,source_relationship_screen_complete=False,
        per_image_live_license_verified=False,training_eligible=False,test_pixels_opened=False,paid_model_api_calls=0)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    progress.write_text(json.dumps(dict(pid=None,stage='complete-awaiting-visual-and-relationship-review',completed=12,required=12,checked_utc=datetime.now(timezone.utc).isoformat()))+'\n');finished=True
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
