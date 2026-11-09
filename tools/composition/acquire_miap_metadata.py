#!/usr/bin/env python3
"""Cache official TRAIN-only people metadata without images or demographic use."""
import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime,timezone
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urljoin
from urllib.request import Request,urlopen

from artifact_identity import sha

BASE='https://storage.googleapis.com/openimages/'
BUDGET=128*1024**2
SOURCES=[
    ('extended.html',BASE+'web/extended.html'),
    ('download.html',BASE+'web/download_v7.html'),
    ('description.html',BASE+'web/factsfigures_v7.html'),
    ('data-card.pdf',BASE+'open_images_extended_miap/Open%20Images%20Extended%20-%20MIAP%20-%20Data%20Card.pdf'),
    ('train-boxes.csv',BASE+'open_images_extended_miap/open_images_extended_miap_boxes_train.csv'),
    ('train-keys.lst',BASE+'open_images_extended_miap/open_images_extended_miap_images_train.lst'),
]


class Links(HTMLParser):
    def __init__(self,base):super().__init__();self.base=base;self.links=[];self.active=None
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            href=dict(attrs).get('href')
            if href is not None:self.active=[urljoin(self.base,href),[]]
    def handle_data(self,text):
        if self.active is not None:self.active[1].append(text)
    def handle_endtag(self,tag):
        if tag=='a' and self.active is not None:
            self.links.append(dict(text=' '.join(''.join(self.active[1]).split()),href=self.active[0]));self.active=None


def retain(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')


def fetch(root,name,url,progress):
    path=root/name;record=path.with_suffix(path.suffix+'.http.json')
    if path.exists():
        saved=json.loads(record.read_text())
        if saved['url']!=url or sha(path)!=saved['sha256']:raise ValueError('cached source metadata differs')
        return saved
    with urlopen(Request(url,headers={'User-Agent':'zensally-source-research','Accept-Encoding':'identity'}),timeout=60) as response:
        declared=response.headers.get('Content-Length')
        if declared is not None and int(declared)>BUDGET:raise ValueError('single metadata source exceeds declared budget')
        headers=dict(response.headers.items());actual=0
        with path.open('xb') as stream:
            while block:=response.read(1024*1024):
                actual+=len(block)
                if actual+sum(p.stat().st_size for p in root.iterdir() if p.is_file())>BUDGET:raise ValueError('128MiB metadata storage allowance exceeded; preserve partial bytes')
                stream.write(block)
                progress.write_text(json.dumps(dict(pid=os.getpid(),stage='downloading-metadata',file=name,bytes=actual,checked_utc=datetime.now(timezone.utc).isoformat()))+'\n')
        if declared is not None and actual!=int(declared):raise ValueError('metadata response length differs')
        value=dict(url=url,resolved_url=response.url,status=response.status,headers=headers,bytes=actual,sha256=sha(path),captured_utc=datetime.now(timezone.utc).isoformat())
    retain(record,value);return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    if shutil.disk_usage(a.root.parent).free<2*1024**3:raise ValueError('metadata acquisition needs two GiB disk headroom')
    a.root.mkdir(parents=True,exist_ok=True)
    if (a.root/'summary.json').exists():raise ValueError('metadata stage already complete; inspect cached inventory')
    progress=a.root/'progress.json';records=[];links=[]
    for name,url in SOURCES:
        records.append(fetch(a.root,name,url,progress))
        if name.endswith('.html'):
            parser=Links(url);parser.feed((a.root/name).read_text());links.extend(dict(source=url,**link) for link in parser.links)
    class_urls={r['href'] for r in links if r['text']=='Boxable class names'}
    if len(class_urls)!=1:raise ValueError('one official boxable-class metadata link required')
    records.append(fetch(a.root,'boxable-classes.csv',class_urls.pop(),progress))
    with (a.root/'links.jsonl').open('x') as stream:
        for row in links:stream.write(json.dumps(row)+'\n')
    image_information=BASE+'2018_04/train/train-images-boxable-with-rotation.csv'
    with urlopen(Request(image_information,method='HEAD',headers={'User-Agent':'zensally-source-research'}),timeout=30) as response:
        retain(a.root/'image-information-head.json',dict(url=image_information,resolved_url=response.url,status=response.status,headers=dict(response.headers.items()),downloaded_body=False))
    keys=(a.root/'train-keys.lst').read_text().splitlines()
    if len(keys)!=70000 or len(set(keys))!=70000 or any(not re.fullmatch(r'train/[a-f0-9]{16}',k) for k in keys):raise ValueError('official seventy-thousand TRAIN image-key contract differs')
    expected={key.split('/')[1] for key in keys}
    counts=defaultdict(Counter);areas=defaultdict(list);flags=Counter();records_count=0
    with (a.root/'train-boxes.csv').open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        needed=['ImageID','LabelName','Confidence','XMin','XMax','YMin','YMax','IsGroupOf','IsOccluded','IsTruncated']
        if not set(needed)<=set(reader.fieldnames):raise ValueError('expected source bounding-box schema missing')
        for row in reader:
            identifier=row['ImageID'];box=[float(row[k]) for k in ['XMin','YMin','XMax','YMax']]
            if identifier not in expected or not all(math.isfinite(v) for v in box) or not(0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1) or row['LabelName']!='/m/01g317' or float(row['Confidence'])!=1:raise ValueError('invalid TRAIN person annotation; preserve source bytes')
            counts[identifier]['box_records']+=1;areas[identifier].append((box[2]-box[0])*(box[3]-box[1]));records_count+=1
            for flag in ['IsGroupOf','IsOccluded','IsTruncated']:
                if row[flag] not in ['0','1','-1']:raise ValueError('unknown source quality flag')
                counts[identifier][flag]+=int(row[flag]=='1');flags[flag+':'+row[flag]]+=1
    if set(counts)!=expected:raise ValueError('every official TRAIN image needs a person annotation')
    inventory=a.root/'inventory.jsonl'
    with inventory.open('x') as stream:
        for identifier in sorted(counts):
            stream.write(json.dumps(dict(id='openimages:'+identifier,official_split='train',role=None,training_eligible=False,
                **dict(counts[identifier]),minimum_box_area=min(areas[identifier]),maximum_box_area=max(areas[identifier])))+'\n')
    with (a.root/'boxable-classes.csv').open(newline='') as stream:
        classes={name:mid for mid,name in csv.reader(stream) if name in ['Person','Human face','Dog','Cat','Bird','Tree']}
    if classes.get('Person')!='/m/01g317':raise ValueError('official person class identity differs')
    summary=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        official_training_images=70000,person_box_records=records_count,group_records=flags['IsGroupOf:1'],flags=dict(flags),
        boxes_per_image=dict(Counter(c['box_records'] for c in counts.values())),classes=classes,source_records=records,
        bytes_stored=sum(p.stat().st_size for p in a.root.iterdir() if p.is_file()),
        source_roles_assigned=False,images_acquired=0,training_eligible=False,demographic_attributes_used=False,
        image_author_license_rotation_join_complete=False,canonical_holdouts_opened=False,
        face_coverage_certified=False,human_artistic_labels=False,paid_model_api_calls=0)
    if summary['bytes_stored']>BUDGET:raise ValueError('128MiB metadata allowance exceeded; preserve output')
    retain(a.root/'summary.json',summary)
    progress.write_text(json.dumps(dict(pid=None,stage='complete',training_images=70000,person_box_records=records_count,checked_utc=datetime.now(timezone.utc).isoformat()))+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['source_records','boxes_per_image']},indent=2))


if __name__=='__main__':main()
