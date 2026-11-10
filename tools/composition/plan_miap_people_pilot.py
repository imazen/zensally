#!/usr/bin/env python3
"""Freeze author/identity components and a bounded human-person annotation pilot."""
import argparse
import base64
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urlsplit,unquote

from artifact_identity import sha

URL_FIELDS={'OriginalURL','OriginalLandingURL','AuthorProfileURL','canonical_author_profile',
            'source_url','original_provider_url','canonical_url','url','raw_url','png_v3_sdr_url'}


def identity_keys(value,field=''):
    """Provider IDs remain case-sensitive; licence/credit text is never an alias."""
    keys=set()
    if isinstance(value,dict):
        for name,item in value.items():keys.update(identity_keys(item,name))
    elif isinstance(value,list):
        for item in value:keys.update(identity_keys(item,field))
    elif isinstance(value,str):
        if field in {'id','source_id'} and re.fullmatch(r'openimages:[0-9a-f]{16}',value):keys.add(value)
        if field in {'id','source_id'} and re.fullmatch(r'unsplash:[A-Za-z0-9_-]{11}',value):keys.add(value)
        if field in {'OriginalMD5','original_md5'}:
            try:
                digest=base64.b64decode(value,validate=True)
                if len(digest)==16:keys.add('original-md5:'+digest.hex())
            except ValueError:pass
        if field in URL_FIELDS:
            url=urlsplit(value);host=(url.hostname or '').lower();path=unquote(url.path)
            if url.scheme in {'http','https'} and host:
                keys.add('source-url:'+host+path.rstrip('/'))
            if host in {'flickr.com','www.flickr.com'}:
                photo=re.fullmatch(r'/photos/([^/]+)/(\d+)/?',path)
                profile=re.fullmatch(r'/(?:people|photos)/([^/]+)/?',path)
                if photo:
                    keys.add('flickr-photo:'+photo[2])
                    if re.fullmatch(r'\d+@N\d+',photo[1]):keys.add('flickr-author:'+photo[1])
                if profile and re.fullmatch(r'\d+@N\d+',profile[1]):keys.add('flickr-author:'+profile[1])
            elif host.endswith('.staticflickr.com') or host in {'staticflickr.com','live.staticflickr.com'}:
                photo=re.match(r'(\d{5,})_[0-9a-f]+(?:_|\.)',path.rsplit('/',1)[-1])
                if photo:keys.add('flickr-photo:'+photo[1])
            elif host in {'unsplash.com','www.unsplash.com'}:
                photo=re.fullmatch(r'/photos/([A-Za-z0-9_-]{11})/?',path)
                if photo:keys.add('unsplash:'+photo[1])
    return keys


def components(records):
    parents={r['id']:r['id'] for r in records};owners={}
    def find(identifier):
        while parents[identifier]!=identifier:
            parents[identifier]=parents[parents[identifier]];identifier=parents[identifier]
        return identifier
    for row in records:
        for key in sorted(identity_keys(row)):
            if key in owners:
                left,right=sorted((find(row['id']),find(owners[key])));parents[right]=left
            else:owners[key]=row['id']
    groups=defaultdict(list)
    for row in records:groups[find(row['id'])].append(row['id'])
    return sorted(sorted(group) for group in groups.values())


def slot_role(anchor):
    slot=int.from_bytes(hashlib.sha256(('zensally-miap-family-v1\0seed31\0'+anchor).encode()).digest()[:8],'big')%10000
    return 'fit' if slot<7000 else 'dev' if slot<8500 else 'test'


def cell(row):
    count=row['box_records'];number='1' if count==1 else '2' if count==2 else '3-5' if count<=5 else '6+'
    area=row['maximum_box_area'];size='small' if area<.02 else 'medium' if area<.2 else 'large'
    return number+'/'+size


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['annotations','information','corpus','photographic','supplementary','canonical','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    for path,digest in [(a.annotations/'SHA256SUMS.json','77fd61540c7748b066383a4940c504358b9e48f6bc45982cb2a53536eb573bb6'),
        (a.information/'SHA256SUMS.json','b3fbdeaa6d6d25dafb86a525949685a0e1ebd03e8f38025714a2e07484fd8921'),
        (a.photographic,'6f8d318cc870248dcc2400a235fcbaa5f7f0edf47d575d965bf5c72b5684d2f6'),
        (a.supplementary,'a52144833a5824386577f63626e599f8dc602f8b77ee9d4d7f0c183574cebf70')]:
        if sha(path)!=digest:raise ValueError('frozen input identity differs: '+str(path))
    a.output.mkdir(parents=True,exist_ok=False);references=a.output/'references';references.mkdir()
    protocol=json.loads((a.corpus/'protocol.json').read_text());reference_rows=[];bindings={}
    def capture(path,name):
        shutil.copy2(path,references/name);bindings[name]=sha(references/name)
    for link in protocol['links']:
        path=a.corpus/link['text']
        if sha(path)!=link['sha256']:raise ValueError('canonical manifest differs')
        capture(path,path.name)
        if path.suffix=='.tsv':
            with path.open() as stream:
                for index,row in enumerate(csv.DictReader(stream,delimiter='\t')):
                    if row.get('provenance_json'):row['provenance']=json.loads(row['provenance_json'])
                    reference_rows.append(dict(reference_id=path.name+':'+str(index),role=row.get('bucket',row.get('split','canonical_metadata')),record=row))
    for path,name in [(a.corpus/'sources.json','current-jxl-selection.json'),(a.photographic,'photographic-registry.json'),(a.supplementary,'supplementary-registry.json')]:
        capture(path,name);value=json.loads(path.read_text());rows=value if isinstance(value,list) else value['sources']
        for index,row in enumerate(rows):reference_rows.append(dict(reference_id=name+':'+str(index),role=row.get('role',row.get('use','unspecified')),record=row))
    # The older pinned original inventory adds original filenames/provenance,
    # without reading any source pixel or synchronizing the sibling checkout.
    canonical_commit='187fbf338ce08e8e6654db7f04ddae58d5263da2'
    raw=subprocess.check_output(['git','-C',str(a.canonical),'show',canonical_commit+':CORPUS-MANIFEST.tsv'])
    original=references/'original-corpus-manifest.tsv';original.write_bytes(raw);bindings[original.name]=sha(original)
    with original.open() as stream:
        for index,row in enumerate(csv.DictReader(stream,delimiter='\t')):reference_rows.append(dict(reference_id='original-manifest:'+str(index),role='canonical_metadata',record=row))
    reference_keys=defaultdict(list);reference_coverage=Counter()
    for row in reference_rows:
        keys=identity_keys(row['record'])
        if keys:reference_coverage[row['reference_id'].split(':')[0]]+=1
        for key in keys:reference_keys[key].append(dict(reference_id=row['reference_id'],role=row['role']))
    with (a.information/'miap-training-image-information.jsonl').open() as stream:records=[json.loads(line) for line in stream]
    with (a.annotations/'inventory.jsonl').open() as stream:inventory={r['id']:r for r in map(json.loads,stream)}
    if len(records)!=70000 or set(inventory)!={r['id'] for r in records}:raise ValueError('complete TRAIN join required')
    groups=components(records);by_id={r['id']:r for r in records};assignments={};overlaps=[]
    for group in groups:
        keys=set().union(*(identity_keys(by_id[identifier]) for identifier in group))
        matches=[dict(key=key,references=reference_keys[key]) for key in sorted(keys & reference_keys.keys())]
        family='miap-lineage:'+hashlib.sha256('\n'.join(group).encode()).hexdigest()
        role='quarantine' if matches else slot_role(group[0])
        if matches:overlaps.append(dict(family_id=family,ids=group,matches=matches))
        for identifier in group:assignments[identifier]=dict(family_id=family,role=role,anchor_id=group[0])
    registry=[];eligible=defaultdict(list);exclusions=Counter()
    for row in records:
        annotation=inventory[row['id']];binding=assignments[row['id']];reasons=list(row['source_metadata_issues'])
        if binding['role']=='quarantine':reasons.append('prior_reference_component_overlap')
        if row['rotation_degrees'] not in (0,None):reasons.append('nonzero_rotation_deferred_pending_pixel_convention')
        if annotation['IsDepictionOf']:reasons.append('depiction_flag_requires_review')
        entry=dict(id=row['id'],**binding,cell=cell(annotation),box_inventory=annotation,
            acquisition_exclusions=reasons,training_eligible=False,face_coverage_certified=False,
            source_information=row)
        registry.append(entry);exclusions.update(reasons)
        if not reasons:eligible[(binding['role'],entry['cell'])].append(entry)
    chosen=[];used=set();quotas={'fit':24,'dev':8,'test':8};availability={}
    for (role,cell_name),pool in sorted(eligible.items()):
        availability[role+'/'+cell_name]=len(pool)
        order=sorted(pool,key=lambda r:hashlib.sha256(('zensally-miap-pilot-v1\0'+r['id']).encode()).digest())
        count=0
        for row in order:
            if row['family_id'] in used:continue
            chosen.append(row);used.add(row['family_id']);count+=1
            if count==quotas[role]:break
    if len({r['family_id'] for r in chosen})!=len(chosen):raise ValueError('pilot repeats a photographer/identity component')
    with (a.output/'registry.jsonl').open('x') as stream:
        for row in registry:stream.write(json.dumps(row)+'\n')
    for name,value in [('pilot.json',chosen),('overlaps.json',overlaps)]:
        (a.output/name).write_text(json.dumps(value,indent=2)+'\n')
    links=protocol['links']+[dict(text='Original corpus manifest',href='https://github.com/imazen/imazen-26/blob/'+canonical_commit+'/CORPUS-MANIFEST.tsv')]
    (a.output/'links.jsonl').write_text(''.join(json.dumps(link)+'\n' for link in links))
    summary=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        sources=70000,components=len(groups),largest_component=max(map(len,groups)),
        roles=dict(Counter(r['role'] for r in registry)),component_roles=dict(Counter(assignments[g[0]]['role'] for g in groups)),
        overlapping_components=len(overlaps),reference_records=len(reference_rows),reference_alias_keys=len(reference_keys),
        reference_rows_with_available_keys=dict(reference_coverage),reference_metadata_bindings=bindings,
        eligible_by_role_and_cell=availability,acquisition_exclusion_counts=dict(exclusions),
        pilot_sources=dict(Counter(r['role'] for r in chosen)),pilot_cells=dict(Counter(r['role']+'/'+r['cell'] for r in chosen)),
        pilot_max_per_component=1,pilot_maximum_sources=480,cell_count_is_box_records_not_individual_people=True,
        scale_proxy='maximum person-box area; small<0.02, medium<0.2, large>=0.2',
        original_canonical_roles_changed=False,prior_internal_roles_changed=False,new_component_roles_frozen=True,
        images_acquired=0,training_eligible=False,complete_duplicate_recall_certified=False,
        original_md5_cannot_be_compared_with_reference_sha256=True,pixel_relationship_screen_pending=True,
        live_per_image_license_check_pending=True,annotation_pixel_orientation_verification_pending=True,
        human_artistic_labels=False,face_coverage_certified=False,demographic_attributes_used=False,paid_model_api_calls=0)
    if sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())>256*1024**2:raise ValueError('256MiB planning allowance exceeded; preserve outputs')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
