#!/usr/bin/env python3
"""Separate crowd boxes from individual count/scale without changing source roles."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from artifact_identity import sha


def group_cell(stats):
    individual,groups=stats['individual_records'],stats['group_records']
    if groups and not individual:return 'group-only/unresolved'
    if individual<1:raise ValueError('person source has no annotation records')
    area=stats['maximum_individual_area']
    scale='small' if area<.02 else 'medium' if area<.2 else 'large'
    number=('1' if individual==1 else '2' if individual==2 else '3-5' if individual<=5 else '6+')
    return ('group+individual' if groups else 'individual-'+number)+'/'+scale


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','witnesses','annotations','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    for path,digest in [(a.parent/'SHA256SUMS.json','133daebb418f83587f4f1077c59ba7a54a80e088626ba9c1c896a416b4d880ab'),
        (a.witnesses/'SHA256SUMS.json','8468972acdcd5edc34bf2f7042631d83bc104071cf2bd99e9147dd0f280316ad'),
        (a.annotations/'SHA256SUMS.json','77fd61540c7748b066383a4940c504358b9e48f6bc45982cb2a53536eb573bb6')]:
        if sha(path)!=digest:raise ValueError('frozen parent differs')
    a.output.mkdir(parents=True,exist_ok=False)
    review=json.loads((a.witnesses/'review-and-relationships.json').read_text())
    blocked={r['id']:r['status'] for r in review['observations'] if r['status'] in {'exclude_document','quarantine_depiction_target'}}
    if len(blocked)!=2:raise ValueError('retain both measured scope/target exclusions')
    stats=defaultdict(lambda:dict(individual_records=0,group_records=0,maximum_individual_area=None,maximum_group_area=None,invalid_geometry_records=0))
    rows=0
    with (a.annotations/'train-boxes.csv').open() as stream:
        for row in csv.DictReader(stream):
            rows+=1;item=stats['openimages:'+row['ImageID']]
            left,top,right,bottom=[float(row[k]) for k in ['XMin','YMin','XMax','YMax']]
            valid=0<=left<right<=1 and 0<=top<bottom<=1
            item['invalid_geometry_records']+=not valid
            group=int(row['IsGroupOf'])
            if group not in (0,1):raise ValueError('invalid group flag')
            name='group' if group else 'individual';item[name+'_records']+=1
            key='maximum_'+name+'_area';area=(right-left)*(bottom-top)
            if valid:item[key]=area if item[key] is None else max(item[key],area)
    if len(stats)!=70000 or rows!=373386 or sum(s['invalid_geometry_records'] for s in stats.values())!=5:raise ValueError('annotation inventory differs')
    eligible=defaultdict(list);inherited_roles=Counter();cells=Counter();excluded=Counter();all_rows=[]
    with (a.parent/'registry.jsonl').open() as stream:
        for row in map(json.loads,stream):
            inherited_roles[row['role']]+=1;s=stats[row['id']]
            # Degenerate sources remain quarantined. Do not manufacture a scale.
            cell=None if s['invalid_geometry_records'] else group_cell(s)
            reasons=list(row['acquisition_exclusions'])
            if row['id'] in blocked:reasons.append('visual_review:'+blocked[row['id']])
            if s['invalid_geometry_records'] and 'quarantined_annotation_geometry' not in reasons:raise ValueError('invalid source lost its quarantine')
            all_rows.append(dict(id=row['id'],family_id=row['family_id'],role=row['role'],cell=cell,stats=s,exclusions=reasons,training_eligible=False))
            excluded.update(reasons)
            if not reasons:
                entry=dict(row,group_cell=cell,individual_group_stats=s,visual_content_review_complete=False)
                eligible[(row['role'],cell)].append(entry);cells[row['role']+'/'+cell]+=1
    parent_summary=json.loads((a.parent/'summary.json').read_text())
    if dict(inherited_roles)!=parent_summary['roles']:raise ValueError('prior frozen source roles changed')
    chosen=[];used=set();shortfalls=[];quota={'fit':18,'dev':6,'test':6}
    for (role,cell),pool in sorted(eligible.items()):
        count=0
        for row in sorted(pool,key=lambda r:hashlib.sha256(('zensally-miap-pilot-v1\0'+r['id']).encode()).digest()):
            if row['family_id'] in used:continue
            chosen.append(row);used.add(row['family_id']);count+=1
            if count==quota[role]:break
        if count<quota[role]:shortfalls.append(dict(role=role,cell=cell,available_sources=len(pool),selected=count,quota=quota[role]))
    if len({r['family_id'] for r in chosen})!=len(chosen) or len(chosen)>480:raise ValueError('component or storage sampling contract differs')
    protocol=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        parent_packet_sha256=sha(a.parent/'SHA256SUMS.json'),witness_packet_sha256=sha(a.witnesses/'SHA256SUMS.json'),
        annotation_packet_sha256=sha(a.annotations/'SHA256SUMS.json'),changed_axis='separate group records and individual-area scale; exclude two reviewed bad sources',
        previous_source_roles_changed=False,source_role_seed_rerolled=False,pilot_hash_order_unchanged=True,
        maximum_sources=480,maximum_sources_per_component=1,per_cell_role_quota=quota,
        training_eligible=False,face_coverage_certified=False,human_artistic_labels=False,paid_model_api_calls=0)
    summary=dict(sources=70000,person_box_records=rows,group_records=sum(s['group_records'] for s in stats.values()),
        group_sources=sum(bool(s['group_records']) for s in stats.values()),
        group_only_sources=sum(bool(s['group_records']) and not s['individual_records'] for s in stats.values()),
        inherited_source_roles=dict(inherited_roles),eligible_cells=dict(cells),
        pilot_sources=dict(Counter(r['role'] for r in chosen)),pilot_cells=dict(Counter(r['role']+'/'+r['group_cell'] for r in chosen)),
        shortfalls=shortfalls,acquisition_exclusions=dict(excluded),previous_source_roles_changed=False,
        images_acquired_for_this_plan=0,test_pixels_opened=False,training_eligible=False,paid_model_api_calls=0)
    for name,value in [('protocol.json',protocol),('summary.json',summary),('pilot.json',chosen)]:
        (a.output/name).write_text(json.dumps(value,indent=2)+'\n')
    with (a.output/'cells.jsonl').open('x') as stream:
        for row in all_rows:stream.write(json.dumps(row)+'\n')
    total=sum(p.stat().st_size for p in a.output.iterdir() if p.is_file())
    if total>64*1024**2:raise ValueError('64MiB derived metadata allowance exceeded; preserve outputs')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
