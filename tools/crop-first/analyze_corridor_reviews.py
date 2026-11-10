#!/usr/bin/env python3
"""Join blinded pixel judgments, repeat controls and exact teacher interval identities."""
import argparse
from collections import Counter,defaultdict
from decimal import Decimal
import json
from pathlib import Path
import re
from api_budget import cost
from direct_zones import canonical,sha
from label_quality import build
from sunburst_generate import journal,exposure


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    tasks={}
    for path in a.root.rglob('tasks/*.json'):
        task=json.loads(path.read_text())
        if 'candidates' in task:
            if task['id'] in tasks and tasks[task['id']]!=task:raise ValueError('ambiguous blinded task identity')
            tasks[task['id']]=task
    group_info={}
    for path in a.root.rglob('plan.json'):
        for group in json.loads(path.read_text()).get('groups',[]):
            if group['case_id'] in group_info:raise ValueError('ambiguous interval group identity')
            group_info[group['case_id']]=group
    states=journal(a.root/'testing/budget.jsonl');proposals=[];reviews=[];invalid=[];ids=set();native_total=Decimal(0);control_failures=[];duplicate_conflicts=[]
    for entry in sorted((a.root/'vision-cache').iterdir()):
        if not entry.is_dir() or not re.fullmatch('[0-9a-f]{64}',entry.name):continue
        for name,digest in json.loads((entry/'hashes.json').read_text()).items():
            if sha((entry/name).read_bytes())!=digest:raise ValueError('response cache changed')
        metadata=json.loads((entry/'metadata.json').read_text());response=json.loads((entry/'response.json').read_text());request=json.loads((entry/'request.json').read_text())
        if response['id'] in ids:raise ValueError('duplicate native response ID')
        ids.add(response['id']);native_total+=cost(response['usage'],request['model'])
        if not metadata['annotation_valid']:
            invalid.append(dict(key=entry.name,source_id=metadata['source_id'],error=metadata.get('validation_error')));continue
        zones=json.loads((entry/'zones.json').read_text())
        if 'projected_ranges' in zones:
            version=2 if request['text']['format']['name'].endswith('_v2') else 1
            proposals.append(dict(source_id=zones['source_id'],version=version,face_inventory=zones['face_inventory'],faces=len(zones['faces']),ranges=zones['projected_ranges'],response_id=response['id']))
            continue
        task=tasks[metadata['source_id']]
        if sha(canonical(dict(cache_protocol=1,request=build(task))))!=entry.name:raise ValueError('blinded pixel task/request identity differs')
        shown={r['id']:r for r in task['candidates']};grades={shown[g['candidate_id']]['origin']:dict(g,pixel_sha256=shown[g['candidate_id']]['rgb_sha256'],png_sha256=shown[g['candidate_id']]['sha256']) for g in zones['grades']}
        review=dict(group=task['case_id'],presentation=task['presentation_id'],model=request['model'],role=task['role'],family_id=task['family_id'],
            response_id=response['id'],face_inventory=zones['face_inventory'],source_faces=len(zones['faces']),grades=grades,semantic_inconsistencies=zones['semantic_inconsistencies'])
        reviews.append(review)
        if 'contain' in grades and grades['contain']['new_face_truncation']!='no':control_failures.append(dict(group=task['case_id'],presentation=task['presentation_id'],label=grades['contain']['new_face_truncation']))
        pixels=defaultdict(list)
        for origin,g in grades.items():pixels[g['pixel_sha256']].append((origin,g))
        for digest,rows in pixels.items():
            for field in ['new_face_truncation','subject_coherence','acceptable','composition_quality']:
                if len({g[field] for _,g in rows})>1:duplicate_conflicts.append(dict(group=task['case_id'],presentation=task['presentation_id'],pixel_sha256=digest,origins=[x for x,_ in rows],field=field))
    grouped=defaultdict(dict)
    for row in reviews:
        if row['presentation'] in grouped[row['group']]:raise ValueError('duplicate paid presentation')
        grouped[row['group']][row['presentation']]=row
    comparisons=[];intervals=[]
    for group,presentations in grouped.items():
        for name,left,right in [('cross-model','terra-first','sol-review'),('sol-order-repeat','sol-review','sol-repeat')]:
            if left not in presentations or right not in presentations:continue
            l,r=presentations[left]['grades'],presentations[right]['grades']
            if set(l)!=set(r) or any(l[k]['pixel_sha256']!=r[k]['pixel_sha256'] for k in l):raise ValueError('paired review pixels differ')
            comparisons.append(dict(group=group,comparison=name,candidates=len(l),**{field:sum(l[k][field]==r[k][field] for k in l) for field in ['acceptable','new_face_truncation','subject_coherence','composition_quality']}))
        info=group_info[group];required=['terra-first','sol-review','sol-repeat'];complete=all(k in presentations for k in required)
        selected=[presentations[k] for k in required if k in presentations];range_names=['inner','quarter25','midpoint','quarter75','outer']
        samples_accepted=bool(selected) and all(all(row['grades'][name]['acceptable']=='yes' and row['grades'][name]['new_face_truncation']=='no' and row['grades'][name]['subject_coherence']=='yes' for name in range_names) and not row['semantic_inconsistencies'] for row in selected)
        endpoints_excellent=bool(selected) and all(all(row['grades'][name]['composition_quality']=='excellent' for name in ['inner','outer']) for row in selected)
        certainty=bool(selected) and all(row['face_inventory']!='uncertain' for row in selected)
        intervals.append(dict(group=group,role=info['role'],family_id=info['family_id'],ratio=info['ratio'],pair=info['pair'],
            complete_three_presentations=complete,available_presentations=list(presentations),all_reviewed_samples_accepted=samples_accepted,
            all_reviewed_endpoints_excellent=endpoints_excellent,review_inventory_certain=certainty,
            eligible_for_weak_interval_admission=complete and samples_accepted and endpoints_excellent and certainty and info['pair']['ready_for_rendered_review'] and info['pair'].get('pair_protocol',1)==2,
            training_eligible=False,face_coverage_certified=False,off_path_origin_extremes_reviewed=False))
    settled=sum((Decimal(v['usd']) for v in states.values() if v['state']=='settled'),Decimal(0));v2=[r for r in proposals if r['version']==2]
    summary=dict(native_responses=len(ids),settled_ledger_usd=str(settled),native_usage_priced_usd=str(native_total),ledger_native_difference_usd=str(settled-native_total),
        pending_reservations=sum(v['state']=='reserved' for v in states.values()),budget_exposure_usd=str(exposure(states)),pilot_ceiling_usd='2',
        proposal_sources_v2=len(v2),ratios_requested_per_source=4,proposed_pairs_v2=sum(len(r['ranges']) for r in v2),
        ready_for_rendered_review_v2=sum(p['ready_for_rendered_review'] for r in v2 for p in r['ranges']),
        smaller_than_minimal_outer_v2=sum(not p['outer_is_maximum_area'] for r in v2 for p in r['ranges']),
        crossing_face_intervals_v2=sum(bool(p['geometry']['faces_crossing_crop_interval']) for r in v2 for p in r['ranges']),
        legacy_v1_proposals=len(proposals)-len(v2),review_presentations=len(reviews),reviewed_groups=len(grouped),
        grade_counts=dict(Counter(g['composition_quality'] for r in reviews for g in r['grades'].values())),
        contain_control_failures=control_failures,duplicate_pixel_label_conflicts=duplicate_conflicts,invalid_annotations=invalid,
        comparisons=comparisons,intervals=intervals,proposals=proposals,
        test_inference=False,model_promoted=False,human_artistic_gold=False,training_started=False,
        limitation='Small reused FIT/DEV pilot, unreviewed face inventories; sampled-path acceptance cannot establish every off-path artistic composition.')
    a.output.mkdir(exist_ok=False);(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.output/'reviews.json').write_text(json.dumps(reviews,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['intervals','proposals','comparisons','duplicate_pixel_label_conflicts','contain_control_failures']}))


if __name__=='__main__':main()
