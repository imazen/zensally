#!/usr/bin/env python3
"""Audit weak subject references and exact-bank retention before label admission."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil

import torch

from artifact_identity import sha
from audit_crop_feasibility import interval
from region_retention import retained_fraction


def retain(path,value):
    with path.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root','cache','previews','old_predictions']:
        p.add_argument(name,type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    if sha(args.cache)!='2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031':
        raise ValueError('frozen photographic fit/development rank cache differs')
    plan_path = args.root/'source-plan.json'
    if sha(plan_path)!='f597532945f20be5fb48770467affceabdee454b0051cd5698c6f13efe877d18':
        raise ValueError('registered photographic subject queries differ')
    plan = json.loads(plan_path.read_text())
    cache = torch.load(args.cache,map_location='cpu',weights_only=False)
    if len(cache['sources'])!=119 or any(s['role']=='test' for s in cache['sources']):
        raise ValueError('fit/development-only source cache required')
    summary = json.loads((args.root/'references/photo87-summary.json').read_text())
    calls = {r['id']:r for r in summary['calls']}
    if summary['completed']!=87 or len(calls)!=87:
        raise ValueError('all registered cached generations required')
    lookup = {s['id']:i for i,s in enumerate(cache['sources'])}
    preview_data = json.loads((args.previews/'sources.json').read_text())
    previews = {s['id']:s for s in preview_data['sources']}
    known = torch.zeros(119,dtype=torch.bool)
    fraction = torch.zeros(119,11,128)
    contained = torch.zeros(119,11,128,dtype=torch.bool)
    bank_valid = torch.zeros_like(contained)
    receipts,requests,review_rows = [],[],[]
    counts = defaultdict(Counter)
    review = args.root/'review'
    review.mkdir()
    (review/'previews').mkdir()
    for index,source in enumerate(plan['sources']):
        i = lookup[source['id']]
        original = cache['sources'][i]
        if any(source[k]!=original[k] for k in ['id','family_id','role','rgb_sha256','native_size']):
            raise ValueError('subject/rank-cache source identity differs')
        call = calls[source['id']]
        path = args.root/'references/responses'/call['response']
        response = json.loads(path.read_text())
        contract = response['contract']
        key = source['source']['provider_id']
        digest = hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (digest!=path.stem or digest!=response['contract_sha256'] or contract['source_id']!=source['id']
                or contract['source_role']!=source['role'] or contract['source_plan_sha256']!=sha(plan_path)
                or contract['native_size']!=source['native_size']
                or contract['prompt']!='<CAPTION_TO_PHRASE_GROUNDING>'+source['query']
                or sha(args.root/'references/inputs'/(key+'.rgb'))!=contract['owned_input_sha256']):
            raise ValueError('cached subject/source contract differs')
        width,height = source['native_size']
        reason,boxes = ('incomplete_generation' if call['invalid'] else None),[]
        if reason is None:
            parsed = response['parsed']['<CAPTION_TO_PHRASE_GROUNDING>']
            if len(parsed['bboxes'])!=len(parsed['labels']):
                raise ValueError('reference labels/boxes differ')
            for box in parsed['bboxes']:
                if not (len(box)==4 and all(math.isfinite(v) for v in box)
                        and 0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):
                    reason = 'invalid_native_geometry'
                else:
                    boxes.append([math.floor(box[0]),math.floor(box[1]),math.ceil(box[2]),math.ceil(box[3])])
            if not boxes:
                reason = 'no_entities_unknown'
        known[i] = reason is None
        receipts.append(dict(id=source['id'],role=source['role'],category=source['query_category'],
            response_sha256=sha(path),boxes=boxes,unknown_reason=reason,training_eligible=False))
        for group in ['all',source['role'],source['query_category']]:
            counts[group]['sources']+=1
            counts[group]['known_sources' if reason is None else 'unknown_sources']+=1
            counts[group]['valid_boxes']+=len(boxes)
        previous = previews[source['id']]
        image = args.previews/previous['preview']
        if sha(image)!=previous['preview_sha256']:
            raise ValueError('owned source preview differs')
        shutil.copy2(image,review/'previews'/(key+'.png'))
        review_rows.append(dict(previous,index=index,sheet=index//10,preview='previews/'+key+'.png',
            query=source['query'],regions=boxes,reference_unknown=reason))
        if reason is not None:
            continue
        for ri,ratio in enumerate(cache['ratios']):
            native = cache['native_boxes'][i][ri]
            p,q = map(int,ratio.split(':'))
            divisor = math.gcd(p,q)
            p,q = p//divisor,q//divisor
            maximum = min(width//p,height//q)
            for j,crop in enumerate(native):
                if not (0<=crop[0]<crop[2]<=width and 0<=crop[1]<crop[3]<=height and (crop[2]-crop[0])*q==(crop[3]-crop[1])*p):
                    raise ValueError('old exact-ratio bank geometry differs')
                bank_valid[i,ri,j] = True
                contained[i,ri,j] = all(crop[0]<=b[0] and crop[1]<=b[1] and crop[2]>=b[2] and crop[3]>=b[3] for b in boxes)
                fraction[i,ri,j] = min(retained_fraction(crop,b) for b in boxes)
            largest = interval(width,height,(p,q),boxes,maximum)
            aggressive = next((witness for scale in [.9,.8,.7,.65]
                if (witness:=interval(width,height,(p,q),boxes,math.floor(maximum*scale))) is not None),None)
            for mode in range(2):
                qi = ri*2+mode
                requested = largest if mode==0 else aggressive
                eligible = cache['eligible'][i,qi]
                positives = int((contained[i,ri]&eligible).sum())
                if positives and requested is None:
                    raise ValueError('bank contradicts independent known-region feasibility')
                status = ('infeasible_integer_geometry' if largest is None else
                    'least_crop_fallback_possible' if requested is None else
                    'coarse_bank_miss' if not positives else 'bank_contains_subject')
                requests.append(dict(id=source['id'],role=source['role'],family_id=source['family_id'],
                    ratio=ratio,mode='maximal' if mode else 'minimal',query=qi,
                    scope='essential' if ri<9 else 'secondary',status=status,
                    eligible_candidates=int(eligible.sum()),containing_candidates=positives,
                    exact_witness_pixels=requested or largest,face_coverage_certified=False))
    baseline = Counter()
    predictions = json.loads(args.old_predictions.read_text())
    for row in predictions:
        i = lookup[row['id']]
        if not known[i] or not row['query_valid']:
            continue
        qi = row['query']
        for name in ['selected','teacher']:
            j = row[name]
            if not cache['eligible'][i,qi,j] or not bank_valid[i,qi//2,j]:
                raise ValueError('old saved crop does not belong to its exact candidate bank')
            baseline[name+'_requests']+=1
            baseline[name+'_contains_subject']+=int(contained[i,qi//2,j])
            baseline[name+'_loses_subject']+=int(not contained[i,qi//2,j])
    tensor = args.root/'retention-targets.pt'
    torch.save(dict(sources=cache['sources'],known=known,fraction=fraction,contained=contained,bank_valid=bank_valid,
        reference_receipts=receipts,rank_cache_sha256=sha(args.cache),source_plan_sha256=sha(plan_path),
        training_eligible=False,face_coverage_certified=False),tensor)
    retain(args.root/'reference-audit.json',dict(counts={k:dict(v) for k,v in counts.items()},
        receipts=receipts,requests=requests,request_counts=dict(Counter(r['status'] for r in requests)),
        by_role={role:dict(Counter(r['status'] for r in requests if r['role']==role)) for role in ['fit','dev']},
        old_development_ranker=dict(baseline),old_predictions_sha256=sha(args.old_predictions),
        retention_cache_sha256=sha(tensor),reference_labels_admitted=False,face_coverage_certified=False,
        human_artistic_quality_measured=False,new_test_inference=False,paid_model_api_calls=0))
    data = dict(sources=review_rows,dataset_kind='photo-subjects',training_eligible=False,face_coverage_certified=False)
    retain(review/'sources.json',data)
    template = Path(__file__).with_name('unsplash_content_audit.html').read_text()
    (review/'index.html').write_text(template.replace('/*DATA*/null',json.dumps(data).replace('</','<\\/')))
    print(json.dumps(dict(counts={k:dict(v) for k,v in counts.items()},requests=len(requests),
        request_counts=dict(Counter(r['status'] for r in requests)),old_development_ranker=dict(baseline),
        retention_cache_sha256=sha(tensor)),indent=2))


if __name__ == '__main__':
    main()
