#!/usr/bin/env python3
"""Render fixed people witnesses: source, teacher, initializer, candidate, geometry."""
import argparse
import json
from pathlib import Path
import shutil

import torch

from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','fits','analysis','plan','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    if sha(a.cache)!='420b7fad28e64af4292f1c155d0447abd106b0fe190bbd07bb88d750d1df804d' or sha(a.plan)!='370525cfc88ec3e5070a915ba3107d95dfa9070f96523c31eb42277c505d7307':
        raise ValueError('fixed cache/review selection differs')
    cache=torch.load(a.cache,map_location='cpu',weights_only=False);analysis=json.loads((a.analysis/'summary.json').read_text())
    candidate=analysis['candidate_for_review'];seed=candidate['seed'];plan=json.loads(a.plan.read_text())
    paths=dict(candidate=a.fits/(candidate['arm']+'-seed'+str(seed))/'best-predictions.json',
        baseline=a.fits/'trainable-image-seed31/predictions-000.json',geometry=a.fits/('geometry-only-seed'+str(seed))/'best-predictions.json')
    predictions={key:{(r['id'],r['query']):r for r in json.loads(path.read_text())} for key,path in paths.items()}
    a.output.mkdir(parents=True,exist_ok=False);sources=[]
    for selected in plan['sources']:
        index=next(i for i,s in enumerate(cache['sources']) if s['id']==selected['id']);source=cache['sources'][index]
        if source['role']!='dev' or source['family_id']!=selected['family_id']:raise ValueError('review source/role differs')
        original=Path(source['png'])
        if sha(original)!=source['png_sha256']:raise ValueError('owned review preview differs')
        name=source['id'].split(':')[1]+'.png';shutil.copy2(original,a.output/name);tasks=[]
        for q in range(22):
            rows={key:values[(source['id'],q)] for key,values in predictions.items()}
            if len({r['teacher'] for r in rows.values()})!=1:raise ValueError('review teacher choices differ')
            bank=cache['native_boxes'][index][q//2];choices=dict(teacher=rows['candidate']['teacher'],**{key:row['selected'] for key,row in rows.items()})
            boxes={key:bank[j] for key,j in choices.items()};w,h=source['native_size'];p,d=map(int,cache['ratios'][q//2].split(':'))
            if any(not(0<=l<r<=w and 0<=t<b<=h and (r-l)*d==(b-t)*p) for l,t,r,b in boxes.values()):
                raise ValueError('native crop/aspect geometry differs')
            tasks.append(dict(query=q,ratio=cache['ratios'][q//2],mode='minimal' if q%2==0 else 'maximal',
                choices=choices,boxes=boxes,teacher_query_informative=rows['candidate']['query_valid']))
        sources.append(dict(selected,image=name,image_sha256=sha(original),native_size=source['native_size'],
            source_url=source['source']['source_url'],photographer=source['source']['photographer'],author_url=source['source']['author_url'],
            listed_license=source['source']['listed_license'],tasks=tasks))
    data=dict(sources=sources,ratios=cache['ratios'],candidate=candidate,review_plan_sha256=sha(a.plan),
        prediction_sha256={key:sha(path) for key,path in paths.items()},screenshot_budget_bytes=plan['screenshot_budget_bytes'],
        face_coverage_certified=False,human_artistic_quality_measured=False,model_promoted=False,paid_model_api_calls=0)
    (a.output/'selections.json').write_text(json.dumps(data,indent=2)+'\n')
    template=Path(__file__).with_suffix('.html').read_text();clip=Path(__file__).with_name('crop_clip.js').read_text()
    (a.output/'index.html').write_text(template.replace('/*CLIP*/',clip).replace('/*DOCUMENT*/null',json.dumps(data).replace('</','<\\/')))
    print(json.dumps(dict(sources=8,requests=176,crop_views=704)))


if __name__=='__main__':main()
