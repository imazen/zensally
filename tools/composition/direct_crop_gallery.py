#!/usr/bin/env python3
"""Show both direct windows on the existing independently selected DEV witnesses."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import torch
from artifact_identity import sha
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from direct_crop import DirectCropHead,geometry,decode
from subject_geometry import project


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','direct','rank_gallery','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    cache=torch.load(a.direct/'cache/cache.pt',map_location='cpu',weights_only=False)
    if sha(a.direct/'cache/cache.pt')!='2556ff1a29837918d3c383276fe64542157a8cfdb60133a90dc6d9ffee3f9472':raise ValueError('fixed source-feature cache differs')
    reports=json.loads((a.direct/'fits/summary.json').read_text());candidate=min((r for r in reports if r['arm']=='image'),key=lambda r:(r['metrics']['essential']['edge_error'],r['metrics']['essential']['brier'],r['seed']))
    path=a.direct/'fits'/('image-seed'+str(candidate['seed']))/'best.pt'
    if sha(path)!=candidate['weights_sha256']:raise ValueError('selected direct head differs')
    state=torch.load(path,map_location='cpu',weights_only=False);model=DirectCropHead();model.load_state_dict(state['model']);model.eval()
    context=geometry([s['native_size'] for s in cache['sources']],cache['ratios'])
    with torch.no_grad():
        o,s,offer=model((cache['visual']-cache['mean'])/cache['std'],context['query']);_,native=decode(o,s,context,integer=True)
    dev=[i for i,s in enumerate(cache['sources']) if s['role']=='dev']
    if not torch.equal(native[dev],state['replay']['native']):raise ValueError('full DEV direct pixels differ from selected GPU replay')
    if not torch.equal(offer[dev]>=0,state['replay']['offer']>=0):raise ValueError('CPU/GPU offer choices differ')
    original=json.loads((a.rank_gallery/'selections.json').read_text());lookup={s['id']:i for i,s in enumerate(cache['sources'])};rows=[]
    a.output.mkdir(parents=True,exist_ok=False)
    for old in original['sources']:
        i=lookup[old['id']];source=cache['sources'][i];w,h=source['native_size']
        if source['role']!='dev' or old['role']!='dev' or source['family_id']!=old['family_id']:raise ValueError('independent witness role/identity differs')
        image=a.rank_gallery/old['image']
        if sha(image)!=old['image_sha256']:raise ValueError('owned source PNG differs')
        shutil.copy2(image,a.output/old['image'])
        zones=json.loads((a.run/'vision-cache'/source['reference_cache_key']/'zones.json').read_text()) if source['annotation_valid'] else None
        requests={(r['ratio'],r['mode']):r for r in zones['crops']} if zones else {};queries=[]
        for q in range(22):
            ratio=cache['ratios'][q//2];mode='minimal' if q%2==0 else 'maximal';request=requests.get((ratio,mode));prior=next(r for r in old['tasks'] if (r['ratio'],r['mode'])==(ratio,mode))
            teacher=project(request['options'][0]['box'],w,h,ratio,mode) if request and request['action']!='contain' and request['options'] else [0,0,w,h]
            boxes=dict(teacher=teacher,baseline=prior['boxes']['candidate'],candidate=native[i,q,0].int().tolist(),geometry=native[i,q,1].int().tolist())
            queries.append(dict(ratio=ratio,mode=mode,boxes=boxes,teacher_query_informative=bool(request and request['action']!='contain'),
                direct_offer=bool(offer[i,q]>=0),direct_offer_probability=float(offer[i,q].sigmoid())))
        rows.append(dict(old,tasks=queries))
    data=dict(sources=rows,ratios=cache['ratios'],direct_candidate=candidate,rank_candidate=original['candidate'],
        screenshot_budget_bytes=96*1024**2,model_promoted=False,face_coverage_certified=False,full_development_native_replay_exact=True)
    (a.output/'selections.json').write_text(json.dumps(data,indent=2)+'\n')
    template=Path(__file__).with_name('miap_rank_gallery.html').read_text().replace('People composition controls','Direct two-window scout')
    start=template.index('<p>');end=template.index('</p>',start)+4
    template=template[:start]+'<p>Eight DEV witnesses fixed independently by source hash and count/size cell. Compare unreviewed Sol nominations, the earlier hybrid ranker and both raw alternatives of the selected direct model. Raw alternatives remain visible even when its offer head says contain. No artistic or complete-face approval.</p>'+template[end:]
    for old,new in [('Local teacher','Sol / source if contain'),('Previous model','Fixed hybrid ranker'),('Broader candidate','Direct alternative A'),('Geometry control','Direct alternative B'),('Teacher ranking is uninformative for this request.','Sol requests containment, has invalid annotation or supplies no reference for this ratio.')]:template=template.replace(old,new)
    template=template.replace('article.append(title,link,views);',"article.append(title,link,views);const directNote=document.createElement('p');directNote.textContent='Direct offer: '+(task.direct_offer?'crop':'contain')+' (probability '+task.direct_offer_probability.toFixed(3)+'). Raw windows shown for inspection.';article.append(directNote);")
    clip=Path(__file__).with_name('crop_clip.js').read_text();(a.output/'index.html').write_text(template.replace('/*CLIP*/',clip).replace('/*DOCUMENT*/null',json.dumps(data).replace('</','<\\/')))
    print(json.dumps(dict(sources=8,crop_views=704,selected_direct_seed=candidate['seed'],full_dev_native_replay_exact=True)))


if __name__=='__main__':main()
