#!/usr/bin/env python3
"""Native-pixel comparison of raw direct proposals and exact per-person placement."""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import torch
from artifact_identity import sha
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from direct_crop import DirectCropHead,decode,geometry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['direct','regions','projection','cues','rank_gallery','output']:p.add_argument(key,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if sha(a.direct/'cache/cache.pt')!='2556ff1a29837918d3c383276fe64542157a8cfdb60133a90dc6d9ffee3f9472':raise ValueError('fixed direct source cache differs')
    cache=torch.load(a.direct/'cache/cache.pt',map_location='cpu',weights_only=False);lookup={s['id']:i for i,s in enumerate(cache['sources'])}
    audit=json.loads((a.projection/'summary.json').read_text());protocol=json.loads((a.projection/'protocol.json').read_text())
    if not audit['numerical_gate_passed']:raise ValueError('complete projection arithmetic gate required')
    path=a.regions/'none/fits/image-seed31/best.pt'
    if sha(path)!=protocol['weights_sha256']['31']:raise ValueError('fixed preselected seed31 head differs')
    model=DirectCropHead(visual=2095);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);model.eval()
    context=geometry([s['native_size'] for s in cache['sources']],cache['ratios']);visual=torch.cat(((cache['visual']-cache['mean'])/cache['std'],torch.zeros(72,47)),-1)
    with torch.no_grad():o,s,offer=model(visual,context['query']);_,raw=decode(o,s,context,integer=True)
    predictions={(r['id'],r['query'],r['regions']):r for r in json.loads((a.projection/'predictions.json').read_text()) if r['seed']==31 and r['broad_fallback'] and not r['use_model_offer']}
    local={}
    for row in json.loads((a.cues/'analysis/boxes.json').read_text()):local.setdefault(row['id'],{})[row['model']]=row['boxes']
    original=json.loads((a.rank_gallery/'selections.json').read_text());a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for old in original['sources']:
        i=lookup[old['id']];source=cache['sources'][i];w,h=source['native_size'];image=a.rank_gallery/old['image']
        if source['role']!='dev' or source['family_id']!=old['family_id'] or sha(image)!=old['image_sha256']:raise ValueError('fixed independent witness role/pixels differ')
        shutil.copy2(image,a.output/old['image']);queries=[]
        for q in range(22):
            supplied=predictions.get((old['id'],q,'provided'));automatic=predictions.get((old['id'],q,'local_face_pose'))
            known=supplied['native_boxes'] if supplied else [];detected=automatic['native_boxes'] if automatic else []
            queries.append(dict(ratio=cache['ratios'][q//2],mode='minimal' if q%2==0 else 'maximal',
                boxes=dict(teacher=raw[i,q,0].int().tolist(),baseline=list(known[0]) if known else [0,0,w,h],
                    candidate=list(known[1]) if len(known)>1 else [0,0,w,h],geometry=list(detected[0]) if detected else [0,0,w,h]),
                teacher_query_informative=True,evaluated=supplied is not None and automatic is not None,supplied_options=len(known),automatic_options=len(detected),
                model_offer=bool(offer[i,q]>=0),largest_extent_used=bool(supplied and supplied['largest_extent_used'])))
        outlines=[]
        for roi in cache['head_rois'][i]:
            b=[roi[k]*extent/100 for k,extent in zip(['left','top','right','bottom'],[w,h,w,h])]
            outlines.append(dict(kind='supplied',box=[math.floor(b[0]),math.floor(b[1]),math.ceil(b[2]),math.ceil(b[3])]))
        for kind in [protocol['local_face_model'],'float16-pose_landmarker_lite']:
            for b in local[old['id']][kind]:outlines.append(dict(kind='automatic',box=[math.floor(b[0]*w),math.floor(b[1]*h),math.ceil(b[2]*w),math.ceil(b[3]*h)]))
        rows.append(dict(old,tasks=queries,outlines=outlines))
    data=dict(sources=rows,ratios=cache['ratios'],seed=31,weights_sha256=sha(path),projection_protocol=protocol,
        screenshot_budget_bytes=96*1024**2,model_promoted=False,face_coverage_certified=False)
    (a.output/'selections.json').write_text(json.dumps(data,indent=2)+'\n')
    template=Path(__file__).with_name('miap_rank_gallery.html').read_text().replace('People composition controls','Exact per-person origin placement')
    first=template.index('<p>');last=template.index('</p>',first)+4
    template=template[:first]+('<p>Eight DEV witnesses fixed independently by source hash and count/size cell. Seed31 direct head; no re-selection for these images. Gold outlines are unreviewed supplied heads; cyan are '+protocol['local_face_model']+' plus pose estimates. Full-source comparison panes mean contain or no second option. Geometry permits other people to be wholly excluded. Largest-extent Maximal fallback occurs only when the predicted extent cannot place a legal crop. Learned offer decisions are shown separately. No complete-face or artistic approval.</p>')+template[last:]
    for old,new in [('Local teacher','Raw direct A'),('Previous model','Supplied-region option A'),('Broader candidate','Supplied-region option B'),('Geometry control','Automatic-region option A')]:template=template.replace(old,new)
    template=template.replace('svg.append(image);figure.append(caption,svg);',"svg.append(image);if(!box)for(const roi of row.outlines){const b=roi.box,rect=document.createElementNS(ns,'rect');rect.setAttribute('x',b[0]);rect.setAttribute('y',b[1]);rect.setAttribute('width',b[2]-b[0]);rect.setAttribute('height',b[3]-b[1]);rect.setAttribute('fill','none');rect.setAttribute('stroke',roi.kind==='supplied'?'#ffe066':'#43e5ed');rect.setAttribute('stroke-width',w/350);svg.append(rect)}figure.append(caption,svg);")
    template=template.replace('article.append(title,link,views);',"article.append(title,link,views);const note=document.createElement('p');note.textContent=task.evaluated?'Supplied-region options: '+task.supplied_options+'; automatic options: '+task.automatic_options+'; learned offer: '+(task.model_offer?'crop':'contain')+(task.largest_extent_used?'; largest-extent Maximal fallback':'')+'. Full source is shown when an option is absent.':'No geometric audit for this ratio/source: annotation is invalid or the ratio was not requested. Comparison panes retain the source.';article.append(note);")
    clip=Path(__file__).with_name('crop_clip.js').read_text();(a.output/'index.html').write_text(template.replace('/*CLIP*/',clip).replace('/*DOCUMENT*/null',json.dumps(data).replace('</','<\\/')))
    print(json.dumps(dict(sources=8,crop_views=704,seed=31,weights_sha256=sha(path))))


if __name__=='__main__':main()
