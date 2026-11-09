#!/usr/bin/env python3
"""Actual-pixel viewer on the eight independently hash-selected Sol DEV sources."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import torch
from artifact_identity import sha
from fit_unsplash_ranks import encode
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from local_rank_distillation import LocalCropRanker
from subject_geometry import project

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','rank_cache','initializer','output']:p.add_argument(key,type=Path)
    p.add_argument('--bank-study',type=Path)
    a=p.parse_args()
    if sha(a.rank_cache)!='420b7fad28e64af4292f1c155d0447abd106b0fe190bbd07bb88d750d1df804d' or sha(a.initializer)!='c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df':raise ValueError('fixed gallery parents differ')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache_path=a.bank_study/'cache/cache.pt' if a.bank_study else a.run/'set-cache/cache.pt'
    fits=a.bank_study/'fits' if a.bank_study else a.run/'fits'
    cache=torch.load(cache_path,map_location='cpu',weights_only=False)
    original=torch.load(a.rank_cache,map_location='cpu',weights_only=False);lookup={s['id']:i for i,s in enumerate(original['sources'])}
    dev=[i for i,s in enumerate(cache['sources']) if s['role']=='dev'];positions={cache['sources'][i]['id']:j for j,i in enumerate(dev)}
    tasks=[json.loads(Path(p).read_text()) for p in json.loads((a.run/'real-sol-tasks/tasks.json').read_text())]
    models=json.loads((fits/'summary.json').read_text())
    candidate=min((r for r in models if r['old_nonregression'] and (not a.bank_study or r['arm']=='frozen-image')),key=lambda r:(r['new']['essential']['explicit_face_cut'],-r['new']['essential']['reference_match'],r['old']['essential']['teacher_regret'],r['seed']))
    if candidate['arm']!='frozen-image':raise ValueError('registered review candidate arm differs')
    paths=dict(baseline=a.initializer,candidate=fits/('frozen-image-seed'+str(candidate['seed']))/'best.pt',
        geometry=fits/('geometry-only-seed'+str(candidate['seed']))/'best.pt')
    per_mode='query_boxes' in cache
    image=cache['image'][dev].cuda();content=cache['content'][dev].cuda();boxes=cache['query_boxes' if per_mode else 'boxes'][dev].cuda();eligible=cache['eligible'][dev].cuda()
    aspects=torch.tensor([cache['sources'][i]['native_size'][0]/cache['sources'][i]['native_size'][1] for i in dev],device='cuda')
    choices={}
    for name,path in paths.items():
        model=LocalCropRanker().cuda();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
        model.configure('geometry-only' if name=='geometry' else 'trainable-image');model.eval()
        with torch.no_grad():
            features=encode(model,image,list(range(len(dev))));selected=[]
            for q in range(22):
                p,r=map(int,cache['ratios'][q//2].split(':'))
                raw=model.scores(features,boxes[:,q if per_mode else q//2],aspects,aspects.new_full((len(dev),),p/r),aspects.new_full((len(dev),),q%2),content)
                if not bool(torch.isfinite(raw).all()):raise ValueError('nonfinite gallery inference')
                selected.append(raw.masked_fill(~eligible[:,q],-torch.inf).argmax(-1).cpu().tolist())
            choices[name]=selected
        if name in ['candidate','geometry']:
            arm='frozen-image' if name=='candidate' else 'geometry-only'
            saved=json.loads((fits/(arm+'-seed'+str(candidate['seed']))/'best-predictions.json').read_text())['new']
            for row in saved:
                if choices[name][row['query']][positions[row['id']]]!=row['selected']:raise ValueError('native viewer selection differs from complete development replay')
        del model
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for task in tasks:
        sid=task['id'];i=lookup[sid];source=original['sources'][i];j=positions[sid];w,h=source['native_size']
        path=Path(source['png'])
        if source['role']!='dev' or sha(path)!=source['png_sha256']:raise ValueError('owned DEV source identity differs')
        filename=sid.split(':')[1]+'.png';shutil.copy2(path,a.output/filename)
        result=json.loads((a.run/'real-terra'/(sid.replace(':','-')+'.json')).read_text());zones=json.loads((Path(result['entry'])/'zones.json').read_text())
        proposed={(r['ratio'],r['mode']):r for r in zones['crops']};queries=[]
        for q in range(22):
            ratio=cache['ratios'][q//2];mode='minimal' if q%2==0 else 'maximal';request=proposed.get((ratio,mode))
            bank=[[round(float(value)*extent) for value,extent in zip(box,[w,h,w,h])] for box in boxes[j,q].cpu()] if per_mode else original['native_boxes'][i][q//2]
            teacher=project(request['options'][0]['box'],w,h,ratio,mode) if request and request['action']!='contain' and request['options'] else [0,0,w,h]
            b=dict(teacher=teacher,**{name:bank[choice[q][j]] for name,choice in choices.items()})
            queries.append(dict(ratio=ratio,mode=mode,boxes=b,teacher_query_informative=bool(request and request['action']!='contain'),
                teacher_status=request['action'] if request else 'not requested'))
        head_rois=[dict(provider='Terra',box=r['box']) for r in zones['rois'] if r['kind']=='face_head']
        sol=json.loads((a.run/'real-sol'/(sid.replace(':','-')+'.json')).read_text())
        if sol['annotation_valid']:
            head_rois +=[dict(provider='Sol',box=r['box']) for r in json.loads((Path(sol['entry'])/'zones.json').read_text())['rois'] if r['kind']=='face_head']
        rows.append(dict(id=sid,role='dev',family_id=source['family_id'],group_cell=task['cell'],native_size=[w,h],image=filename,
            image_sha256=sha(path),tasks=queries,head_rois=head_rois,**source['source']))
    data=dict(sources=rows,ratios=cache['ratios'],candidate=candidate,bank=cache['protocol'].get('bank','original-shared'),screenshot_budget_bytes=96*1024**2,model_promoted=False,face_coverage_certified=False)
    (a.output/'selections.json').write_text(json.dumps(data,indent=2)+'\n')
    template=Path(__file__).with_name('miap_rank_gallery.html').read_text()
    template=template.replace('People composition controls','Weak subject-choice controls').replace('Eight development sources fixed by count/scale cell and source hash independently of detector and crop-model predictions. Compare the cached local teacher, the previous photographic model, the selected broader candidate and a geometry control. These are research proposals, with no face-safety or artistic-quality approval. Minimal keeps the largest available area; Maximal allows tighter framing.',
        'Eight DEV sources independently selected for Sol checks by count/size cell and source hash. Terra windows are weak proposals, including uncertain ones. A whole-source teacher pane means contain or no reference at this ratio. Compare initializer, frozen-encoder candidate and geometry control; no model or face-safety promotion. Source outlines show Terra head proposals in red and Sol in yellow; backs of heads can be included, so these are not facial-recall labels.')
    template=template.replace("'Local teacher'","'Terra proposal / source if contain'").replace("'Previous model'","'Initializer'").replace("'Broader candidate'","'Frozen-encoder candidate'")
    template=template.replace("'Teacher ranking is uninformative for this request.'","'Terra requests containment or supplied no reference for this ratio; teacher pane retains the source.'")
    if a.bank_study:template=template.replace('Weak subject-choice controls','Subject bank control: '+cache['protocol']['bank']).replace('Compare initializer, frozen-encoder candidate and geometry control;','Compare initializer, frozen-encoder candidate and geometry control using the same research bank;')
    template=template.replace('svg.append(image);figure.append(caption,svg);','''svg.append(image);if(!box)for(const roi of row.head_rois){const b=roi.box,rect=document.createElementNS(ns,'rect');rect.setAttribute('x',b.left*w/100);rect.setAttribute('y',b.top*h/100);rect.setAttribute('width',(b.right-b.left)*w/100);rect.setAttribute('height',(b.bottom-b.top)*h/100);rect.setAttribute('fill','none');rect.setAttribute('stroke',roi.provider==='Terra'?'#ff6565':'#ffe066');rect.setAttribute('stroke-width',w/300);svg.append(rect)}figure.append(caption,svg);''')
    clip=Path(__file__).with_name('crop_clip.js').read_text()
    (a.output/'index.html').write_text(template.replace('/*CLIP*/',clip).replace('/*DOCUMENT*/null',json.dumps(data).replace('</','<\\/')))
    print(json.dumps(dict(sources=8,crop_views=704,selected_candidate=candidate['weights_sha256'],full_development_choices_replayed=True)))

if __name__=='__main__':main()

