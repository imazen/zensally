#!/usr/bin/env python3
"""Cache broader photographic rankings from a fixed local composition teacher."""
import argparse
import atexit
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from artifact_identity import sha
from cached_focalnet import CachedContext,session
from constrained_bank import candidates
from focalnet_bank import RATIOS
from rank_targets import targets as relative_targets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['auxiliary','acquired','models','split','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    full_path=a.models/'focalnet/focalnet-human.onnx'
    for path,digest in [(a.auxiliary,'b986f2740585449a390177e199fda71eefd1bb4034c49fce50e4d667fbdb4bc4'),
        (a.acquired/'SHA256SUMS.json','93f63d69180d82f197e34a733005118de260eda728e18998ced3194959581bb1'),
        (full_path,'59164c601c98cea3f62b25166710831dac63e1a872fc64767c65316ad5385439'),
        (a.split/'analysis.onnx','a5f322876a6d53b9221e1eca6d6a6b95e91b7b911e7003d660444492ca504768'),
        (a.split/'query.onnx','8d14feeb5218baea8d885bba17e5b13b21e5c4dc6b6942a4d5de63adc51dc7ad')]:
        if sha(path)!=digest:raise ValueError('frozen broader input/teacher differs')
    torch.set_num_threads(2);aux=torch.load(a.auxiliary,map_location='cpu',weights_only=False);sources=aux['sources']
    members={r['path']:r['sha256'] for r in json.loads((a.acquired/'SHA256SUMS.json').read_text())}
    if sha(a.acquired/'manifest.json')!=members['manifest.json']:raise ValueError('source attribution manifest differs')
    originals={r['id']:r for r in json.loads((a.acquired/'manifest.json').read_text())['rows']}
    for source in sources:
        original=originals[source['id']];raw=original['source']['source_information']['source']
        if original['source']['role']!=source['role'] or original['source']['family_id']!=source['family_id']:
            raise ValueError('source/role attribution join differs')
        source.update(source=dict(source_url=raw['OriginalLandingURL'],photographer=raw['Author'],
            author_url=raw['AuthorProfileURL'],listed_license=raw['License']),
            png=str(a.acquired/source['id'].split(':')[1]/'preview.png'),png_sha256=original['preview.png_sha256'],
            working_size=[original['geometry']['width'],original['geometry']['height']])
    if Counter(s['role'] for s in sources)!=Counter(fit=275,dev=91) or len({s['family_id'] for s in sources})!=366:
        raise ValueError('fixed screened366 fit/development sources required')
    a.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        auxiliary_cache_sha256=sha(a.auxiliary),source_manifest_sha256=sha(a.acquired/'manifest.json'),teacher_sha256=sha(full_path),analysis_threads=4,query_threads=1,
        sources=366,fit_sources=275,development_sources=91,test_sources=0,purpose='unspecified weak local relative rank',
        teacher_labels_are_not_artistic_acceptability=True,face_coverage_certified=False,paid_model_api_calls=0)
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    status=dict(pid=os.getpid(),stage='selecting-numerical-witnesses',completed=0,required=366)
    def update(**fields):status.update(fields,checked_unix=time.time());(a.output/'progress.json').write_text(json.dumps(status)+'\n')
    def failed():
        if status['pid'] is not None:update(pid=None,stage='failed-artifacts-retained')
    atexit.register(failed);update()
    image=aux['image'].permute(0,3,1,2).float()/255;bounds=aux['content'].numpy();witnesses=[]
    for role in ['fit','dev']:
        used=set()
        for aspect in [.5,1.,2.]:
            choices=[i for i,s in enumerate(sources) if s['role']==role and s['family_id'] not in used]
            index=min(choices,key=lambda i:(abs(math.log(sources[i]['native_size'][0]/sources[i]['native_size'][1]/aspect)),sources[i]['id']))
            witnesses.append(index);used.add(sources[index]['family_id'])
    context=CachedContext(a.split,threads=4,query_threads=1);full=session(full_path,threads=4)
    mean=np.array([.485,.456,.406],np.float32)[:,None,None];std=np.array([.229,.224,.225],np.float32)[:,None,None]
    checks=[];collected={};counts=Counter()
    for index in witnesses+[i for i in range(366) if i not in witnesses]:
        source=sources[index];normalized=np.zeros((1,3,256,256),np.float32)
        x0,y0,x1,y1=np.rint(bounds[index]*256).astype(int)
        normalized[0,:,y0:y1,x0:x1]=((image[index].numpy()-mean)/std)[:,y0:y1,x0:x1]
        token=context.analyze(normalized,bounds[index:index+1],(source['id'],source['input_sha256']))
        boxes=[];scores=[];eligible=[];known=[];pairs=[];pair_valid=[];targets=[];native_boxes=[]
        width,height=source['native_size']
        for ratio,_ in RATIOS:
            p,q=map(int,ratio.split(':'));native,mw,mh=candidates(width,height,(p,q),[])
            if not native or len(native)>128 or any(not(0<=l<r<=width and 0<=t<b<=height and (r-l)*q==(b-t)*p) for l,t,r,b in native):
                raise ValueError('native exact-ratio candidate bank differs')
            padded=np.zeros((1,128,4),np.float32);padded[0,:len(native)]=np.array(native,np.float64)/[width,height,width,height]
            value=context.score(padded,token)[0].copy()
            if index in witnesses:
                reference=full.run(['crop_scores'],dict(image=normalized,boxes=padded,content=bounds[index:index+1]))[0][0]
                exact=np.array_equal(value.view(np.uint32),reference.view(np.uint32))
                checks.append(dict(id=source['id'],ratio=ratio,score_bits_exact=exact,max_absolute_error=float(np.abs(value-reference).max())))
                if not exact:raise ValueError('new source cached/full score bits differ')
            boxes.append(padded[0]);scores.append(value);native_boxes.append(native)
            for mode in ['minimal','maximal']:
                selected=[j for j,b in enumerate(native) if (b[2]-b[0]==mw if mode=='minimal' else b[2]-b[0]<mw)]
                if not selected and mode=='maximal':selected=list(range(len(native)))
                mask,available,target,pair,pm=relative_targets(value,selected)
                eligible.append(mask);known.append(available);targets.append(target);pairs.append(pair);pair_valid.append(pm)
                counts['queries']+=1;counts['ranking_queries']+=int(available);counts['pairs']+=int(pm.sum())
        collected[index]=(boxes,scores,eligible,known,pairs,pair_valid,targets,native_boxes)
        update(stage='numerical-witnesses' if len(collected)<6 else 'relative-rank-cache',completed=len(collected))
    if len(checks)!=66 or not all(c['score_bits_exact'] for c in checks):raise ValueError('complete66 numerical witnesses required')
    arrays=list(zip(*(collected[i] for i in range(366))))
    cache=dict(schema='miap-local-rank-cache-v1',sources=sources,image=image,content=aux['content'],
        **{key:torch.tensor(np.array(values)) for key,values in zip(['boxes','teacher_scores','eligible','query_valid','pairs','pair_valid','targets'],arrays[:7])},
        native_boxes=arrays[7],ratios=[ratio for ratio,_ in RATIOS],teacher_sha256=sha(full_path),
        purpose='unspecified weak local relative rank',training_label_origin='local FocalNet ranking, not artistic acceptability',
        canonical_holdouts_opened=False,new_test_composition_scores_opened=False,face_coverage_certified=False,diagnostic_maps_saved=False)
    torch.save(cache,a.output/'cache.pt')
    (a.output/'cached-full-scout.json').write_text(json.dumps(checks,indent=2)+'\n')
    summary=dict(sources=366,fit_sources=275,development_sources=91,counts=dict(counts),cache_sha256=sha(a.output/'cache.pt'),
        cache_bytes=(a.output/'cache.pt').stat().st_size,full_cached_scout_checks=66,full_cached_scout_all_bits_exact=True,
        test_sources=0,artistic_labels=False,face_coverage_certified=False,paid_model_api_calls=0)
    if summary['cache_bytes']>1024**3:raise ValueError('1GiB broader-rank cache budget exhausted')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');update(pid=None,stage='complete',completed=366)
    print(json.dumps(summary))


if __name__=='__main__':main()
