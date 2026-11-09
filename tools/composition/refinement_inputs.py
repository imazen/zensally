"""Pinned local head proposals and unchanged accepted reference geometry."""
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from artifact_identity import sha
from audit_face_corroboration import verify_member
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from subject_geometry import audit,pixels,project

FACE_MODELS={'native-ultraface','native-yunet','native-yunet-tiled','native-yunet-tiled-interior','float16-blaze_face_short_range','float16-blaze_face_full_range_sparse','float16-face_landmarker'}


def local_heads(contexts,faces,pose,original):
    expected=[(faces,'c7872a992e607cb062089fdd678a0629a16d9f86841c831a4020cd9b8a579b29'),(pose,'839b366e1c2e210a0ba1656916f9e61bef55a6ea999ae83c0b9ec2d7a585b2e7'),(original,'0c1909193ec1b219adb22a005cfbd1e32d09929e6890ef0d6e8820c28b8f54e5')]
    if any(sha(root/'SHA256SUMS.json')!=digest for root,digest in expected):raise ValueError('frozen head proposal packets differ')
    for root,relative in [(faces,'analysis/boxes.json'),(pose,'pose-heads.json'),(original,'regions/boxes.json'),(original,'regions/pose-heads.json')]:verify_member(root,relative)
    rows={'new':json.loads((faces/'analysis/boxes.json').read_text()),'old':json.loads((original/'regions/boxes.json').read_text())};poses={'new':json.loads((pose/'pose-heads.json').read_text())['source_boxes'],'old':{s['id']:[] for s in contexts['old']['data']['sources']}}
    for row in json.loads((original/'regions/pose-heads.json').read_text()):poses['old'][row['id']].append(row['native_box'])
    result={}
    for cohort,ctx in contexts.items():
        sources={s['id']:s for s in ctx['data']['sources']};parsed={sid:[] for sid in sources}
        for row in rows[cohort]:
            if row['model']=='float16-pose_landmarker_lite':continue
            if row['model'] not in FACE_MODELS:raise ValueError('unregistered face proposal model')
            sid=row['id'];w,h=sources[sid]['native_size']
            for b in row['boxes']:
                box=[max(0,math.floor(b[0]*w)),max(0,math.floor(b[1]*h)),min(w,math.ceil(b[2]*w)),min(h,math.ceil(b[3]*h))]
                if not 0<=box[0]<box[2]<=w or not 0<=box[1]<box[3]<=h:raise ValueError('invalid native face proposal')
                parsed[sid].append(box)
        result[cohort]={sid:dict(faces=[list(b) for b in sorted(set(map(tuple,parsed[sid])))],**{'faces-pose':[list(b) for b in sorted(set(map(tuple,[*parsed[sid],*poses[cohort][sid]])))]}) for sid in sources}
    return result,{str(root/'SHA256SUMS.json'):digest for root,digest in expected}


def accepted_options(context,run):
    c=context['data'];options={};identities={};source_heads={};replayed=0
    for i,source in enumerate(c['sources']):
        w,h=source['native_size'];scale=np.array([w,h,w,h],np.float64);source_heads[source['id']]=[]
        if not source['annotation_valid']:
            if c['positive'][i].any() or c['reference_valid'][i].any():raise ValueError('invalid annotation has admitted labels')
            continue
        entry=run/'vision-cache'/source['reference_cache_key'];hashes=json.loads((entry/'hashes.json').read_text())
        for name,digest in hashes.items():
            if sha(entry/name)!=digest:raise ValueError('cached reference member differs')
            identities[str(entry/name)]=digest
        metadata=json.loads((entry/'metadata.json').read_text());zones=json.loads((entry/'zones.json').read_text())
        if metadata['source_id']!=source['id'] or metadata['source_role']!=source['role'] or metadata['image_sha256']!=source['png_sha256'] or metadata['cache_key']!=source['reference_cache_key'] or zones['source_id']!=source['id'] or zones['purpose']!='editorial':raise ValueError('reference source identity differs')
        source_heads[source['id']]=[pixels(r['box'],w,h,outward=True) for r in zones['rois'] if r['kind']=='face_head'];requests={(r['ratio'],r['mode']):r for r in zones['crops']}
        for q in range(22):
            ratio=c['ratios'][q//2];mode='minimal' if q%2==0 else 'maximal';request=requests.get((ratio,mode));good=[];positive=np.zeros(128,bool);native=context['native'][i].reshape(22,128,4)[q].tolist()
            for option in request['options'] if request and request['action']!='contain' else []:
                raw=pixels(option['box'],w,h);exact=project(option['box'],w,h,ratio,mode);args=(zones['rois'],option['selected_subject_ids'],option['excluded_subject_ids'],w,h)
                if option['face_policy_check']!='whole_faces_or_excluded' or not audit(raw,*args,outward=False)['geometry_compatible'] or not audit(exact,*args)['geometry_compatible'] or max(np.abs(np.asarray(exact)-raw)/scale*128)>3:continue
                good.append(dict(target=np.asarray(exact,np.float64)/scale,args=args));close=(np.abs(np.asarray(native)/scale-np.asarray(exact)/scale)*128).max(-1)<=3
                for k in np.flatnonzero(close&c['eligible'][i,q].numpy()):positive[k]|=audit(native[k],*args)['geometry_compatible']
            if len(good)!=int(c['reference_valid'][i,q].sum()) or not np.array_equal(positive,c['positive'][i,q].numpy()):raise ValueError('accepted options or candidate-positive masks differ')
            options[source['id'],q]=good;replayed+=1
    return options,source_heads,identities,replayed


def reference_match(box,options,width,height):
    if box is None:return False
    scale=np.array([width,height,width,height],np.float64)
    return any(max(np.abs(np.asarray(box)/scale-row['target'])*128)<=3 and audit(box,*row['args'])['geometry_compatible'] for row in options)
