#!/usr/bin/env python3
"""Counterfactual weak-region crop bank; shipping face guards remain closed."""
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import numpy as np
import torch
import onnxruntime as ort
from crop_policy import Rect,_region_windows
from focalnet_bank import RATIOS

def normalized(rect,width,height):return np.asarray([rect.left/width,rect.top/height,rect.right/width,rect.bottom/height],np.float32)
def contains(box,region):return box[0]<=region[0] and box[1]<=region[1] and box[2]>=region[2] and box[3]>=region[3]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('models',type=Path);p.add_argument('auxiliary',type=Path);p.add_argument('annotations',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False);sys.path.insert(0,str(a.models/'focalnet-source/src'));from focalnet.imaging import Letterbox,restore_map
 from focalnet.crop import Crop,score_crop
 options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL;session=ort.InferenceSession(str(a.models/'focalnet/focalnet-human.onnx'),sess_options=options,providers=['CPUExecutionProvider']);aux=torch.load(a.auxiliary/'auxiliary.pt',map_location='cpu',weights_only=False);lookup={s['id']:i for i,s in enumerate(aux['sources'])};mean=np.array([.485,.456,.406],np.float32)[:,None,None];std=np.array([.229,.224,.225],np.float32)[:,None,None];counts={'sources':0,'requests':0,'contain':0,'minimal_fallback':0,'old_selected_region_violations':0,'selected_region_violations':0}
 for annotation in json.loads(a.annotations.read_text())['sources']:
  id=annotation['id'];i=lookup[id];s=aux['sources'][i];old=json.loads((a.models/'focalnet-bank'/(id+'.json')).read_text());w,h=old['width'],old['height']
  if s['rgb_sha256']!=annotation['rgb_sha256']:raise ValueError('region/source RGB differs')
  # Round protected witness bounds outward to source-pixel boundaries.
  regions=[Rect(math.floor(r['box_percent'][0]*w/100),math.floor(r['box_percent'][1]*h/100),math.ceil(r['box_percent'][2]*w/100),math.ceil(r['box_percent'][3]*h/100)) for r in annotation['regions']];regions.extend(Rect(math.floor(b[0]*w),math.floor(b[1]*h),math.ceil(b[2]*w),math.ceil(b[3]*h)) for b in old['heads']);nr=[[r.left/w,r.top/h,r.right/w,r.bottom/h] for r in regions];valid=aux['packed']['valid_rect'][i].numpy();x0,y0,x1,y1=np.rint(valid*256).astype(int);image=np.zeros((1,3,256,256),np.float32);image[0,:,y0:y1,x0:x1]=((aux['image'][i].numpy()-mean)/std)[:,y0:y1,x0:x1];tasks=[]
  for ratio,aspect in RATIOS:
   old_task=next(t for t in old['tasks'] if t['ratio']==ratio);candidates=[np.asarray(c['box'],np.float32) for c in old_task['candidates'] if all(contains(c['box'],r) for r in nr)]
   for scale in [1.,.9,.8,.7,.65]:
    windows=_region_windows(w,h,tuple(map(int,ratio.split(':'))),regions,(scale,),3)
    for j in sorted({0,len(windows)//2,len(windows)-1}):
     if j<0 or not windows:continue
     b=normalized(windows[j],w,h)
     # No clipping tolerance after casting coordinates to the model's f32 contract.
     if all(contains(b,r) for r in nr) and not any(np.array_equal(b,c) for c in candidates):candidates.append(b)
   if len(candidates)>128:raise ValueError('bounded semantic bank exceeds 128')
   if not candidates:
    tasks.append({'ratio':ratio,'candidates':[],'selected':[{'mode':m,'raw_candidate':None,'detected_head_guarded_candidate':None,'minimal_fallback':False,'face_coverage_certified':False,'deployable':False} for m in ['minimal','maximal']]});counts['contain']+=2;counts['requests']+=2
    for previous in old_task['selected']:
     j=previous['detected_head_guarded_candidate'];counts['old_selected_region_violations']+=int(j is not None and not all(contains(old_task['candidates'][j]['box'],r) for r in nr))
    continue
   candidates=np.stack(candidates);padded=np.zeros((1,128,4),np.float32);padded[0,:len(candidates)]=candidates;importance,scores=session.run(['importance','crop_scores'],{'image':image,'boxes':padded,'content':valid[None]});values=restore_map(importance[0,0],Letterbox(256,x0,y0,x1-x0,y1-y0));retention=[]
   if not np.isfinite(scores).all() or not np.isfinite(importance).all():raise ValueError('nonfinite semantic crop output')
   for b in candidates:
    left,top,right,bottom=b*[w,h,w,h];retention.append(score_crop(values,Crop(round(float(left)),round(float(top)),max(1,round(float(right))-round(float(left))),max(1,round(float(bottom))-round(float(top))),0),w,h))
   area=(candidates[:,2]-candidates[:,0])*(candidates[:,3]-candidates[:,1]);largest=min(1,aspect/(w/h))*min(1,(w/h)/aspect);selected=[]
   for mode in ['minimal','maximal']:
    eligible=np.flatnonzero(area>=largest-1e-6 if mode=='minimal' else area<=largest*.810001);fallback=False
    if not len(eligible):eligible=np.arange(len(candidates));fallback=mode=='maximal'
    maximum=max(retention[j] for j in eligible);eligible=[j for j in eligible if retention[j]>=maximum-.05];chosen=int(max(eligible,key=lambda j:float(scores[0,j])));counts['requests']+=1;counts['minimal_fallback']+=int(fallback);counts['selected_region_violations']+=int(not all(contains(candidates[chosen],r) for r in nr));selected.append({'mode':mode,'raw_candidate':chosen,'detected_head_guarded_candidate':chosen,'minimal_fallback':fallback,'face_coverage_certified':False,'deployable':False})
    previous=next(s for s in old_task['selected'] if s['mode']==mode);j=previous['detected_head_guarded_candidate'];counts['old_selected_region_violations']+=int(j is not None and not all(contains(old_task['candidates'][j]['box'],r) for r in nr))
   tasks.append({'ratio':ratio,'candidates':[{'box':b.tolist(),'score':float(scores[0,j]),'retained_importance':float(retention[j]),'contains_detected_heads':True,'contains_weak_regions':True} for j,b in enumerate(candidates)],'selected':selected})
  result={k:old[k] for k in ['id','family_id','use','width','height','heads']};result.update(tasks=tasks,semantic_regions=annotation['regions'],face_coverage_certified=False,new_training_labels_admitted=False,deployable=False)
  with (a.output/(id+'.json')).open('x') as f:json.dump(result,f);f.write('\n')
  counts['sources']+=1
 (a.output/'summary.json').write_text(json.dumps(counts|{'scope':'counterfactual containment of eight unreviewed agent region lists; not face certification or independent composition quality','shipping_coverage_guard_changed':False},indent=2)+'\n');print(json.dumps(counts))
if __name__=='__main__':main()
