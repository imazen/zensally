#!/usr/bin/env python3
"""Full two-dimensional photographic candidate bank; scores/coordinates only."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from fit_heads import sha

RATIOS=[('1:1',1.),('4:3',4/3),('3:4',3/4),('16:9',16/9),('9:16',9/16),('2:1',2.),('1:2',.5),('3:1',3.),('1:3',1/3),('6:1',6.),('1:6',1/6)]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('models',type=Path);p.add_argument('first_gate',type=Path);p.add_argument('auxiliary',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False);sys.path.insert(0,str(a.models/'focalnet-source/src'));from focalnet.candidates import generate_candidates
 from focalnet.imaging import Letterbox,restore_map
 from focalnet.crop import Crop,score_crop
 options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL;session=ort.InferenceSession(str(a.models/'focalnet/focalnet-human.onnx'),sess_options=options,providers=['CPUExecutionProvider']);aux=torch.load(a.auxiliary/'auxiliary.pt',map_location='cpu',weights_only=False);lookup={s['id']:s for s in json.loads((a.first_gate/'inputs.json').read_text())['images']};counts={'sources':0,'requests':0,'no_detected_head_legal_candidate':0,'raw_cuts_detected_head':0};mean=np.array([.485,.456,.406],np.float32)[:,None,None];std=np.array([.229,.224,.225],np.float32)[:,None,None]
 for i,source in enumerate(aux['sources']):
  id=source['id'];s=lookup[id];w,h=s['width'],s['height'];valid=aux['packed']['valid_rect'][i].numpy();x0,y0,x1,y1=np.rint(valid*256).astype(int);image=np.zeros((1,3,256,256),np.float32);image[0,:,y0:y1,x0:x1]=((aux['image'][i].numpy()-mean)/std)[:,y0:y1,x0:x1];heads=[p['head'] for p in json.loads((a.first_gate/'gaze-all'/(id+'.json')).read_text())['result']['people']];tasks=[]
  for ratio,aspect in RATIOS:
   candidates=generate_candidates(w,h,aspect);padded=np.zeros((1,128,4),np.float32);padded[0,:len(candidates)]=candidates;importance,scores=session.run(['importance','crop_scores'],{'image':image,'boxes':padded,'content':valid[None]});scores=scores[0,:len(candidates)]
   if not np.isfinite(scores).all() or not np.isfinite(importance).all():raise ValueError('nonfinite crop output')
   values=restore_map(importance[0,0],Letterbox(256,x0,y0,x1-x0,y1-y0));retention=[]
   for b in candidates:
    left,top,right,bottom=b*[w,h,w,h];retention.append(score_crop(values,Crop(round(float(left)),round(float(top)),max(1,round(float(right))-round(float(left))),max(1,round(float(bottom))-round(float(top))),0),w,h))
   legal=np.array([all(b[0]<=p[0] and b[1]<=p[1] and b[2]>=p[2] and b[3]>=p[3] for p in heads) for b in candidates]);area=(candidates[:,2]-candidates[:,0])*(candidates[:,3]-candidates[:,1]);maximum=area.max();selected=[]
   for mode in ['minimal','maximal']:
    eligible=area>=maximum-1e-6 if mode=='minimal' else area<=maximum*.810001
    def choose(mask):
     indices=np.flatnonzero(mask)
     if not len(indices):return None
     best=max(retention[j] for j in indices);indices=[j for j in indices if retention[j]>=best-.05];return int(max(indices,key=lambda j:float(scores[j])))
    raw=choose(eligible);guarded=choose(eligible&legal);counts['requests']+=1;counts['raw_cuts_detected_head']+=int(raw is not None and not legal[raw]);counts['no_detected_head_legal_candidate']+=int(guarded is None);selected.append({'mode':mode,'raw_candidate':raw,'detected_head_guarded_candidate':guarded,'guarded_action':'contain' if guarded is None else 'research_crop','face_coverage_certified':False})
   tasks.append({'ratio':ratio,'candidates':[{'box':b.tolist(),'score':float(sc),'retained_importance':float(re),'contains_detected_heads':bool(ok)} for b,sc,re,ok in zip(candidates,scores,retention,legal)],'selected':selected})
  row={'id':id,'family_id':source['family_id'],'use':source['use'],'width':w,'height':h,'heads':heads,'tasks':tasks,'new_training_labels_admitted':False}
  with (a.output/(id+'.json')).open('x') as f:json.dump(row,f);f.write('\n')
  counts['sources']+=1;print(json.dumps({'completed':counts['sources'],'id':id}),flush=True)
 counts.update({'model_sha256':sha(a.models/'focalnet/focalnet-human.onnx'),'source_cache_sha256':sha(a.auxiliary/'auxiliary.pt'),'policy':'five positions per free axis, scales1/.9/.8/.7/.65; retained-importance tolerance.05 before rank; minimal largest area, maximal scales<=.9. Detected-head containment is not coverage certification.','diagnostic_maps_saved':False,'source_preprocess':'same owned Lanczos inputs as first probe; upstream bilinear parity not claimed'});(a.output/'summary.json').write_text(json.dumps(counts,indent=2)+'\n');print(json.dumps(counts))
if __name__=='__main__':main()
