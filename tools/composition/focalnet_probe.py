#!/usr/bin/env python3
"""Score exact existing crop banks locally; no importance maps are persisted."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import numpy as np
import onnxruntime as ort
import torch
from crop_fit import CropDataset
from fit_heads import sha

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('models',type=Path);p.add_argument('first_gate',type=Path);p.add_argument('auxiliary',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(2)
 options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL;session=ort.InferenceSession(str(a.models/'focalnet/focalnet-human.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
 sys.path.insert(0,str(a.models/'focalnet-source/src'));from focalnet.ranking import load_human_checkpoint
 reference,_=load_human_checkpoint(a.models/'focalnet/focalnet-human.pt');reference.eval();mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None];rows=[];errors=[]
 manifest=Path(json.loads((a.first_gate/'inputs.json').read_text())['admitted_path']);old_cache=torch.load(a.first_gate/'feature-cache/features.pt',map_location='cpu',weights_only=False)
 if sha(manifest)!=old_cache['manifest_sha256']:raise ValueError('admitted crop manifest differs')
 samples=[]
 for use in ['fit','dev']:
  ds=CropDataset(manifest,use,training=use=='fit')
  for i,s in enumerate(ds.sources):samples.append((s['id'],s['family_id'],use,ds[i],len(s['candidates'])))
 aux=torch.load(a.auxiliary/'auxiliary.pt',map_location='cpu',weights_only=False)
 for i,s in enumerate(aux['sources']):samples.append((s['id'],s['family_id'],'aux_'+s['use'],{'image':aux['image'][i],**{k:v[i] for k,v in aux['packed'].items()}},110))
 for id,family,use,batch,count in samples:
  image=(batch['image']-mean)/std;x0,y0,x1,y1=(batch['valid_rect']*256).round().int().tolist();normalized=torch.zeros_like(image);normalized[:,y0:y1,x0:x1]=image[:,y0:y1,x0:x1];inputs={'image':normalized[None].numpy(),'boxes':batch['boxes'][None].numpy(),'content':batch['valid_rect'][None].numpy()};scores=session.run(['crop_scores'],inputs)[0][0]
  if not np.isfinite(scores).all():raise ValueError('nonfinite rank score')
  if len(errors)<16:
   with torch.inference_mode():_,expected=reference(torch.from_numpy(inputs['image']),torch.from_numpy(inputs['boxes']),torch.from_numpy(inputs['content']))
   expected=expected[0].numpy();np.testing.assert_allclose(scores,expected,atol=.05,rtol=.001);errors.append({'id':id,'max_abs_error':float(np.max(np.abs(scores-expected))),'argmax_changed':int(scores[:count].argmax())!=int(expected[:count].argmax())})
  row={'id':id,'family_id':family,'use':use,'scores':scores[:count].tolist(),'candidate_boxes':batch['boxes'][:count].tolist()};rows.append(row)
  with (a.output/(id+'.json')).open('x') as f:json.dump(row,f);f.write('\n')
  print(json.dumps({'completed':len(rows),'id':id,'role':use}),flush=True)
 result={'sources':len(rows),'model_sha256':sha(a.models/'focalnet/focalnet-human.onnx'),'checkpoint_sha256':sha(a.models/'focalnet/focalnet-human.pt'),'input':'previous owned zenresize Lanczos letterbox RGB, ImageNet normalization on valid content, zero normalized padding; differs from upstream bilinear preprocessing','session':'CPUExecutionProvider ORT_DISABLE_ALL, one thread','numeric_reference':errors,'max_abs_error':max(e['max_abs_error'] for e in errors),'new_composition_labels_admitted':False,'diagnostic_maps_saved':False,'canonical_holdouts_opened':False};(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 # Compare within the exact old task bank, without inventing acceptance thresholds.
 lookup={r['id']:r for r in rows};counts={}
 for use in ['fit','dev']:
  tasks=judged=accepted=0;essential={'tasks':0,'judged':0,'accepted':0};different=0
  for i,s in enumerate(old_cache['sources']):
   if old_cache['uses'][i]!=use:continue
   groups={}
   for ci,c in enumerate(s['candidates']):groups.setdefault(c['task_id'],[]).append(ci)
   for group in groups.values():
    chosen=max(group,key=lambda ci:lookup[s['id']]['scores'][ci]);c=s['candidates'][chosen];tasks+=1;judged+=int(c['weight']>0);accepted+=int(c['weight']>0 and c['acceptable']);different+=int(chosen!=max(group,key=lambda ci:float(old_cache['initial_logits'][i,ci,0])))
    if 1/3<=c['aspect']<=3:
     essential['tasks']+=1;essential['judged']+=int(c['weight']>0);essential['accepted']+=int(c['weight']>0 and c['acceptable'])
  counts[use]={'tasks':tasks,'rank_judged':judged,'rank_accepted':accepted,'essential':essential,'rank_choices_changed_from_frozen_student':different}
 (a.output/'exact-bank-grades.json').write_text(json.dumps(counts,indent=2)+'\n');print(json.dumps({'sources':len(rows),'max_abs_error':result['max_abs_error'],'counts':counts}))
if __name__=='__main__':main()
