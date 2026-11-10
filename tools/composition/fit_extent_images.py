#!/usr/bin/env python3
"""Registered end-to-end primary-extent learning under fixed source roles."""
import argparse,copy,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import torch
from torch import nn
from crop_fit import TrainingModel
from composition_extent import ExtentHead,box_metrics,extent_objective
from composition_extent_image import ImageExtentModel
from extent_augmentation import augment
from fit_heads import sha
from fit_extents import retain,alias

@torch.no_grad()
def evaluate(model,image,geometry,target,indices,visual_indices,sources):
 box=model(image[visual_indices],geometry[indices]);metrics,iou=box_metrics(box,target[indices]);metrics['objective']=float(extent_objective(box,target[indices]));rows=[]
 if not bool(torch.isfinite(box).all()):raise ValueError('nonfinite image-extent evaluation')
 for j,i in enumerate(indices):rows.append({'id':sources[i]['id'],'family_id':sources[i]['family_id'],'box':box[j].cpu().tolist(),'weak_target':target[i].cpu().tolist(),'iou':float(iou[j]),'edge_mae':float((box[j]-target[i]).abs().mean()),'face_coverage_certified':False})
 return metrics,rows

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('auxiliary',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('--augment',choices=['none','flip-color'],default='none');a=p.parse_args();out=a.root/('extent-joint-augmented-fits' if a.augment!='none' else 'extent-joint-fits');out.mkdir(exist_ok=True);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;cache_path=a.root/'extent-cache/features.pt';cache=torch.load(cache_path,map_location='cpu',weights_only=False);aux_path=a.auxiliary/'auxiliary.pt';aux=torch.load(aux_path,map_location='cpu',weights_only=False);sources=cache['sources']
 if sha(aux_path)!=cache['auxiliary_source_cache_sha256'] or sha(a.checkpoint)!=cache['encoder_sha256'] or [s['id'] for s in aux['sources']]!=[s['id'] for s in sources]:raise ValueError('registered image/teacher/encoder identity differs')
 fit=[i for i,s in enumerate(sources) if s['primary_extent_training_eligible']];dev=[i for i,s in enumerate(sources) if s['use']=='dev' and s['primary_extent_present']];permutation=json.loads((a.root/'extent-fits/protocol.json').read_text())['source_permutation'];mean=cache['visual'][fit].mean(0);std=cache['visual'][fit].std(0,unbiased=False).clamp_min(.001);geometry=cache['geometry'].cuda();target=cache['targets'].cuda();image=aux['image'].cuda().contiguous(memory_format=torch.channels_last);cached_normalized=((cache['visual']-mean)/std).clamp(-10,10).cuda();base=TrainingModel();base.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model']);base.ranker.scorer=nn.Identity();base.cuda().eval()
 protocol={'cache_sha256':sha(cache_path),'auxiliary_sha256':sha(aux_path),'checkpoint_sha256':sha(a.checkpoint),'arms':['image','permuted'],'seeds':[31,137,911],'epochs':200,'batch':16,'head_lr':.001,'encoder_lr':.0001,'weight_decay':.01,'clip':5.,'precision':'FP32 TF32 off','batchnorm':'frozen running statistics, trainable affine','loss':'5*edge L1 + 2*generalized IoU','selection':'minimum weak TRAIN-development objective including epoch zero','source_permutation':permutation,'canonical_holdouts_opened':False,'face_coverage_certified':False}
 if a.augment!='none':protocol['augmentation']={'name':'content-local horizontal flip plus brightness/contrast','flip_probability':.5,'brightness':[.8,1.2],'contrast':[.8,1.2],'contrast_center':.5,'padding':'unchanged zero','development_augmented':False}
 valid_content=aux['packed']['valid_rect'].cuda() if a.augment!='none' else None
 if (out/'protocol.json').exists():
  if json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('resume changes image-extent protocol')
 else:retain(out/'protocol.json',protocol)
 reports=[]
 for arm in protocol['arms']:
  for seed in protocol['seeds']:
   directory=out/(arm+'-seed'+str(seed));directory.mkdir(exist_ok=True)
   if (directory/'summary.json').exists():reports.append(json.loads((directory/'summary.json').read_text()));continue
   torch.manual_seed(seed);head=ExtentHead();model=ImageExtentModel(copy.deepcopy(base.ranker),head,mean,std).cuda().eval();optimizer=torch.optim.AdamW([{'params':model.ranker.parameters(),'lr':.0001},{'params':model.head.parameters(),'lr':.001}],weight_decay=.01,fused=True);history=[];best=float('inf');best_epoch=None;start=0
   visual_indices=permutation if arm=='permuted' else list(range(56))
   if (directory/'latest.pt').exists():
    state=torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
    if state['protocol']!=protocol:raise ValueError('checkpoint protocol differs')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);history=state['history'];best=state['best'];best_epoch=state['best_epoch'];start=state['epoch']+1;torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
   else:
    with torch.no_grad():
     native=torch.cat([model(image[visual_indices[i:i+16]],geometry[i:i+16]) for i in range(0,56,16)]);cached=model.head(torch.cat((cached_normalized[visual_indices],geometry),-1));difference=float((native-cached).abs().max())
     if difference>3/128:raise ValueError('initial cache/native coordinate difference exceeds registered map-edge budget')
     retain(directory/'initial-numerics.json',{'max_normalized_coordinate_difference':difference,'map_pixel_difference':difference*128,'bit_parity_claimed':False})
   for epoch in range(start,201):
    model.eval()
    if epoch:
     order=torch.randperm(len(fit),generator=torch.Generator().manual_seed(seed+epoch-1)).tolist()
     for offset in range(0,len(fit),16):
      indices=[fit[j] for j in order[offset:offset+16]];vi=[visual_indices[i] for i in indices];optimizer.zero_grad(set_to_none=True);pixels,expected=image[vi],target[indices]
      if a.augment!='none':pixels,expected=augment(pixels,expected,valid_content[vi])
      prediction=model(pixels,geometry[indices]);value=extent_objective(prediction,expected)
      if not bool(torch.isfinite(value)):raise ValueError('nonfinite image-extent fitting loss')
      value.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    metrics,rows=evaluate(model,image,geometry,target,dev,[visual_indices[i] for i in dev],sources);history.append({'epoch':epoch,'dev':metrics});improved=metrics['objective']<best
    if improved:best=metrics['objective'];best_epoch=epoch
    if improved or epoch%25==0:
     state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'arm':arm,'seed':seed,'protocol':protocol,'history':history,'best':best,'best_epoch':best_epoch,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()};checkpoint=directory/f'epoch-{epoch:03}.pt'
     with checkpoint.open('xb') as f:torch.save(state,f)
     alias(directory/'latest.pt',checkpoint);prediction_path=directory/f'predictions-epoch-{epoch:03}.json';retain(prediction_path,rows)
     if improved:alias(directory/'best.pt',checkpoint);alias(directory/'best-predictions.json',prediction_path)
    (out/'progress.json').write_text(json.dumps({'pid':os.getpid(),'arm':arm,'seed':seed,'epoch':epoch,'required_epochs':200,'completed_runs':len(reports),'required_runs':6,'checked_unix':time.time()}))
    if epoch%50==0:print(json.dumps({'arm':arm,'seed':seed,'epoch':epoch,'dev':metrics,'best_epoch':best_epoch}),flush=True)
   report={'arm':arm,'seed':seed,'best_epoch':best_epoch,'best_dev':history[best_epoch]['dev'],'parameters':sum(p.numel() for p in model.parameters())};retain(directory/'history.json',history);retain(directory/'summary.json',report);reports.append(report)
 retain(out/'summary.json',reports);(out/'progress.json').write_text(json.dumps({'pid':None,'stage':'complete','completed_runs':len(reports),'required_runs':6,'checked_unix':time.time()}));print(json.dumps({'completed_runs':len(reports),'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_peak_reserved':torch.cuda.max_memory_reserved()}))
if __name__=='__main__':main()
