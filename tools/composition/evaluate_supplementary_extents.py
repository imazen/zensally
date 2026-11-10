#!/usr/bin/env python3
"""Evaluate fixed extent models on source-bound supplementary weak references."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from artifact_identity import sha
from composition_extent import ExtentHead,box_metrics
from composition_extent_image import ImageExtentModel
from crop_fit import TrainingModel
from local_rank_distillation import LocalCropRanker


def retain(path,value):
    with path.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n')


def descriptor(model,image,content,focal=False):
    if not focal:
        maps = model.encode(image.contiguous(memory_format=torch.channels_last))
        return torch.cat([F.adaptive_avg_pool2d(f,(4,4)).flatten(1) for f in maps],-1)
    image = image.contiguous()
    mean = image.new_tensor([.485,.456,.406])[None,:,None,None]
    std = image.new_tensor([.229,.224,.225])[None,:,None,None]
    normalized = (image-mean)/std
    prepared = torch.zeros_like(normalized)
    for i,valid in enumerate(content):
        x0,y0,x1,y1 = (valid*256).round().int().tolist()
        prepared[i,:,y0:y1,x0:x1] = normalized[i,:,y0:y1,x0:x1]
    fused = model.base.feature_pyramid(prepared)
    rank = model.rank_projection(fused)
    return torch.cat((F.adaptive_avg_pool2d(rank,(4,4)).flatten(1),
        F.adaptive_avg_pool2d(fused,(2,2)).flatten(1),fused.mean((-2,-1)),
        image.new_zeros(len(image),16)),-1)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['cohort','root','studies','models','auxiliary']:
        p.add_argument(name,type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha(args.cohort)!='a52144833a5824386577f63626e599f8dc602f8b77ee9d4d7f0c183574cebf70':
        raise ValueError('frozen supplementary source roles differ')
    if sha(args.root/'evaluation-plan.json')!='4a1b4a8d26fae709960d0cc240fcb9ce8a570ade50fe6154e8252d41f1e66ae9':
        raise ValueError('pre-reference selected model bindings differ')
    if sha(args.auxiliary)!='463e4884be5ada2128d08b9a755cf836d23e116059e6716390d64c0712b768a3':
        raise ValueError('original numerical-witness images differ')
    sources = json.loads(args.cohort.read_text())['sources']
    plan = json.loads((args.root/'evaluation-plan.json').read_text())
    # Bind the same fixed seed/aspect recipe before reading reference boxes or
    # producing predictions. This supplement is recorded after generation,
    # not misrepresented as part of the earlier pre-reference manifest.
    permutation = list(range(15))
    rng = torch.Generator().manual_seed(31)
    groups = []
    for low,high in [(0,.5),(.5,.9),(.9,1.1),(1.1,2.),(2.,float('inf'))]:
        indices = [i for i,s in enumerate(sources) if low<=s['native_size'][0]/s['native_size'][1]<high]
        order = torch.randperm(len(indices),generator=rng).tolist()
        for i,j in enumerate(order):
            permutation[indices[i]] = indices[j]
        groups.append(dict(low=low,high=high if math.isfinite(high) else None,indices=indices))
    retain(args.root/'permutation-plan.json',dict(seed=31,source_permutation=permutation,groups=groups,
        moved_sources=sum(i!=j for i,j in enumerate(permutation)),
        timing='same prior recipe, bound after reference generation and before reference parsing/model predictions'))
    summary = json.loads((args.root/'references/supplementary15-summary.json').read_text())
    calls = {r['id']:r for r in summary['calls']}
    if summary['completed']!=15 or len(calls)!=15:
        raise ValueError('all registered reference generations required')
    images,geometry,targets,present,receipts = [],[],[],[],[]
    for source in sources:
        if source['role']!='supplementary_dev' or source['fit_eligible']:
            raise ValueError('only supplementary development sources permitted')
        raw = Path(source['tensor_path'])
        if sha(raw)!=source['tensor_sha256'] or raw.stat().st_size!=256*256*3:
            raise ValueError('owned neural-input bytes differ')
        images.append(torch.from_numpy(np.frombuffer(raw.read_bytes(),np.uint8).copy().reshape(256,256,3)).permute(2,0,1).float()/255)
        width,height = source['native_size']
        valid = source['valid_rect']
        aspect = math.log(width/height)
        geometry.append([aspect,math.tanh(aspect),valid[2]-valid[0],valid[3]-valid[1]])
        call = calls[source['id']]
        path = args.root/'references/responses'/call['response']
        response = json.loads(path.read_text())
        contract = response['contract']
        digest = hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        input_path = args.root/'references/inputs'/(source['source']['provider_id']+'.rgb')
        if (digest!=path.stem or digest!=response['contract_sha256'] or contract['source_id']!=source['id']
                or contract['source_role']!='supplementary_dev' or contract['native_size']!=source['native_size']
                or contract['source_plan_sha256']!=sha(args.cohort) or sha(input_path)!=contract['owned_input_sha256']
                or contract['prompt']!='<CAPTION_TO_PHRASE_GROUNDING>'+source['query']):
            raise ValueError('cached reference/source contract differs')
        boxes = []
        reason = 'incomplete_generation' if call['invalid'] else None
        if reason is None:
            parsed = response['parsed']['<CAPTION_TO_PHRASE_GROUNDING>']
            if len(parsed['bboxes'])!=len(parsed['labels']):
                raise ValueError('reference labels/boxes differ')
            for box in parsed['bboxes']:
                if not (len(box)==4 and all(math.isfinite(v) for v in box)
                        and 0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):
                    reason = 'invalid_native_geometry'
                else:
                    boxes.append([box[0]/width,box[1]/height,box[2]/width,box[3]/height])
            if not boxes:
                reason = 'no_entities_unknown'
        available = reason is None
        targets.append([min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)] if available else [0,0,1,1])
        present.append(available)
        receipts.append(dict(id=source['id'],response_sha256=sha(path),valid_boxes=len(boxes),unknown_reason=reason))
    image = torch.stack(images).cuda()
    geom = torch.tensor(geometry,dtype=torch.float32).cuda()
    target = torch.tensor(targets,dtype=torch.float32).cuda()
    content = torch.tensor([s['valid_rect'] for s in sources]).cuda()
    eligible = [i for i,valid in enumerate(present) if valid]
    if not eligible:
        raise ValueError('no valid weak references; preserve generations without inventing grades')
    old_aux = torch.load(args.auxiliary,map_location='cpu',weights_only=False)
    root = args.studies/'florence-primary56/2026-11-09'
    cache_paths = dict(small=root/'extent-cache/features.pt',focal=root/'focalnet/extent-cache/features.pt',
        photo=args.studies/'florence-photo-transfer/2026-11-09/extent-cache/features.pt')
    expected = dict(small='f2d6987d8d2290ae51ad43b5bf46d4ea99cf3c3c156e0c384b24ca2c5ba0944f',
        focal='633dbd8eb07b3775d97bafe8760b99e9224b1e3b31d5dc2122f61565c09dc07d',
        photo='d40114c2ffe057bced81413de58b042e3a036e7fe8cd7a5af9eca386878702a5')
    caches = {}
    for key,path in cache_paths.items():
        if sha(path)!=expected[key]:
            raise ValueError('frozen descriptor witness differs')
        caches[key] = torch.load(path,map_location='cpu',weights_only=False)
    initial = args.studies/'florence-finetune/2026-11-09/initializers/epoch11.pt'
    photographic = args.studies/'florence-photo-transfer/2026-11-09/model/photographic-student.pt'
    focal_path = args.models/'focalnet/focalnet-human.pt'
    for path,digest in [(initial,'910219ef1af5eecf4530b675fecddba04798b62e323def2b781de8fda5354f71'),
        (photographic,'c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df'),
        (focal_path,'82c4d695310c220e5f3bcfbf5e64d315a5e826cfd77e945e61e032460014480a')]:
        if sha(path)!=digest:
            raise ValueError('frozen encoder differs')
    small = TrainingModel().eval().cuda()
    small.load_state_dict(torch.load(initial,map_location='cpu',weights_only=False)['model'])
    photo = LocalCropRanker().eval().cuda()
    photo.load_state_dict(torch.load(photographic,map_location='cpu',weights_only=False)['model'])
    sys.path.insert(0,str(args.models/'focalnet-source/src'))
    from focalnet.ranking import load_human_checkpoint
    focal,_ = load_human_checkpoint(focal_path)
    focal = focal.eval().cuda()
    old_image = old_aux['image'].cuda()
    old_content = old_aux['packed']['valid_rect'].cuda()
    old_geom = caches['small']['geometry'].cuda()
    new_vectors,old_vectors = {},{}
    for key,model in [('small',small.ranker),('photo',photo.ranker),('focal',focal)]:
        new_vectors[key] = descriptor(model,image,content,focal=key=='focal')
        old_vectors[key] = descriptor(model,old_image[:16],old_content[:16],focal=key=='focal')
        if new_vectors[key].shape!=(15,512) or not bool(torch.isfinite(new_vectors[key]).all()):
            raise ValueError('invalid new descriptor')
    results,qualification = [],[]
    for binding in plan['models']:
        path = Path(binding['checkpoint'])
        if sha(path)!=binding['checkpoint_sha256']:
            raise ValueError('checkpoint changed after pre-reference model freeze')
        state = torch.load(path,map_location='cpu',weights_only=False)
        order = permutation if binding['arm']=='permuted' else list(range(15))
        if binding['study'] in ['small-trained','small-augmented']:
            ranker = copy.deepcopy(small.ranker)
            ranker.scorer = nn.Identity()
            mean,std = state['model']['visual_mean'],state['model']['visual_std']
            model = ImageExtentModel(ranker,ExtentHead(),mean,std).cuda().eval()
            model.load_state_dict(state['model'])
            box = model(image[order].contiguous(memory_format=torch.channels_last),geom)
            dev = [i for i,s in enumerate(caches['small']['sources']) if s['use']=='dev' and s['primary_extent_present']]
            old_order = state['protocol']['source_permutation'] if binding['arm']=='permuted' else list(range(56))
            replay = model(old_image[[old_order[i] for i in dev]].contiguous(memory_format=torch.channels_last),old_geom[dev])
            saved = json.loads((path.parent/'best-predictions.json').read_text())
            difference = float((replay-torch.tensor([s['box'] for s in saved],device='cuda')).abs().max())*128
            method = 'all eight original selected development predictions'
        else:
            key = 'focal' if binding['study']=='focal-mse' else 'photo' if binding['study']=='photo-mse' else 'small'
            mean,std = state['normalization_mean'].cuda(),state['normalization_std'].cuda()
            def features(vector,geometry,zero=False):
                normalized = ((vector-mean)/std).clamp(-10,10)
                return torch.cat((torch.zeros_like(normalized) if zero else normalized,geometry),-1)
            model = ExtentHead().cuda().eval()
            model.load_state_dict(state['model'])
            box = model(features(new_vectors[key][order],geom,binding['arm']=='aspect'))
            native = model(features(old_vectors[key],old_geom[:16],binding['arm']=='aspect'))
            cached = model(features(caches[key]['visual'][:16].cuda(),old_geom[:16],binding['arm']=='aspect'))
            difference = float((native-cached).abs().max())*128
            method = 'first sixteen prior frozen descriptors propagated through this head'
        if difference>3:
            raise ValueError('descriptor/native qualification exceeds unchanged map-edge gate')
        qualification.append(dict(study=binding['study'],arm=binding['arm'],seed=binding['seed'],method=method,
                                  max_map_edge_difference=difference))
        if not bool(torch.isfinite(box).all() and (box[:,:2]>=0).all() and (box[:,2:]<=1).all() and (box[:,2:]>box[:,:2]).all()):
            raise ValueError('invalid supplementary extent')
        metrics,iou = box_metrics(box[eligible],target[eligible])
        predictions = [dict(id=sources[i]['id'],family_id=sources[i]['family_id'],role='supplementary_dev',
            input_source=sources[order[i]]['id'],box=box[i].cpu().tolist(),weak_target=target[i].cpu().tolist(),
            iou=float(iou[j]),human_rated=False,face_coverage_certified=False) for j,i in enumerate(eligible)]
        results.append(dict(binding,metrics=metrics,predictions=predictions))
        del model
    # Geometry controls have identical trained states in the three MSE studies.
    for seed in [31,137,911]:
        rows = [r['predictions'] for r in results if r['study'] in ['small-mse','focal-mse','photo-mse']
                and r['arm']=='aspect' and r['seed']==seed]
        if not (rows[0]==rows[1]==rows[2]):
            raise ValueError('matched MSE geometry controls differ on supplementary sources')
    fit = [i for i,s in enumerate(caches['small']['sources']) if s['primary_extent_training_eligible']]
    mean_box = caches['small']['targets'][fit].mean(0).cuda().expand(len(eligible),4)
    metrics,_ = box_metrics(mean_box,target[eligible])
    cache = dict(image=image.cpu(),geometry=geom.cpu(),targets=target.cpu(),present=torch.tensor(present),
        descriptors={k:v.cpu() for k,v in new_vectors.items()},sources=sources,reference_receipts=receipts,
        training_eligible=False,diagnostic_maps_saved=False)
    torch.save(cache,args.root/'evaluation-inputs.pt')
    retain(args.root/'qualification.json',dict(models=len(qualification),checks=qualification,
        max_map_edge_difference=max(r['max_map_edge_difference'] for r in qualification),matched_mse_geometry_exact=True))
    retain(args.root/'evaluation.json',dict(models=results,reference_receipts=receipts,qualified_models=48,
        evaluated_sources=len(eligible),unknown_sources=15-len(eligible),fit_mean_control=metrics,
        selected_models_changed=False,new_gradients=False,paid_model_api_calls=0,
        face_coverage_certified=False,cuda_peak_allocated=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved=torch.cuda.max_memory_reserved()))
    print(json.dumps(dict(models=len(results),evaluated_sources=len(eligible),unknown_sources=15-len(eligible),
        max_qualification_map_edge_difference=max(r['max_map_edge_difference'] for r in qualification),
        cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())))


if __name__ == '__main__':
    main()
