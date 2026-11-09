#!/usr/bin/env python3
"""Measure fit/development behavior of completed frozen and encoder-trained heads."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import torch
from torch import nn

from artifact_identity import sha
from crop_fit import TrainingModel
from composition_extent import ExtentHead,box_metrics,extent_objective
from composition_extent_image import ImageExtentModel


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('auxiliary',type=Path)
    p.add_argument('--augmented',action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = args.root/'extent-cache/features.pt'
    if sha(path) != 'f2d6987d8d2290ae51ad43b5bf46d4ea99cf3c3c156e0c384b24ca2c5ba0944f':
        raise ValueError('registered targets differ')
    cache = torch.load(path,map_location='cpu',weights_only=False)
    if sha(args.auxiliary) != cache['auxiliary_source_cache_sha256']:
        raise ValueError('registered source tensors differ')
    auxiliary = torch.load(args.auxiliary,map_location='cpu',weights_only=False)
    sources = cache['sources']
    if [s['id'] for s in sources] != [s['id'] for s in auxiliary['sources']]:
        raise ValueError('source order differs')
    roles = {'fit':[i for i,s in enumerate(sources) if s['primary_extent_training_eligible']],
             'dev':[i for i,s in enumerate(sources) if s['use']=='dev' and s['primary_extent_present']]}
    if len(roles['fit']) != 48 or len(roles['dev']) != 8:
        raise ValueError('registered fit/development counts differ')
    visual = cache['visual']
    mean = visual[roles['fit']].mean(0)
    std = visual[roles['fit']].std(0,unbiased=False).clamp_min(.001)
    normalized = ((visual-mean)/std).clamp(-10,10).cuda()
    geometry,target,image = cache['geometry'].cuda(),cache['targets'].cuda(),auxiliary['image'].cuda().contiguous(memory_format=torch.channels_last)
    base = TrainingModel().ranker
    base.scorer = nn.Identity()
    reports = []
    studies = [('trained','extent-joint-augmented-fits',6)] if args.augmented else [('frozen','extent-giou-fits',9),('trained','extent-joint-fits',6)]
    for kind,folder,required in studies:
        directory = args.root/folder
        runs = json.loads((directory/'summary.json').read_text())
        if len(runs) != required:
            raise ValueError('all registered runs required')
        for run in runs:
            run_dir = directory/(run['arm']+'-seed'+str(run['seed']))
            for phase,file in [('selected','best.pt'),('final','epoch-200.pt')]:
                checkpoint = run_dir/file
                state = torch.load(checkpoint,map_location='cpu',weights_only=False)
                permutation = state['protocol']['source_permutation'] if run['arm']=='permuted' else list(range(56))
                if kind == 'frozen':
                    model = ExtentHead().cuda().eval()
                    features = torch.cat((torch.zeros_like(normalized) if run['arm']=='aspect' else normalized[permutation],geometry),-1)
                else:
                    model = ImageExtentModel(copy.deepcopy(base),ExtentHead(),mean,std).cuda().eval()
                model.load_state_dict(state['model'])
                result = dict(kind=kind,arm=run['arm'],seed=run['seed'],phase=phase,
                    epoch=state['epoch'],checkpoint_sha256=sha(checkpoint),roles={})
                for role,indices in roles.items():
                    predicted = []
                    for offset in range(0,len(indices),16):
                        batch = indices[offset:offset+16]
                        box = (model(features[batch]) if kind=='frozen' else
                               model(image[[permutation[i] for i in batch]],geometry[batch]))
                        predicted.append(box)
                    box = torch.cat(predicted)
                    if not bool(torch.isfinite(box).all() and (box[:,:2]>=0).all()
                                and (box[:,2:]<=1).all() and (box[:,2:]>box[:,:2]).all()):
                        raise ValueError('invalid native extent')
                    metrics,iou = box_metrics(box,target[indices])
                    metrics['objective'] = float(extent_objective(box,target[indices]))
                    rows = [dict(id=sources[i]['id'],family_id=sources[i]['family_id'],
                                 box=box[j].cpu().tolist(),iou=float(iou[j])) for j,i in enumerate(indices)]
                    result['roles'][role] = dict(metrics=metrics,sources=len(rows),predictions=rows)
                    if role=='dev' and phase=='selected':
                        saved = json.loads((run_dir/'best-predictions.json').read_text())
                        if [s['id'] for s in saved] != [s['id'] for s in rows]:
                            raise ValueError('selected prediction identities differ')
                        difference = max(abs(x-y) for a,b in zip(rows,saved) for x,y in zip(a['box'],b['box']))
                        result['selected_replay_max_map_edge_difference'] = difference*128
                        if difference*128 > 3:
                            raise ValueError('selected replay exceeds original numerical gate')
                reports.append(result)
                del model
    result = dict(runs=reports,cache_sha256=sha(path),source_roles_changed=False,
        new_gradients=False,new_test_inference=False,paid_model_api_calls=0,
        cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())
    with (args.root/'generalization.json').open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(audited_checkpoints=len(reports),
        max_selected_replay_map_edge_difference=max(r.get('selected_replay_max_map_edge_difference',0) for r in reports),
        cuda_peak_allocated=result['cuda_peak_allocated'],cuda_peak_reserved=result['cuda_peak_reserved'])))


if __name__ == '__main__':
    main()
