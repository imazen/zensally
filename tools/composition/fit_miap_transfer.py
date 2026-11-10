#!/usr/bin/env python3
"""Bounded multitask person-relationship transfer with unchanged crop evaluation."""
import argparse
import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from artifact_identity import sha
from fit_extents import retain
from fit_unsplash_ranks import alias, evaluate
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from composition_extent import box_metrics
from local_rank_distillation import LocalCropRanker, family_weights, objective
from person_relationships import PersonHead, auxiliary_loss, freeze_bn

ARMS=['rank-only','joint-image','joint-permuted','joint-geometry']
SEEDS=[31,137,911]
EPOCHS=44


def permutation(sources):
    donors=list(range(len(sources)));groups={}
    for i,s in enumerate(sources):
        aspect=s['native_size'][0]/s['native_size'][1]
        key=(s['role'],sum(aspect>=edge for edge in [.5,.9,1.1,2.]))
        groups.setdefault(key,[]).append(i)
    rng=torch.Generator().manual_seed(31)
    for indices in groups.values():
        order=[indices[i] for i in torch.randperm(len(indices),generator=rng).tolist()]
        for i,j in zip(order,order[1:]+order[:1]):donors[i]=j
    return donors


def image_batch(data, indices):
    return data['image'][indices].permute(0,3,1,2).float()/255


@torch.no_grad()
def auxiliary_evaluate(model,head,data,indices,donors,geometry):
    model.eval();head.eval();outputs=[]
    for offset in range(0,len(indices),16):
        batch=indices[offset:offset+16]
        features=None if geometry else model.feature_maps(image_batch(data,donors[batch]))
        outputs.append(head(features,data['content'][batch],data['source_aspect'][batch],geometry=geometry))
    prediction=torch.cat(outputs);target=data['target'][indices];valid=data['valid'][indices]
    if not bool(torch.isfinite(prediction).all()) or not bool((prediction[:,2:4]>prediction[:,:2]).all()):
        raise ValueError('invalid auxiliary development geometry')
    metrics,_=box_metrics(prediction[:,:4],target[:,:4])
    metrics['objective']=float(auxiliary_loss(prediction,target,valid).mean())
    metrics['statistics_mae']=float((((prediction-target).abs()*valid).sum(-1)/valid.sum(-1)).mean())
    return metrics,prediction.cpu().tolist()


def checkpoint(model,head,optimizer,arm,seed,epoch,protocol,history,visits):
    return dict(model=model.state_dict(),person_head=head.state_dict(),optimizer=optimizer.state_dict(),
        arm=arm,seed=seed,epoch=epoch,protocol=protocol,history=history,visits=visits,
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['rank_cache','auxiliary','weights','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    for path,digest in [(a.rank_cache,'2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'),
                        (a.weights,'c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df'),
                        (a.auxiliary,'b986f2740585449a390177e199fda71eefd1bb4034c49fce50e4d667fbdb4bc4')]:
        if sha(path)!=digest:raise ValueError('frozen photographic data/initializer differs')
    if sha(a.auxiliary)!=json.loads((a.auxiliary.parent/'summary.json').read_text())['cache_sha256']:
        raise ValueError('auxiliary cache differs')
    quality_sources=['training/local_rank_distillation.py','training/crop_regions.py',
        'training/crop_ranker.py','training/model_v3ds.py','tools/composition/fit_unsplash_ranks.py']
    for name in quality_sources:
        if Path(name).read_bytes()!=subprocess.check_output(['git','show','9f12692417d774653ed676d236e0f89eaafa7385:'+name]):
            raise ValueError('original quality source changed: '+name)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.35);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rank=torch.load(a.rank_cache,map_location='cpu',weights_only=False)
    auxiliary=torch.load(a.auxiliary,map_location='cpu',weights_only=False)
    if auxiliary['protocol']['acquisition_packet_sha256']!='93f63d69180d82f197e34a733005118de260eda728e18998ced3194959581bb1':
        raise ValueError('auxiliary source packet differs')
    sources=rank['sources'];aux_sources=auxiliary['sources']
    for s in sources:s['ratios']=rank['ratios']
    fit=[i for i,s in enumerate(sources) if s['role']=='fit'];dev=[i for i,s in enumerate(sources) if s['role']=='dev']
    aux_fit=[i for i,s in enumerate(aux_sources) if s['role']=='fit'];aux_dev=[i for i,s in enumerate(aux_sources) if s['role']=='dev']
    if any(s['training_eligible']!=(s['role']=='fit') or s['face_coverage_certified'] for s in aux_sources):
        raise ValueError('auxiliary target admission differs')
    if (len(sources),len(fit),len(dev),len(aux_fit),len(aux_dev))!=(119,94,25,275,91):raise ValueError('fixed fit/development counts required')
    if any(s['role'] not in {'fit','dev'} for s in sources+aux_sources):raise ValueError('TEST cannot enter transfer')
    if {s['family_id'] for s in sources+aux_sources if s['role']=='fit'} & {s['family_id'] for s in sources+aux_sources if s['role']=='dev'}:
        raise ValueError('source component crosses fitting/development')
    phase=torch.tensor([int.from_bytes(hashlib.sha256(('query-phase-v1\0'+s['id']).encode()).digest()[:8],'big')%22 for s in sources])
    permuted=permutation(aux_sources)
    protocol=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        rank_cache_sha256=sha(a.rank_cache),auxiliary_cache_sha256=sha(a.auxiliary),initializer_sha256=sha(a.weights),
        quality_source_sha256={name:sha(Path(name)) for name in quality_sources},
        arms=ARMS,seeds=SEEDS,epochs=EPOCHS,batch=16,lr_encoder=.0001,lr_heads=.001,weight_decay=.01,clip=5.,
        precision='FP32 TF32 off',batchnorm='running statistics frozen; affine trainable',
        rank_loss='original family-balanced logistic+0.25 masked MSE',auxiliary_loss='0.5*(masked statistics L1+0.2 union GIoU)',
        selection='minimum essential-ratio family-mean teacher regret, then pair agreement, then earliest epoch',
        query_phases=phase.tolist(),auxiliary_donors=permuted,
        unmoved_auxiliary={role:sum(i==permuted[i] for i,s in enumerate(aux_sources) if s['role']==role) for role in ['fit','dev']},
        new_test_inference=False,face_coverage_certified=False,artistic_labels=False,paid_model_api_calls=0)
    a.output.mkdir(parents=True,exist_ok=False);retain(a.output/'protocol.json',protocol)
    progress=a.output/'progress.json';status=dict(pid=os.getpid(),stage='loading-resident-inputs',completed_fits=0,required_fits=12)
    def update(**fields):
        status.update(fields,checked_unix=time.time());progress.write_text(json.dumps(status)+'\n')
    def failed():
        if status['pid'] is not None:update(pid=None,stage='failed-artifacts-retained')
    atexit.register(failed);update()
    data={key:rank[key].cuda() for key in ['image','content','boxes','targets','pairs','pair_valid','eligible','query_valid']}
    data['source_aspect']=torch.tensor([s['native_size'][0]/s['native_size'][1] for s in sources],device='cuda')
    data['aspect']=torch.tensor([int(r.split(':')[0])/int(r.split(':')[1]) for r in rank['ratios'] for _ in range(2)],device='cuda')
    data['mode']=torch.tensor([i%2 for i in range(22)],dtype=torch.float32,device='cuda')
    aux={key:auxiliary[key].cuda() for key in ['image','content','target','valid']}
    aux['source_aspect']=torch.tensor([s['native_size'][0]/s['native_size'][1] for s in aux_sources],device='cuda')
    true_rank_donors=torch.arange(119,device='cuda');weight=torch.zeros(119,device='cuda')
    weight[fit]=family_weights(sources,fit).cuda()*len(fit)*22/data['query_valid'][fit].sum(-1)
    initial=torch.load(a.weights,map_location='cpu',weights_only=False)['model'];reports=[]
    for arm in ARMS:
        aux_donors=torch.tensor(permuted if arm=='joint-permuted' else list(range(366)),device='cuda')
        for seed in SEEDS:
            directory=a.output/(arm+'-seed'+str(seed));directory.mkdir()
            torch.manual_seed(seed);model=LocalCropRanker().cuda();model.load_state_dict(initial);head=PersonHead().cuda()
            encoder=[p for name,p in model.named_parameters() if not name.startswith('ranker.scorer.')]
            optimizer=torch.optim.AdamW([dict(params=encoder,lr=.0001),dict(params=model.ranker.scorer.parameters(),lr=.001),dict(params=head.parameters(),lr=.001)],weight_decay=.01,fused=True)
            visits=torch.zeros(119,22,dtype=torch.int32);history=[];best=None;best_path=None
            for epoch in range(EPOCHS+1):
                update(stage='fitting',arm=arm,seed=seed,epoch=epoch,completed_fits=len(reports))
                if epoch:
                    model.train_arm();head.train();freeze_bn(model)
                    order=torch.randperm(len(fit),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
                    for offset in range(0,len(fit),16):
                        raw_indices=torch.tensor([fit[j] for j in order[offset:offset+16]])
                        query=(phase[raw_indices]+epoch-1)%22;visits[raw_indices,query]+=1
                        known=rank['query_valid'][raw_indices,query];indices=raw_indices[known].cuda();q=query[known].cuda()
                        if not len(indices):continue
                        optimizer.zero_grad(set_to_none=True)
                        features=model.feature_maps(data['image'][indices])
                        scores=model.scores(features,data['boxes'][indices,q//2],data['source_aspect'][indices],data['aspect'][q],data['mode'][q],data['content'][indices])
                        value=(objective(scores,data['targets'][indices,q],data['pairs'][indices,q],data['pair_valid'][indices,q],data['eligible'][indices,q],data['query_valid'][indices,q])*weight[indices]).sum()/16
                        if not bool(torch.isfinite(value)):raise ValueError('nonfinite crop objective')
                        value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
                        if any(p.grad is not None for p in head.parameters()):raise ValueError('rank task activated auxiliary head')
                    if arm!='rank-only':
                        order=torch.randperm(len(aux_fit),generator=torch.Generator().manual_seed(seed+epoch+10000)).tolist()
                        for offset in range(0,len(aux_fit),16):
                            indices=torch.tensor([aux_fit[j] for j in order[offset:offset+16]],device='cuda')
                            optimizer.zero_grad(set_to_none=True)
                            features=None if arm=='joint-geometry' else model.feature_maps(image_batch(aux,aux_donors[indices]))
                            predicted=head(features,aux['content'][indices],aux['source_aspect'][indices],geometry=arm=='joint-geometry')
                            value=.5*auxiliary_loss(predicted,aux['target'][indices],aux['valid'][indices]).mean()
                            if not bool(torch.isfinite(value)):raise ValueError('nonfinite auxiliary objective')
                            value.backward();torch.nn.utils.clip_grad_norm_([*model.parameters(),*head.parameters()],5.,error_if_nonfinite=True);optimizer.step()
                            if any(p.grad is not None for p in model.ranker.scorer.parameters()):raise ValueError('auxiliary task activated crop head')
                metrics,predictions=evaluate(model,data,dev,true_rank_donors,sources)
                aux_metrics,aux_predictions=auxiliary_evaluate(model,head,aux,aux_dev,aux_donors,arm=='joint-geometry')
                history.append(dict(epoch=epoch,development=metrics,person_development=aux_metrics))
                key=(metrics['essential']['teacher_regret'],-metrics['essential']['strong_pair_accuracy'],epoch)
                if best is None or key<best:
                    best=key;best_path=directory/f'weights-{epoch:03}.pt'
                    torch.save(dict(model=model.state_dict(),person_head=head.state_dict(),arm=arm,seed=seed,epoch=epoch,protocol=protocol,metrics=metrics,person_development=aux_metrics),best_path)
                    retain(directory/f'predictions-{epoch:03}.json',predictions)
                    retain(directory/f'person-predictions-{epoch:03}.json',dict(source_ids=[aux_sources[i]['id'] for i in aux_dev],prediction=aux_predictions))
                    alias(directory/'best.pt',best_path);alias(directory/'best-predictions.json',directory/f'predictions-{epoch:03}.json')
                if epoch in [0,22,44]:
                    path=directory/f'checkpoint-{epoch:03}.pt';torch.save(checkpoint(model,head,optimizer,arm,seed,epoch,protocol,history,visits),path)
                    alias(directory/'latest.pt',path)
                if epoch%11==0:
                    print(json.dumps(dict(arm=arm,seed=seed,epoch=epoch,best_epoch=best[-1],development=metrics['essential'],person_development=aux_metrics)),flush=True)
                    if sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())>2*1024**3:raise ValueError('2GiB retained fit artifact limit reached')
            if not bool((visits[fit]==2).all()) or bool(visits[dev].any()):raise ValueError('fixed source/query visit contract differs')
            selected=torch.load(best_path,map_location='cpu',weights_only=False);model.load_state_dict(selected['model']);head.load_state_dict(selected['person_head'])
            replay,replay_predictions=evaluate(model,data,dev,true_rank_donors,sources)
            if replay!=selected['metrics'] or replay_predictions!=json.loads((directory/'best-predictions.json').read_text()):raise ValueError('selected native replay changed metrics or crop decisions')
            retain(directory/'history.json',history)
            report=dict(arm=arm,seed=seed,best_epoch=best[-1],best_dev=replay,person_development=selected['person_development'],
                best_weights_sha256=sha(best_path),model_parameters=sum(p.numel() for p in model.parameters()),auxiliary_parameters=sum(p.numel() for p in head.parameters()),
                selected_replay_exact=True,source_query_visits_exact=True)
            retain(directory/'summary.json',report);reports.append(report)
    retain(a.output/'summary.json',reports)
    retain(a.output/'resources.json',dict(cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved()))
    update(pid=None,stage='complete',completed_fits=12,required_fits=12)
    print(json.dumps(dict(completed_fits=12,**json.loads((a.output/'resources.json').read_text()))))


if __name__=='__main__':main()
