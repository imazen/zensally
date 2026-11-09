#!/usr/bin/env python3
"""Full cached-image CUDA and compact cut-head qualification of fixed risk policies."""
import argparse
import json
from pathlib import Path
import torch
from artifact_identity import sha
from audit_face_corroboration import verify_member
from distill_attention_linear import Student,write
from local_rank_distillation import LocalCropRanker
from fit_unsplash_ranks import encode
from pair_relationship_head import PairCorrection
from candidate_cut_head import CutHead,features
from person_risk_inputs import load
from person_risk_policy import adjusted_scores,hard_logits
from candidate_cut_routing import route


def select(scores,mask,logits,person,row):
    active=person if row['applicability']=='person' else torch.ones_like(person)
    if row['kind']=='hard':return route(scores,mask,hard_logits(logits,active))
    return route(adjusted_scores(scores,logits,active,row['strength']),mask,torch.full_like(logits,-1.))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fits','study','miap_objects','unsplash_objects','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;contexts,fits,people,identities=load(a.fits,a.miap_objects,a.unsplash_objects);expected_people=json.loads((a.study/'people.json').read_text())
    if people!=expected_people:raise ValueError('runtime applicability source identities differ')
    selected=json.loads((a.study/'selected.json').read_text());grid=json.loads((a.study/'grid.json').read_text());policies=[*selected,*[r for r in grid if r['kind']=='hard']]
    if len(policies)!=60:raise ValueError('thirty FIT-selected strengths and thirty hard controls required')
    by_hash={v:Path(k) for k,v in identities.items()};pair_path=by_hash['05f41e086a3043bce8797b33b3e2d932c1117025c074101161a73b72aa52dede'];pair_root=pair_path.parents[1];verify_member(pair_root,'qualification/summary.json');image_inputs=json.loads((pair_root/'qualification/summary.json').read_text())['inputs'];image_paths={v:Path(k) for k,v in image_inputs.items()}
    for path,digest in image_inputs.items():
        if sha(Path(path))!=digest:raise ValueError('frozen image-network input differs')
    network=LocalCropRanker().cuda();network.load_state_dict(torch.load(image_paths['c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df'],map_location='cpu',weights_only=False)['model']);network.eval();state=torch.load(by_hash['4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b'],map_location='cpu',weights_only=False)['model'];endpoint=Student({k:state['base.'+k] for k in ['weight','bias']},state['teacher'],state['mean'],state['std'],nonlinear=True);endpoint.load_state_dict(state);endpoint.cuda().eval();pair=PairCorrection('additive-linear',state['mean'],state['std']);pair.load_state_dict(torch.load(pair_path,map_location='cpu',weights_only=False)['model']);pair.cuda().eval();reports=[];images=0;queries=0
    for cohort,image_hash in [('new','bfd84ff0ea48a7c3a2d2d32082f3fd8a9bd4a6c59b576899a80c4a7812f873c9'),('old','2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031')]:
        ctx=contexts[cohort];c=ctx['data'];indices=[i for i,s in enumerate(c['sources']) if s['role']=='dev'];source_ids=[c['sources'][i]['id'] for i in indices];image_cache=torch.load(image_paths[image_hash],map_location='cpu',weights_only=False);image_lookup={s['id']:i for i,s in enumerate(image_cache['sources'])};ix=[image_lookup[sid] for sid in source_ids]
        for j,i in zip(indices,ix):
            if any(c['sources'][j][k]!=image_cache['sources'][i][k] for k in ['id','role','family_id','rgb_sha256']):raise ValueError('full network source identity differs')
        count=len(indices);features_image=encode(network,image_cache['image'][ix].cuda(),list(range(count)));content=image_cache['content'][ix].cuda();aspect=torch.tensor([c['sources'][j]['native_size'][0]/c['sources'][j]['native_size'][1] for j in indices],device='cuda');captured=[];hidden=[];hook=network.ranker.scorer[2].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
        for q in range(22):
            p,r=map(int,c['ratios'][q//2].split(':'));box=c['query_boxes'][indices,q].cuda();network.scores(features_image,box,aspect,torch.full_like(aspect,p/r),torch.full_like(aspect,q%2),content)
            if len(captured)!=1:raise ValueError('one frozen hidden capture required')
            hidden.append(captured.pop());queries+=count
        hook.remove();images+=count;full_hidden=torch.stack(hidden,1);h=full_hidden.reshape(count,11,2,128,96);full_scores=pair(h,endpoint(h,None),ctx['eligible'][indices].cuda()).cpu();person=ctx['person'][indices];mask=ctx['mask'][indices];native=ctx['native'][indices];scale=ctx['scale'][indices]
        for entry in fits:
            name=f"{entry['arm']}-seed{entry['seed']}";path=a.fits/name/'selected.pt'
            if sha(path)!=entry['checkpoint_sha256']:raise ValueError('classifier checkpoint differs')
            s=torch.load(path,map_location='cpu',weights_only=False)['model'];model=CutHead(s['mean'],s['std']);model.load_state_dict(s);x=features(c,entry['arm'])[indices];reference=model(x).reshape(count,11,2,128);compact=model.compact(x).reshape(count,11,2,128)
            if entry['arm']=='geometry':gx=x.cuda()
            elif entry['arm']=='hidden':gx=full_hidden
            else:
                extra=c['permuted_extras'] if entry['arm']=='permuted-attention' else c['extras'];gx=torch.cat([full_hidden,extra[indices,...,:16 if entry['arm']=='hidden-pose' else 32].cuda()],-1)
            gpu=model.cuda()(gx).cpu().reshape(count,11,2,128)
            for row in [r for r in policies if r['model']==name]:
                ref,offered=select(ctx['scores'][indices],mask,reference,person,row)
                for implementation,scores,logits in [('compact',ctx['scores'][indices],compact),('full-cuda',full_scores,gpu)]:
                    actual,present=select(scores,mask,logits,person,row);presence=int((offered!=present).sum());changed=(actual!=ref).any(-1)&offered&present;edge=0.;violations=presence
                    for j,ri in changed.nonzero().tolist():
                        movement=max(float(((native[j,ri,m,actual[j,ri,m]]-native[j,ri,m,ref[j,ri,m]]).abs().float()/scale[j,0,0]*128).max()) for m in range(2));edge=max(edge,movement);violations+=movement>3
                    reports.append(dict(cohort=cohort,model=name,kind=row['kind'],applicability=row['applicability'],strength=row['strength'],implementation=implementation,pair_requests=count*11,endpoint_changes=int(((actual!=ref)&offered[...,None]&present[...,None]).sum()),presence_changes=presence,max_edge_movement_grid=edge,gate_violations=violations,max_logit_difference=float((logits-reference).abs().max())))
    a.output.mkdir(exist_ok=False);write(a.output/'summary.json',dict(inputs=identities,image_inputs=image_inputs,study_selected_sha256=sha(a.study/'selected.json'),images=images,neural_queries=queries,rows=reports,scope='full fixed encoder/query, endpoint, pair and risk heads on cached owned DEV image tensors; measured auxiliary features and detector applicability reused; excludes preprocessing and teacher/detector inference',api_calls=0,test_inference=False,model_promoted=False));print(json.dumps(dict(images=images,neural_queries=queries,comparisons=len(reports),endpoint_changes=sum(r['endpoint_changes'] for r in reports),presence_changes=sum(r['presence_changes'] for r in reports),max_edge_movement_grid=max(r['max_edge_movement_grid'] for r in reports),gate_violations=sum(r['gate_violations'] for r in reports))))
    if any(r['gate_violations'] for r in reports):raise SystemExit('full risk routing numerical gate failed')


if __name__=='__main__':main()
