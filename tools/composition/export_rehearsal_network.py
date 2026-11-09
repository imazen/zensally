#!/usr/bin/env python3
"""FIT/DEV-only encoder/query export, qualification and original-quality replay."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import onnx
import torch
from artifact_identity import sha
from export_unsplash_ranker import Encoder,Query,session,LocalCropRanker
from evaluate_attention_fallback import StudentScorer
from distill_attention_linear import Student


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['old','new','weights','student','output']:p.add_argument(key,type=Path)
    p.add_argument('--gather',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parents=[(a.old,'2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'),(a.new,'bfd84ff0ea48a7c3a2d2d32082f3fd8a9bd4a6c59b576899a80c4a7812f873c9'),(a.weights,'c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df'),(a.student,'4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b')]
    if any(sha(path)!=digest for path,digest in parents):raise ValueError('frozen network inputs differ')
    caches=[torch.load(path,map_location='cpu',weights_only=False) for path in [a.old,a.new]]
    if [Counter(s['role'] for s in c['sources']) for c in caches]!=[Counter(fit=94,dev=25),Counter(fit=48,dev=24)]:raise ValueError('fixed FIT/DEV only panels required')
    initial=torch.load(a.weights,map_location='cpu',weights_only=False)['model'];checkpoint=torch.load(a.student,map_location='cpu',weights_only=False);s=checkpoint['model']
    baseline=LocalCropRanker().eval();baseline.load_state_dict(initial)
    candidate=LocalCropRanker().eval();candidate.load_state_dict(initial)
    if any(not torch.equal(s['base.'+k],v) for k,v in candidate.ranker.scorer[2].state_dict().items()):raise ValueError('original scalar base differs')
    student=Student({k:s['base.'+k] for k in ['weight','bias']},s['teacher'],s['mean'],s['std'],nonlinear=True);student.load_state_dict(s);candidate.ranker.scorer[2]=StudentScorer(student);candidate.eval()
    a.output.mkdir(parents=True,exist_ok=False);sample=caches[0];image=sample['image'][:1];features=baseline.feature_maps(image)
    args=(*features,sample['boxes'][:1,0],torch.ones(1),torch.ones(1),torch.zeros(1),sample['content'][:1]);inputs=['f8','f16','boxes','source_aspect','target_aspect','mode','content'];exports={}
    before={k:v.clone() for k,v in candidate.state_dict().items()}
    for name,module,values,names,outputs in [('encoder',Encoder(baseline),(image,),['image'],['f8','f16']),('baseline-query',Query(baseline),args,inputs,['scores']),('candidate-query',Query(candidate),args,inputs,['scores'])]:
        path=a.output/(name+'.onnx');module.eval();torch.onnx.export(module,values,str(path),input_names=names,output_names=outputs,opset_version=18,dynamo=False,external_data=False)
        if a.gather and name!='encoder':
            from gather_grid import lower_student_grid_samples,lower_rehearsal_grid_samples
            (lower_student_grid_samples if name=='baseline-query' else lower_rehearsal_grid_samples)(path)
        graph=onnx.load(str(path));onnx.checker.check_model(graph,full_check=True);exports[name]=dict(bytes=path.stat().st_size,sha256=sha(path),operators=dict(Counter(n.op_type for n in graph.graph.node)))
    if any(m.training for m in candidate.modules()) or any(not torch.equal(v,before[k]) for k,v in candidate.state_dict().items()):raise ValueError('export mutated candidate')
    encoder=session(a.output/'encoder.onnx');queries={name:session(a.output/(name+'-query.onnx')) for name in ['baseline','candidate']};gpu=candidate.cuda().eval();cpu=baseline
    totals={k:dict(queries=0,changed_choices=0,gate_violations=0,max_edge_movement_128_grid=0.,max_abs_score_difference=0.) for k in ['torch-cpu-candidate','ort-candidate','ort-baseline']};changed=[];completed=0
    candidate_cpu=LocalCropRanker().eval();candidate_cpu.load_state_dict(initial);student_cpu=Student({k:s['base.'+k] for k in ['weight','bias']},s['teacher'],s['mean'],s['std'],nonlinear=True);student_cpu.load_state_dict(s);candidate_cpu.ranker.scorer[2]=StudentScorer(student_cpu);candidate_cpu.eval()
    for cohort,cache in zip(['old','new'],caches):
        for i,source in enumerate(cache['sources']):
            image=cache['image'][i:i+1];cpu_features=cpu.feature_maps(image);gpu_features=gpu.feature_maps(image.cuda());ort_features=encoder.run(None,dict(image=image.numpy()))
            for q in range(22):
                p,r=map(int,cache['ratios'][q//2].split(':'));boxes=cache['query_boxes'][i:i+1,q] if 'query_boxes' in cache else cache['boxes'][i:i+1,q//2]
                values=(boxes,torch.tensor([source['native_size'][0]/source['native_size'][1]]),torch.tensor([p/r]),torch.tensor([q%2],dtype=torch.float32),cache['content'][i:i+1]);feed={name:v.numpy() for name,v in zip(inputs[2:],values)}
                reference=gpu.scores(gpu_features,*(v.cuda() for v in values)).cpu().numpy()[0];base=cpu.scores(cpu_features,*values).numpy()[0]
                results={'torch-cpu-candidate':candidate_cpu.scores(cpu_features,*values).numpy()[0],**{'ort-'+name:query.run(None,dict(f8=ort_features[0],f16=ort_features[1],**feed))[0][0] for name,query in queries.items()}}
                eligible=cache['eligible'][i,q].numpy();native_boxes=boxes.numpy()[0]
                for name,score in results.items():
                    if not np.isfinite(score).all():raise ValueError('nonfinite runtime')
                    target=base if name=='ort-baseline' else reference;left=int(np.where(eligible,target,-np.inf).argmax());right=int(np.where(eligible,score,-np.inf).argmax());motion=float(np.abs(native_boxes[left]-native_boxes[right]).max()*128);row=totals[name]
                    row['queries']+=1;row['changed_choices']+=left!=right;row['gate_violations']+=motion>3;row['max_edge_movement_128_grid']=max(row['max_edge_movement_128_grid'],motion);row['max_abs_score_difference']=max(row['max_abs_score_difference'],float(np.abs(target-score).max()))
                    if left!=right:changed.append(dict(cohort=cohort,id=source['id'],role=source['role'],query=q,runtime=name,reference=left,selected=right,edge_movement_128_grid=motion))
            completed+=1
            (a.output/'PROGRESS.json').write_text(json.dumps(dict(stage='qualifying',completed=completed,required=191,pid=__import__('os').getpid()))+'\n')
    report=dict(inputs={str(path):digest for path,digest in parents},exports=exports,totals=totals,changed_choices=changed,sources=191,fit_sources=142,dev_sources=49,gather_layout=a.gather,test_inference=False,api_calls=0,model_promoted=False,reference='candidate Torch CUDA full encoder/query batch1; baseline ORT compares own Torch CPU full model',scope='owned cached256x256 tensors and frozen128-candidate banks; excludes decode, resize and safety guards')
    (a.output/'qualification.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'PROGRESS.json').write_text(json.dumps(dict(stage='complete',completed=191,required=191,pid=None))+'\n');print(json.dumps(dict(totals=totals,sources=191)))
    if any(r['gate_violations'] for r in totals.values()):raise SystemExit(1)


if __name__=='__main__':main()
