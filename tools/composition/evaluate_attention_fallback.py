#!/usr/bin/env python3
"""Apply the unchanged old quality gate to attention heads' teacher-free fallback."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
from artifact_identity import sha
from fit_unsplash_ranks import evaluate
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from local_rank_distillation import LocalCropRanker
from distill_attention_linear import Student


class StudentScorer(torch.nn.Module):
    def __init__(self,student):super().__init__();self.student=student
    def forward(self,hidden):return self.student(hidden,None)[...,None]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fits','cache','weights','output']:p.add_argument(name,type=Path)
    p.add_argument('--fused-students',action='store_true')
    p.add_argument('--relationships',action='store_true')
    p.add_argument('--nonlinear-students',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(a.cache)!='2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031' or sha(a.weights)!='c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df':raise ValueError('frozen old inputs differ')
    for name in ['training/local_rank_distillation.py','training/crop_regions.py','training/crop_ranker.py','training/model_v3ds.py','tools/composition/fit_unsplash_ranks.py']:
        if Path(name).read_bytes()!=subprocess.check_output(['git','show','9f12692417d774653ed676d236e0f89eaafa7385:'+name]):raise ValueError('old quality implementation differs')
    cache=torch.load(a.cache,map_location='cpu',weights_only=False);sources=[dict(s,ratios=cache['ratios']) for s in cache['sources']]
    dev=[i for i,s in enumerate(sources) if s['role']=='dev']
    if len(dev)!=25:raise ValueError('fixed25 old DEV required')
    data={key:cache[key].cuda() for key in ['image','content','boxes','eligible','targets','pairs','pair_valid','query_valid']}
    data['source_aspect']=torch.tensor([s['native_size'][0]/s['native_size'][1] for s in sources],device='cuda')
    data['aspect']=torch.tensor([int(r.split(':')[0])/int(r.split(':')[1]) for r in cache['ratios'] for _ in range(2)],device='cuda')
    data['mode']=torch.tensor([q%2 for q in range(22)],device='cuda',dtype=torch.float32)
    donors=torch.arange(len(sources),device='cuda');initial=torch.load(a.weights,map_location='cpu',weights_only=False)['model']
    model=LocalCropRanker().cuda();model.load_state_dict(initial)
    baseline,base_rows=evaluate(model,data,dev,donors,sources)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'baseline-predictions.json').write_text(json.dumps(base_rows)+'\n');rows=[]
    fits=json.loads((a.fits/'summary.json').read_text())['fits']
    if sum([a.fused_students,a.relationships,a.nonlinear_students])>1:raise ValueError('distinct registered panels required')
    if len(fits)!=(6 if a.fused_students or a.nonlinear_students else 12 if a.relationships else 15):raise ValueError('complete registered fit panel required')
    for fit in fits:
        name=fit['arm']+'-seed'+str(fit['seed']);path=a.fits/name/'selected.pt';state=torch.load(path,map_location='cpu',weights_only=False)
        if a.nonlinear_students:
            model=LocalCropRanker().cuda();model.load_state_dict(initial)
            s=state['model'];student=Student({k:s['base.'+k] for k in ['weight','bias']},s['teacher'],s['mean'],s['std'],nonlinear=True);student.load_state_dict(s)
            model.ranker.scorer[2]=StudentScorer(student).cuda()
        else:
            model.load_state_dict(initial);model.ranker.scorer[2].load_state_dict(state['fused'] if a.fused_students else {k:state['model']['base.'+k] for k in ['weight','bias']})
        metrics,predictions=evaluate(model,data,dev,donors,sources)
        (a.output/(name+'-predictions.json')).write_text(json.dumps(predictions)+'\n')
        rows.append(dict(arm=fit['arm'],seed=fit['seed'],checkpoint_sha256=sha(path),metrics=metrics,
            old_essential_nonregression=metrics['essential']['teacher_regret']<=baseline['essential']['teacher_regret']))
    result=dict(baseline=baseline,rows=rows,old_cache_sha256=sha(a.cache),initializer_sha256=sha(a.weights),api_calls=0,test_inference=False,model_promoted=False,
        fallback='Enabled teacher-free nonlinear student' if a.nonlinear_students else 'Enabled fused teacher-free linear student' if a.fused_students else 'Disable the entire normalized attention contribution; keep trained base scalar weights. This is an explicit fallback, not zero raw features or teacher-free student distillation.',
        limitation='Old weak quality gate on25 reused DEV photos; full attention with old-source sidecars and artistic/face safety remain unverified.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(baseline=baseline['essential'],rows=[dict(arm=r['arm'],seed=r['seed'],regret=r['metrics']['essential']['teacher_regret'],passes=r['old_essential_nonregression']) for r in rows])))


if __name__=='__main__':main()
