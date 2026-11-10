#!/usr/bin/env python3
"""Cache frozen candidate hidden values and geometry-correct attention controls."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from artifact_identity import sha
from fit_unsplash_ranks import encode
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from local_rank_distillation import LocalCropRanker
from composition_features import candidate_features,NAMES


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','set_cache','direct_cache','cues','weights']:p.add_argument(key,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(a.set_cache)!='bfd84ff0ea48a7c3a2d2d32082f3fd8a9bd4a6c59b576899a80c4a7812f873c9' or sha(a.direct_cache)!='2556ff1a29837918d3c383276fe64542157a8cfdb60133a90dc6d9ffee3f9472' or sha(a.weights)!='c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df':raise ValueError('fixed source/bank/initializer differs')
    if sha(a.cues/'SHA256SUMS.json')!='d69e2460bfe6c0fd6a680969a35eb0828bb697049c22fb81cb64cfb65755215a':raise ValueError('fixed pose packet differs')
    for kind in ['gaze','scan']:
        summary=json.loads((a.root/kind/'summary.json').read_text())
        if summary['completed']!=72 or summary['required']!=72 or summary['diagnostic_maps_saved']:raise ValueError('complete compact teachers required')
    cache=torch.load(a.set_cache,map_location='cpu',weights_only=False);direct=torch.load(a.direct_cache,map_location='cpu',weights_only=False);sources=cache['sources']
    if sources!=direct['sources']:raise ValueError('fixed source metadata differs')
    inputs={r['id']:r for r in json.loads((a.root/'inputs.json').read_text())['images']}
    pose={r['id']:r['result'] for r in json.loads((a.cues/'local/float16-pose_landmarker_lite/results.json').read_text())['rows']}
    documents=[]
    for source in sources:
        row=inputs[source['id']];parts=[]
        for kind in ['gaze','scan']:
            doc=json.loads((a.root/kind/(source['id']+'.json')).read_text())
            if any(doc[key]!=row[key] for key in ['id','role','family_id','rgb_sha256']):raise ValueError('teacher source identity differs')
            parts.append(doc['result'])
        documents.append((pose[source['id']],*parts))
    model=LocalCropRanker().cuda();model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=False)['model']);model.eval()
    captured=[];hook=model.ranker.scorer[2].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
    image=cache['image'].cuda();features=encode(model,image,list(range(72)));content=cache['content'].cuda()
    aspect=torch.tensor([s['native_size'][0]/s['native_size'][1] for s in sources],device='cuda');hidden=[];scores=[]
    for q in range(22):
        p,r=map(int,cache['ratios'][q//2].split(':'));raw=model.scores(features,cache['query_boxes'][:,q].cuda(),aspect,aspect.new_full((72,),p/r),aspect.new_full((72,),q%2),content)
        if len(captured)!=1:raise ValueError('unexpected hidden capture count')
        value=captured.pop();replay=F.linear(value,model.ranker.scorer[2].weight,model.ranker.scorer[2].bias)[...,0]
        if value.shape!=(72,128,96) or not torch.equal(raw,replay):raise ValueError('frozen hidden replay differs')
        hidden.append(value.cpu());scores.append(raw.cpu())
    hook.remove();scorer={k:v.cpu().clone() for k,v in model.ranker.scorer[2].state_dict().items()};del model,features,image,content
    donors=[int(value) for value in direct['donors']]
    out=a.root/'cache';out.mkdir(exist_ok=False);extras=[];permuted=[];status=dict(pid=os.getpid(),stage='candidate attention features',completed=0,required=72)
    for i,source in enumerate(sources):
        if sources[donors[i]]['role']!=source['role']:raise ValueError('source donor crosses roles')
        boxes=cache['query_boxes'][i].reshape(-1,4).numpy()
        extras.append(torch.from_numpy(candidate_features(boxes,*documents[i])).reshape(22,128,32))
        permuted.append(torch.from_numpy(candidate_features(boxes,*documents[donors[i]])).reshape(22,128,32))
        status.update(completed=i+1,checked_unix=time.time());(out/'PROGRESS.json').write_text(json.dumps(status)+'\n')
        if (i+1)%12==0:print(json.dumps(status),flush=True)
    result=dict(schema='real-attention-hidden-v1',sources=sources,ratios=cache['ratios'],hidden=torch.stack(hidden,1),initial_scores=torch.stack(scores,1),
        extras=torch.stack(extras),permuted_extras=torch.stack(permuted),donors=donors,scorer_state=scorer,feature_names=NAMES,
        **{key:cache[key] for key in ['positive','negative','reference_valid','eligible','query_boxes']},
        identities=dict(set_cache_sha256=sha(a.set_cache),direct_cache_sha256=sha(a.direct_cache),weights_sha256=sha(a.weights),
            input_manifest_sha256=sha(a.root/'inputs.json'),pose_packet_sha256=sha(a.cues/'SHA256SUMS.json'),
            teacher_summaries={kind:sha(a.root/kind/'summary.json') for kind in ['gaze','scan']}),
        api_calls=0,test_inference=False,source_roles_changed=False,model_promoted=False,face_coverage_certified=False)
    if not all(bool(torch.isfinite(result[key]).all()) for key in ['hidden','initial_scores','extras','permuted_extras']):raise ValueError('nonfinite candidate cache')
    torch.save(result,out/'cache.pt');status.update(pid=None,stage='complete');(out/'PROGRESS.json').write_text(json.dumps(status)+'\n')
    summary=dict(sources=72,hidden_shape=list(result['hidden'].shape),extra_shape=list(result['extras'].shape),source_donors_moved=sum(i!=d for i,d in enumerate(donors)),
        cache_sha256=sha(out/'cache.pt'),cache_bytes=(out/'cache.pt').stat().st_size,initial_replay_exact=True,
        pose_positive_sources=sum(bool(d[0]['landmarks']) for d in documents),gaze_positive_sources=sum(bool(d[1]['people']) for d in documents))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
