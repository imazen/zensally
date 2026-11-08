#!/usr/bin/env python3
"""CPU fixture: interrupted real-loader training must match uninterrupted state."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sources=[]
    for i,use in enumerate(['fit','fit','dev']):
        raw=bytes([30+i*30,60,120])*(256*256);path=a.output/f'{i}.rgb';path.write_bytes(raw)
        s={'id':str(11000+i*2),'family_id':str(1000+i*2),'bucket':'train','use':use,'historical_eval_family':False,'width':640,'height':480,'valid_rect':[0.,.125,1.,.875],'tensor_path':str(path),'tensor_sha256':hashlib.sha256(raw).hexdigest(),'training_eligible':use=='fit','human_rated':False,'face_safety_certified':False,'face_context':[0,0,0,0,0,0],
           'candidates':[{'task_id':str(i)+'-one','candidate_id':'A','ratio':'2:3','box':[0.,0.,.5,1.],'xywh':[0,0,320,480],'aspect':2/3,'mode':0,'acceptable':True,'weight':1.},{'task_id':str(i)+'-one','candidate_id':'B','ratio':'2:3','box':[.5,0.,1.,1.],'xywh':[320,0,320,480],'aspect':2/3,'mode':0,'acceptable':False,'weight':1.}],'pairs':[{'preferred':0,'other':1,'weight':1.}]};sources.append(s)
    manifest=a.output/'fixture.json';manifest.write_text(json.dumps({'schema':1,'supervision':'noisy_vlm_rendered_pixels','sources':sources}))
    command=[sys.executable,str(Path(__file__).with_name('fit.py')),str(manifest)]
    args=['--device','cpu','--precision','fp32','--epochs','4','--batch','1','--workers','2','--threads','1','--seed','101']
    for name,extra in [('whole',[]),('partial',['--stop-after-epoch','2']),('resumed',['--resume',str(a.output/'partial/last.pth')])]:
        with (a.output/(name+'.log')).open('x') as log:subprocess.run(command+[str(a.output/name)]+args+extra,check=True,stdout=log,stderr=subprocess.STDOUT)
    one=torch.load(a.output/'whole/last.pth',map_location='cpu',weights_only=False);two=torch.load(a.output/'resumed/last.pth',map_location='cpu',weights_only=False)
    for k,v in one['model'].items():
        if not torch.equal(v,two['model'][k]):raise ValueError('resumed model differs: '+k)
    if one['history']!=two['history']:raise ValueError('resumed training history differs')
    # Holdout gradients and source identity must fail before an optimizer step.
    bad=json.loads(manifest.read_text());bad['sources'][0]['bucket']='validate';badpath=a.output/'heldout-invalid.json';badpath.write_text(json.dumps(bad))
    with (a.output/'heldout-invalid.log').open('x') as log:result=subprocess.run([sys.executable,str(Path(__file__).with_name('fit.py')),str(badpath),str(a.output/'invalid-run')]+args,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode==0 or (a.output/'invalid-run/last.pth').exists():raise ValueError('heldout source reached fitting')
    result={'fixture_only':True,'epochs':4,'model_state_bit_exact':True,'history_exact':True,'canonical_holdout_rejected_before_fitting':True,'tensors_compared':len(one['model'])}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
