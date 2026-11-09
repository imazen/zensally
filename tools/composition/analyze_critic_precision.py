#!/usr/bin/env python3
"""Compare the complete unquantized critic reference with frozen NF4 controls."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
from artifact_identity import sha
from critic_probe import protocol_for


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','plan','reference','nf4','vision_bf16','output']:p.add_argument(name,type=Path)
    p.add_argument('--prior-reference',type=Path)
    a=p.parse_args();plan=json.loads(a.plan.read_text());rows=plan['images'];roots=dict(reference=a.reference,nf4=a.nf4,vision_bf16=a.vision_bf16);protocols={name:json.loads((root/'protocol.json').read_text()) for name,root in roots.items()};inputs={str(a.plan):sha(a.plan)}
    if len(rows)!=16 or sha(a.plan)!='51b788924c962a93ac9e8ec069ec72a147bf08baa06e7fdb5342e2153cc5b901':raise ValueError('registered original16-source plan differs')
    if protocols['reference']!=protocol_for(a.model,a.plan,False,'bf16') or protocols['vision_bf16']!=protocol_for(a.model,a.plan,True,'nf4'):raise ValueError('registered reference or NF4/BF16-vision protocol differs')
    for name,protocol in protocols.items():
        for key in ['plan_sha256','model_assets_sha256','prompt','loader_adjustment','max_new_tokens','max_pixels','seed']:
            if protocol[key]!=protocols['reference'][key]:raise ValueError('quantization comparison changes source/model/prompt inputs')
    placement=json.loads((a.reference/'placement.json').read_text())
    if placement['quantized'] or placement['parameter_dtypes']!=['torch.bfloat16'] or 'disk' in placement['modules'].values():raise ValueError('reference is not unquantized BF16 without disk offload')
    results={};summary={};groups={}
    for row in rows:
        if row['role'] not in ['annotation_only','dev']:raise ValueError('original calibration roles differ; TEST closed')
        if row['kind']=='candidate':groups.setdefault(row['task_id'],[]).append(row)
    if len(groups)!=6 or any(len(v)!=2 or {r['weak_teacher_acceptable'] for r in v}!={True,False} for v in groups.values()):raise ValueError('all six fixed weak-judgment pairs required')
    for name,root in roots.items():
        terminal=json.loads((root/'summary.json').read_text())
        if terminal['required']!=16 or terminal['completed']!=16:raise ValueError('complete16-source critic result required')
        items={}
        for row in rows:
            path=root/(row['id']+'.json');r=json.loads(path.read_text());inputs[str(path)]=sha(path)
            if r['input']!=row or r['image_grid_thw']!=[[1,row['height']//14,row['width']//14]]:raise ValueError('raw response source/grid differs')
            items[row['id']]=r
        results[name]=items;pairs=[]
        for task,candidates in groups.items():
            accepted=next(r for r in candidates if r['weak_teacher_acceptable']);rejected=next(r for r in candidates if not r['weak_teacher_acceptable']);x,y=[items[r['id']]['rating'] for r in [accepted,rejected]];delta=Decimal(str(x))-Decimal(str(y)) if x is not None and y is not None else None;pairs.append(dict(task_id=task,accepted=accepted['id'],rejected=rejected['id'],accepted_rating=x,rejected_rating=y,difference=float(delta) if delta is not None else None,order='unparsed' if delta is None else 'correct' if delta>0 else 'tie' if delta==0 else 'reversed',separation_pass=delta is not None and delta>=Decimal('.05')))
        summary[name]=dict(parsed=sum(r['rating'] is not None and r['parse_error'] is None for r in items.values()),truncated=sum(r['token_limit_reached'] for r in items.values()),correct=sum(p['order']=='correct' for p in pairs),tied=sum(p['order']=='tie' for p in pairs),reversed=sum(p['order']=='reversed' for p in pairs),separation_passes=sum(p['separation_pass'] for p in pairs),pairs=pairs)
    differences=[]
    for name in ['nf4','vision_bf16']:
        deltas=[];changes=0
        for row in rows:
            ref=results['reference'][row['id']];other=results[name][row['id']]
            if ref['input_tokens']!=other['input_tokens']:raise ValueError('tokenized model inputs differ')
            if ref['rating'] is not None and other['rating'] is not None:deltas.append(abs(Decimal(str(ref['rating']))-Decimal(str(other['rating']))))
        for ref,other in zip(summary['reference']['pairs'],summary[name]['pairs']):
            if ref['task_id']!=other['task_id']:raise ValueError('pair alignment differs')
            changes+=ref['order']!=other['order']
        differences.append(dict(control=name,comparable_ratings=len(deltas),max_absolute_rating_difference=float(max(deltas)) if deltas else None,changed_pair_orders=changes))
    replay=None
    if a.prior_reference is not None:
        checked=[]
        for row in rows[:4]:
            path=a.prior_reference/(row['id']+'.json');prior=json.loads(path.read_text());current=results['reference'][row['id']];inputs[str(path)]=sha(path)
            if prior['input']!=row:raise ValueError('partial reference source identity differs')
            checked.append(dict(id=row['id'],rating_equal=prior['rating']==current['rating'],response_equal=prior['response']==current['response'],token_count_equal=prior['output_tokens']==current['output_tokens']))
        replay=dict(checked=4,all_equal=all(r['rating_equal'] and r['response_equal'] and r['token_count_equal'] for r in checked),rows=checked)
    ref=summary['reference'];passed=ref['parsed']==16 and ref['truncated']==0 and ref['separation_passes']==6;report=dict(inputs=inputs,protocols={name:str(root/'protocol.json') for name,root in roots.items()},models=summary,quantization_differences=differences,resource_repair_replay=replay,qualification_passed=passed,admit_training_labels=False,wider_panel_authorized_by_protocol=passed,scope='small unreviewed weak-judgment calibration panel; no human artistic or complete-face claim',api_calls=0,test_inference=False,model_promoted=False)
    with a.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(models={name:{k:v for k,v in r.items() if k!='pairs'} for name,r in summary.items()},quantization_differences=differences,resource_repair_replay=replay,qualification_passed=passed)))


if __name__=='__main__':main()
