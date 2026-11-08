#!/usr/bin/env python3
"""Admit authorized TRAIN-only rendered weak labels; retain every exclusion."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
from corpus import save,sha
import review

def read_reviews(plan_path,results,cache):
    labels={};evidence=[]
    for job in json.loads(plan_path.read_text()):
        path=results/(job['id']+'.json')
        if not path.exists():raise ValueError(f'missing review result: {job["id"]}')
        result=json.loads(path.read_text());entry=cache/Path(result['entry']).name
        hashes=json.loads((entry/'hashes.json').read_text())
        for name,digest in hashes.items():
            if sha((entry/name).read_bytes())!=digest:raise ValueError('cached review bytes changed')
        task=json.loads(Path(job['task']).read_text());response=json.loads((entry/'response.json').read_text())
        if response.get('model')!=job['model']:raise ValueError('review response model differs from frozen job')
        label=review.validate(response,task)
        for t in label['tasks']:
            if t['task_id'] in labels:raise ValueError('duplicate rendered review task')
            labels[t['task_id']]=t
        evidence.append({'job_id':job['id'],'cache_key':result['cache_key'],'response_id':response['id'],'usage':response.get('usage')})
    return labels,evidence

def admitted_source(packet,terra,sol,audit):
    candidates=[];pairs=[];reasons=Counter();face_rois=[r['box'] for r in packet['rois'] if r['kind']=='face_head']
    face_context=[len(face_rois),min([r['left']/100 for r in face_rois],default=0),min([r['top']/100 for r in face_rois],default=0),max([r['right']/100 for r in face_rois],default=0),max([r['bottom']/100 for r in face_rois],default=0),0.]
    for task in packet['tasks']:
        tid=task['task_id'];label=terra[tid];other=sol.get(tid);other_by_id={r['candidate_id']:r for r in other['candidates']} if other else {}
        preferred=label['preferred'];rank_agrees=not other or other['preferred']==preferred
        rw,rh=map(int,task['ratio'].split(':'));weight=1. if 1/3<=rw/rh<=3 else .25
        indices={};start=len(candidates);task_admitted=[]
        for c,j in zip(task['candidates'],label['candidates']):
            if c['id']!=j['candidate_id']:raise ValueError('candidate/judgment identity mismatch')
            eligible=True;why=[]
            if j['confidence']=='low':eligible=False;why.append('low_teacher_confidence')
            if j['face_cut']=='uncertain':eligible=False;why.append('uncertain_face_coverage')
            if j['acceptable'] and c['missed_proposed_faces']:eligible=False;why.append('positive_conflicts_with_proposed_face_geometry')
            if other:
                sj=other_by_id[c['id']]
                trusted=sj['confidence']!='low' and sj['face_cut']!='uncertain'
                if trusted:
                    audit[task['ratio']]['compared_candidates']+=1
                    audit[task['ratio']]['acceptability_agree']+=int(sj['acceptable']==j['acceptable'])
                    audit[task['ratio']]['face_assessment_agree']+=int(sj['face_cut']==j['face_cut'])
                    if sj['acceptable']!=j['acceptable'] or sj['face_cut']!=j['face_cut']:eligible=False;why.append('terra_sol_disagreement')
            index=len(candidates);indices[c['id']]=index;x,y,w,h=c['xywh'];sw,sh=packet['width'],packet['height']
            candidates.append({'task_id':tid,'candidate_id':c['id'],'ratio':task['ratio'],'mode':int(task['mode']=='maximal'),'aspect':rw/rh,'box':[x/sw,y/sh,(x+w)/sw,(y+h)/sh],'xywh':c['xywh'],'image':c['image'],'acceptable':j['acceptable'],'weight':weight if eligible else 0.,'exclusions':why})
            reasons.update(why)
            if eligible:task_admitted.append(index)
        if other:
            audit[task['ratio']]['compared_tasks']+=1;audit[task['ratio']]['preferred_agree']+=int(rank_agrees)
        # Each task, and then each source, contributes equally within its tier.
        for i in task_admitted:candidates[i]['weight']/=len(task_admitted)
        if preferred is not None and rank_agrees:
            i=indices[preferred]
            others=[j for j in task_admitted if j!=i]
            if candidates[i]['weight']>0 and others:
                pairs.extend({'preferred':i,'other':j,'weight':weight/len(others)} for j in others)
    use=packet['use'];eligible=use=='fit'
    if eligible and (packet['bucket']!='train' or packet['historical_eval_family']):raise ValueError('held-out source cannot be admitted to fitting')
    provenance={k:packet[k] for k in ['canonical_url','license','provenance'] if k in packet}
    return {k:packet[k] for k in ['id','family_id','bucket','use','historical_eval_family','width','height','valid_rect','tensor_sha256','sha256','png_sha256']}|provenance|{'tensor_path':str(Path(packet['directory'])/'tensor.rgb'),'training_eligible':eligible,'human_rated':False,'face_safety_certified':False,'face_context':face_context,'candidates':candidates,'pairs':pairs,'exclusion_counts':dict(reasons)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('rendered',type=Path);p.add_argument('results',type=Path);p.add_argument('cache',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    terra,te=read_reviews(a.rendered/'terra-plan.json',a.results,a.cache);sol,se=read_reviews(a.rendered/'sol-plan.json',a.results,a.cache);audit=defaultdict(Counter);sources=[];excluded=[]
    for path in sorted(a.rendered.glob('*/packet.json')):
        s=admitted_source(json.loads(path.read_text()),terra,sol,audit)
        if any(c['weight']>0 for c in s['candidates']):sources.append(s)
        else:excluded.append({'id':s['id'],'use':s['use'],'reason':'no usable rendered weak labels','counts':s['exclusion_counts']})
    document={'schema':1,'supervision':'noisy_vlm_rendered_pixels','authorization':'User authorized TRAIN-only weak fitting and Sol spot checks on 2026-11-08','essential_ratio_range':['1:3','3:1'],'secondary_ratio_weight':.25,'human_rated':False,'sources':sources,'excluded_sources':excluded,'teacher_evidence':te,'sol_evidence':se,'sol_audit':{k:dict(v) for k,v in audit.items()}}
    save(a.output,document)
    print(json.dumps({'uses':dict(Counter(s['use'] for s in sources)),'candidate_labels':sum(c['weight']>0 for s in sources for c in s['candidates']),'strict_pairs':sum(len(s['pairs']) for s in sources),'excluded_sources':excluded,'sol_audit':document['sol_audit']}))
if __name__=='__main__':main()
