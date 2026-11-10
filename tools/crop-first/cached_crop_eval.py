#!/usr/bin/env python3
"""Grade exact cached crop selections locally; no models, credentials or network."""
import argparse,hashlib,json,math,random
from collections import Counter,defaultdict
from pathlib import Path

FIELDS=['requests','crop_suggestions','judged_suggestions','accepted_suggestions',
        'safe_accepted_suggestions','unjudged_suggestions','proposed_face_box_violations',
        'bank_positive_tasks','bank_reject_all_tasks','bank_unknown_tasks',
        'avoidable_contain_tasks']


def ratio(a,b):return a/b if b else None


def rates(c):
    return {'acceptance_precision':ratio(c['accepted_suggestions'],c['judged_suggestions']),
            'crop_coverage':ratio(c['crop_suggestions'],c['requests']),
            'accepted_yield':ratio(c['accepted_suggestions'],c['requests']),
            'safe_accepted_yield':ratio(c['safe_accepted_suggestions'],c['requests']),
            'bank_positive_coverage':ratio(c['bank_positive_tasks'],c['requests']),
            'avoidable_contain_on_positive_tasks':ratio(c['avoidable_contain_tasks'],c['bank_positive_tasks'])}


def rows_from_predictions(document,predictions,use):
    if document['supervision']!='noisy_vlm_rendered_pixels':raise ValueError('explicit cached supervision required')
    sources=[s for s in document['sources'] if s['use']==use]
    if not sources:raise ValueError('requested source role has no cached labels')
    tasks={};seen_sources=set();families={}
    for s in document['sources']:
        if s['family_id'] in families and families[s['family_id']]!=s['use']:raise ValueError('source family crosses roles')
        families[s['family_id']]=s['use']
    for s in sources:
        if s['id'] in seen_sources:raise ValueError('duplicate source identity')
        seen_sources.add(s['id'])
        if (use=='eval' and s['bucket']!='validate') or (use=='dev' and s['bucket']!='train'):raise ValueError('source role differs from canonical use')
        grouped=defaultdict(dict)
        for c in s['candidates']:
            if c['candidate_id'] in grouped[c['task_id']]:raise ValueError('duplicate cached candidate')
            if not isinstance(c['acceptable'],bool) or not math.isfinite(c['weight']) or c['weight']<0:raise ValueError('invalid cached label')
            grouped[c['task_id']][c['candidate_id']]=c
        for tid,bank in grouped.items():tasks[(s['id'],tid)]=(s,bank)
    seen=set();rows=[]
    for pred in predictions:
        key=(pred['source_id'],pred['task_id'])
        if key not in tasks or key in seen:raise ValueError('unknown/duplicate selection task')
        seen.add(key);s,bank=tasks[key];first=next(iter(bank.values()));action=pred['research_action']
        if any((c['ratio'],c['mode'])!=(first['ratio'],first['mode']) for c in bank.values()):raise ValueError('mixed ratio/mode in cached task')
        if action not in ['crop','contain']:raise ValueError('explicit crop/contain policy required')
        crop=action=='crop';candidate=None
        if crop:
            if pred['candidate_id'] not in bank:raise ValueError('off-bank selection has no cached composition grade')
            candidate=bank[pred['candidate_id']]
            if pred['xywh']!=candidate['xywh']:raise ValueError('off-bank geometry cannot reuse a cached grade')
        elif pred['candidate_id'] is not None:raise ValueError('contain cannot masquerade as a labelled crop')
        positive=any(c['weight']>0 and c['acceptable'] for c in bank.values())
        reject_all=not positive and all(c['weight']>0 for c in bank.values())
        judged=crop and candidate['weight']>0;accepted=judged and candidate['acceptable'];cut=False
        if crop and s['face_context'][0]>0:
            x0,y0,x1,y1=candidate['box'];f=s['face_context']
            cut=not(x0<=f[1] and y0<=f[2] and x1>=f[3] and y1>=f[4])
        counts=dict.fromkeys(FIELDS,0);counts.update(requests=1,crop_suggestions=int(crop),
          judged_suggestions=int(judged),accepted_suggestions=int(accepted),
          safe_accepted_suggestions=int(accepted and not cut),unjudged_suggestions=int(crop and not judged),
          proposed_face_box_violations=int(cut),bank_positive_tasks=int(positive),
          bank_reject_all_tasks=int(reject_all),bank_unknown_tasks=int(not positive and not reject_all),
          avoidable_contain_tasks=int(not crop and positive))
        rows.append({'source_id':s['id'],'family_id':str(s['family_id']),'ratio':first['ratio'],
                     'mode':'maximal' if first['mode'] else 'minimal','counts':counts})
    if seen!=set(tasks):raise ValueError('every frozen task must have one explicit selection')
    return rows


def percentile(values,p):
    ordered=sorted(values);position=(len(ordered)-1)*p;lo=math.floor(position);hi=math.ceil(position)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)


def summarize(rows,resamples,seed):
    grouped=defaultdict(list)
    for r in rows:grouped[r['family_id']].append(r['counts'])
    raw={k:sum(r['counts'][k] for r in rows) for k in FIELDS}
    units=[{k:sum(c[k] for c in cs)/len(cs) for k in FIELDS} for _,cs in sorted(grouped.items())]
    weighted={k:sum(c[k] for c in units) for k in FIELDS};point=rates(weighted)
    rng=random.Random(seed);samples={k:[] for k in point}
    for _ in range(resamples):
        selected=rng.choices(units,k=len(units));value=rates({k:sum(c[k] for c in selected) for k in FIELDS})
        for k,v in value.items():
            if v is not None:samples[k].append(v)
    intervals={k:{'low':percentile(v,.025) if v else None,'high':percentile(v,.975) if v else None,
                  'defined_resamples':len(v),'total_resamples':resamples} for k,v in samples.items()}
    return {'source_count':len({r['source_id'] for r in rows}),'source_families':len(units),
            'counts':raw,'task_rates':rates(raw),'family_balanced_rates':point,
            'family_bootstrap_95_percentile':intervals}


def evaluate(document,predictions,use,*,resamples=2000,seed=20261108):
    if type(resamples) is not int or not 1<=resamples<=100_000:raise ValueError('bounded positive resample count required')
    rows=rows_from_predictions(document,predictions,use);by_ratio=defaultdict(list);by_mode=defaultdict(list)
    for r in rows:by_ratio[r['ratio']].append(r);by_mode[r['mode']].append(r)
    return {'use':use,'label_kind':document['supervision'],'human_rated':False,
      'face_safety_certified':False,'network_requests':0,'off_bank_policy':'error; never transfer grades by IoU',
      'bootstrap':{'unit':'source_family','resamples':resamples,'seed':seed,
                   'point_weighting':'equal family mass within each reported stratum'},
      'overall':summarize(rows,resamples,seed),
      'ratios':{k:summarize(v,resamples,seed) for k,v in sorted(by_ratio.items())},
      'modes':{k:summarize(v,resamples,seed) for k,v in sorted(by_mode.items())}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path)
    p.add_argument('selections',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--use',choices=['dev','eval'],required=True);p.add_argument('--resamples',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261108);a=p.parse_args()
    raw=a.manifest.read_bytes();selected=a.selections.read_bytes()
    result=evaluate(json.loads(raw),json.loads(selected),a.use,resamples=a.resamples,seed=a.seed)
    result['inputs']={'manifest':str(a.manifest),'manifest_sha256':hashlib.sha256(raw).hexdigest(),
                      'selections':str(a.selections),'selections_sha256':hashlib.sha256(selected).hexdigest()}
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'use':a.use,'families':result['overall']['source_families'],'counts':result['overall']['counts'],
                     'family_balanced_rates':result['overall']['family_balanced_rates'],'network_requests':0}))


if __name__=='__main__':main()
