#!/usr/bin/env python3
"""Verify the complete frozen study and summarize teacher decisions, not quality."""
import argparse,collections,csv,hashlib,json,math
from pathlib import Path
from expanded import planned
from teacher import parse_selection


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('inputs',type=Path);p.add_argument('root',type=Path);p.add_argument('--partial',action='store_true');p.add_argument('--coverage-audit',action='store_true');a=p.parse_args()
    sources={r['id']:r for r in json.loads((a.inputs/'sources.json').read_text())};protocol=json.loads((a.root/'protocol.json').read_text())
    if a.coverage_audit:
        sources={k:v for k,v in sources.items() if k in protocol['source_ids']}
        if len(sources)!=4 or protocol['required_tasks']!=16:raise ValueError('coverage audit requires its frozen four sources and 16 tasks')
    expected={t['task_id']:(s,t) for s in sources.values() for t in planned(s) if not a.coverage_audit or t['target_ratio'] in [[1,8],[8,1]]};records=[];seen=set();geometry=0;pixels={};stats=collections.defaultdict(lambda:collections.Counter())
    for file in sorted((a.root/'batches').glob('??-labels.json')):
        packet=json.loads(file.read_text());name=file.name[:2];tasks={t['task_id']:t for t in json.loads((a.root/'batches'/f'{name}-tasks.json').read_text())}
        for record in packet['records']:
            tid=record['task_id']
            if tid in seen or tid not in expected:raise ValueError('duplicate/unexpected task label')
            seen.add(tid);source,t=expected[tid];task=tasks[tid]
            if record['source_role']!=source['role'] or record['training_eligible'] or record['face_safety_certified'] or record['admission']!='unreviewed':raise ValueError('role/admission contract changed')
            cs=[c for c in csv.DictReader((a.root/'crops'/source['id']/'candidates.tsv').open(),delimiter='\t') if c['task_id']==tid]
            ids=[c['candidate'] for c in cs];label=parse_selection(record['label'],ids)
            image_records=[task['source'],*task['candidates']]
            if record['image_sha256']!=[v['sha256'] for v in image_records]:raise ValueError('label images differ from canonical task')
            for image in image_records:
                # Rebind the retained absolute run path to this verified copy.
                path=a.root/'crops'/source['id']/Path(image['path']).name
                if path not in pixels:pixels[path]=hashlib.sha256(path.read_bytes()).hexdigest()
                if pixels[path]!=image['sha256']:raise ValueError('candidate/source bytes changed')
            rw,rh=task['target_ratio'];g=math.gcd(rw,rh);rw//=g;rh//=g;k=min(source['width']//rw,source['height']//rh)
            for c in cs:
                x,y,w,h=[int(c[v]) for v in ['x','y','w','h']]
                if min(w,h)<=0 or min(x,y)<0 or x+w>source['width'] or y+h>source['height'] or w*rh!=h*rw:raise ValueError('invalid crop geometry')
                if task['mode']=='minimal' and (w,h)!=(k*rw,k*rh):raise ValueError('Minimal crop area changed')
                geometry+=1
            decision='selected' if label['preferred'] is not None else 'contain_or_rejected'
            origin='none'
            if label['preferred'] is not None:
                winner=next(c for c in cs if c['candidate']==label['preferred'])
                if 'vision-proposal' in winner['generators']:origin='center_and_vision' if winner['candidate']=='C01' else 'vision'
                elif winner['candidate']=='C01':origin='center'
                elif 'coverage-axis-control' in winner['generators']:origin='coverage_axis_control'
                else:origin='interior_or_scale_baseline'
            for group in ['all',source['role'],source['content_kind'],f"{rw}:{rh}",task['mode']]:stats[group][decision]+=1
            stats['winning_origin'][origin]+=1
            records.append({**record,'content_kind':source['content_kind'],'target_ratio':task['target_ratio'],'mode':task['mode'],'direct_action':task['direct_action']})
    missing=sorted(set(expected)-seen)
    if missing and not a.partial:raise ValueError(f'{len(missing)} of {len(expected)} required judgments missing')
    summary={'required_tasks':len(expected),'judged_tasks':len(seen),'complete':not missing,'source_count':len(sources),'roles':dict(collections.Counter(s['role'] for s in sources.values())),'content_groups':len({s['content_group'] for s in sources.values()}),'decoded_source_aspects':dict(collections.Counter('4+' if max(s['width']/s['height'],s['height']/s['width'])>=4 else '2-4' if max(s['width']/s['height'],s['height']/s['width'])>=2 else '<2' for s in sources.values())),'candidate_rectangles_checked':geometry,'unique_image_hashes_checked':len(pixels),'decisions':dict(stats),'quality_measured':False,'human_calibrated':False,'training_eligible':False,'face_safety_certified':False,'source_roles_changed':False,'note':'Teacher decisions are weak labels, not independent artistic quality or face-detection measurements. Source-aspect and target-aspect coverage are distinct.'}
    suffix=f'-partial-{len(seen)}' if missing else ''
    if a.coverage_audit:
        original={r['task_id']:r for r in json.loads((a.root.parent/'cli-labels.json').read_text())['records']}
        comparison=collections.Counter();reversals=[]
        for r in records:
            before=original[r['task_id']]['label']['preferred'];after=r['label']['preferred']
            comparison[('selected' if before else 'rejected')+'_to_'+('selected' if after else 'rejected')]+=1
            if 'C01' in r['label']['acceptable'] and 'C01' not in original[r['task_id']]['label']['acceptable']:
                name=r['task_id'].split('-',1)[1]+'-C01.png'
                hashes=[hashlib.sha256((root/'crops'/r['source_id']/name).read_bytes()).hexdigest() for root in [a.root.parent,a.root]]
                if hashes[0]!=hashes[1]:raise ValueError('center-control pixels changed between main and audit')
                reversals.append({'task_id':r['task_id'],'identical_center_png_sha256':hashes[0]})
        summary['main_vs_broader_pool']=dict(comparison)
        summary['identical_center_rejected_then_accepted']=reversals
        summary['note']+=' The companion audit changes candidate coverage, not the main study; teacher context/order are not controlled causal comparisons.'
    (a.root/f'summary{suffix}.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.root/f'cli-labels{suffix}.json').write_text(json.dumps({'supervision':'vlm_preference','admission':'unreviewed','records':records},indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
