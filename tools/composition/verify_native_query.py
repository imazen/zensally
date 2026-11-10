#!/usr/bin/env python3
"""Measure native query scores and the existing crop-edge/action gate."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from artifact_identity import sha
from query_selection import eligible


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary',type=Path)
    parser.add_argument('fixture',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    manifest=json.loads((args.fixture/'manifest.json').read_text())
    if sha(args.fixture/'inputs.bin')!=manifest['input_sha256'] or sha(args.fixture/'reference.f32')!=manifest['reference_sha256']:
        raise ValueError('immutable native gate input changed')
    expected=np.fromfile(args.fixture/'reference.f32',dtype='<f4').reshape(-1,128)
    reports={}
    for mode in ['ordered','algebraic']:
        output=args.output/(mode+'.f32')
        command=['/usr/bin/time','-v','-o',str(args.output/(mode+'.time')),str(args.binary.resolve()),'score',str(args.fixture/'inputs.bin'),str(output),mode]
        process=subprocess.run(command,capture_output=True,text=True)
        (args.output/(mode+'.stdout')).write_text(process.stdout)
        (args.output/(mode+'.stderr')).write_text(process.stderr)
        if process.returncode:raise ValueError('native query worker failed; retained stdout/stderr')
        worker=json.loads(process.stdout)
        scores=np.fromfile(output,dtype='<f4').reshape(-1,128)
        if scores.shape!=expected.shape or not np.isfinite(scores).all():raise ValueError('invalid native score arrays')
        report=dict(requests=0,choice_changes=0,action_changes=0,edge_movements_over_three=0,
            max_map_pixel_edge_movement=0.,protected_head_violations=0,
            max_abs_score_error=float(np.abs(scores-expected).max()),
            exactly_equal_score_values=int((scores.view(np.uint32)==expected.view(np.uint32)).sum()),
            score_values=int(scores.size),worker=worker)
        rows=[];offset=0
        for source in manifest['sources']:
            path=args.fixture/source['bank_file']
            if sha(path)!=source['bank_sha256']:raise ValueError('native gate bank changed')
            bank=json.loads(path.read_text());width,height=bank['width'],bank['height']
            for task in bank['tasks']:
                for amount in ['minimal','maximal']:
                    indices=np.flatnonzero(eligible(task,width,height,amount))
                    choices=[int(indices[np.argmax(values[offset,indices])]) if len(indices) else None for values in [expected,scores]]
                    changed=choices[0]!=choices[1];action=(choices[0] is None)!=(choices[1] is None)
                    movement=0.
                    if changed and not action:
                        boxes=[task['candidates'][j]['pixel_box'] for j in choices]
                        movement=128*max(abs(a-b)/scale for a,b,scale in zip(*boxes,[width,height,width,height]))
                    if choices[1] is not None:
                        b=task['candidates'][choices[1]]['pixel_box'];p,q=map(int,task['ratio'].split(':'))
                        if not (0<=b[0]<b[2]<=width and 0<=b[1]<b[3]<=height) or (b[2]-b[0])*q!=(b[3]-b[1])*p:
                            raise ValueError('native chosen crop violates source bounds/ratio')
                        report['protected_head_violations']+=int(any(b[0]>h[0] or b[1]>h[1] or b[2]<h[2] or b[3]<h[3] for h in bank['protected_head_pixels']))
                    report['requests']+=1;report['choice_changes']+=int(changed);report['action_changes']+=int(action)
                    report['edge_movements_over_three']+=int(movement>3)
                    report['max_map_pixel_edge_movement']=max(report['max_map_pixel_edge_movement'],movement)
                    rows.append(dict(id=source['id'],ratio=task['ratio'],mode=amount,reference_choice=choices[0],native_choice=choices[1],map_pixel_movement=movement))
                offset+=1
        if offset!=len(scores):raise ValueError('native gate query count differs')
        report.update(passed=not any(report[k] for k in ['action_changes','edge_movements_over_three','protected_head_violations']),
            binary_sha256=sha(args.binary),fixture_manifest_sha256=sha(args.fixture/'manifest.json'),
            reference_choice_fnv64=manifest['reference_choice_fnv64'],face_coverage_certified=False,
            diagnostic_maps_saved=False,quality_improvement_claim=False)
        reports[mode]=report
        (args.output/(mode+'-choices.json')).write_text(json.dumps(rows,indent=2)+'\n')
        (args.output/(mode+'-summary.json')).write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,indent=2))
    if not all(r['passed'] for r in reports.values()):raise ValueError('native runtime failed the unchanged crop-edge/action gate')


if __name__=='__main__':main()
