#!/usr/bin/env python3
"""Render frozen retention decisions and measure clipping without changing gates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import statistics

import torch

from artifact_identity import sha
from region_retention import retained_fraction


def severity(crop,regions,width,height):
    loss=max(max(0,crop[0]-b[0])*128/width for b in regions)
    loss=max(loss,max(max(0,crop[1]-b[1])*128/height for b in regions))
    loss=max(loss,max(max(0,b[2]-crop[2])*128/width for b in regions))
    return max(loss,max(max(0,b[3]-crop[3])*128/height for b in regions))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    torch.set_num_threads(1)
    analysis=json.loads((a.root/'analysis.json').read_text())
    data=torch.load(a.root/'inputs.pt',map_location='cpu',weights_only=False)
    receipts={r['id']:r for r in data['reference_receipts']}
    lookup={s['id']:i for i,s in enumerate(data['sources'])}
    review=a.root/'review';review.mkdir();(review/'images').mkdir()
    sources=[];image_bytes=0
    for source in data['sources']:
        path=Path(source['original_source_record']['directory'])/'preview.png'
        if sha(path)!=source['preview_sha256']:raise ValueError('owned preview differs')
        target=review/'images'/(source['source']['provider_id']+'.png');shutil.copy2(path,target);image_bytes+=target.stat().st_size
        sources.append(dict(id=source['id'],family_id=source['family_id'],role=source['role'],native_size=source['native_size'],
            working_size=source['working_size'],preview=str(target.relative_to(review)),
            source_url=source['source']['source_url'],provider_url=source['source']['original_provider_url'],
            regions=receipts[source['id']]['boxes'],content_note=source['content_note']))
    if image_bytes>32*1024**2:raise ValueError('bounded preview allowance exceeded')
    models=[];summaries=[]
    for model in analysis['models']:
        rows=[];violations=[]
        for row in model['predictions']:
            i=lookup[row['id']];w,h=data['sources'][i]['native_size'];regions=receipts[row['id']]['boxes']
            if row['offered']:
                crop=data['native_banks'][i][row['query']//2][row['selected']]
                fraction=min(retained_fraction(crop,b) for b in regions)
                full=all(crop[0]<=b[0] and crop[1]<=b[1] and crop[2]>=b[2] and crop[3]>=b[3] for b in regions)
                loss=severity(crop,regions,w,h)
                if full!=row['known_boxes_contained'] or abs(fraction-row['retained_fraction'])>1e-6:raise ValueError('stored/native retained-region geometry differs')
                if not full:violations.append(dict(edge_loss_map128=loss,fraction=fraction,scope=row['scope']))
            else:crop=[0,0,w,h];fraction=1.;loss=0.
            rows.append(dict(row,crop_pixels=crop,edge_loss_map128=loss))
        summaries.append(dict(arm=model['arm'],seed=model['seed'],offered=sum(r['offered'] for r in rows),
            known_box_violations=len(violations),violation_edge_loss_median=statistics.median(r['edge_loss_map128'] for r in violations) if violations else None,
            violation_edge_loss_max=max((r['edge_loss_map128'] for r in violations),default=None),
            violations_within_3_map_edge_loss=sum(r['edge_loss_map128']<=3 for r in violations),
            zero_reference_retention=sum(r['fraction']==0 for r in violations),
            violation_retained_fraction_median=statistics.median(r['fraction'] for r in violations) if violations else None))
        models.append(dict(key=model['arm']+'-'+str(model['seed']),arm=model['arm'],seed=model['seed'],rows=rows))
    value=dict(sources=sources,models=models,ratios=data['ratios'],image_bytes=image_bytes,
        original_scorer_max_error=analysis['original_scorer_max_error'],face_coverage_certified=False,
        note='Owned working previews show exact native crop geometry. Returned subject boxes are incomplete weak references. Containment and face gates are unchanged.')
    (review/'data.json').write_text(json.dumps(value,indent=2)+'\n')
    (review/'severity.json').write_text(json.dumps(dict(models=summaries,diagnostic_only=True,gate_relaxed=False,
        units='maximum lost returned-box edge in128*x/native_width or128*y/native_height coordinates'),indent=2)+'\n')
    template=Path(__file__).with_name('retention_review.html').read_text()
    (review/'index.html').write_text(template.replace('/*DATA*/null',json.dumps(value).replace('</','<\\/')))
    print(json.dumps(dict(sources=len(sources),models=len(models),requests=sum(len(m['rows']) for m in models),image_bytes=image_bytes,severity=summaries),indent=2))


if __name__=='__main__':main()
