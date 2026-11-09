#!/usr/bin/env python3
"""Budgeted cached Astra design consultation; no tools, images or benchmark claims."""
import argparse
import json
from pathlib import Path
from api_budget import guarded
from direct_zones import sha

BRIEF="""Recommend a concrete high-speed CPU network design for artistic photo cropping. Treat every speed/memory estimate as unmeasured. We own Rust SIMD/conv inference infrastructure and can train on a16GB CUDA GPU. Current research student has238629parameters,128x128 image input, shared encoder and aspect-conditioned candidate scoring;128candidate windows/query. CPU inference and repeated aspect requests matter. No API at deployment or routine evaluation. Photos focus on people, groups, pets, scenery and ecommerce; no documents. Purposes catalogue (whole featured product), hero/detail (intentional product detail allowed), editorial. Support minimal maximum-area crop and tighter maximal composition.1:3..3:1 essential,1:6/6:1 secondary. It is legal to choose a primary person/subgroup and wholly exclude other faces. Never slice a face that remains visible; preserve chosen faces. Contain/pad if necessary. Need alternatives representing different coherent choices, not merely left/right extremes. Need preserve text/logos when purpose requires them.
Measured evidence: transfer of person-box auxiliary targets275FIT/91DEV did not improve original crop ranking in12controlled fits. Broader cached FocalNet rank distillation369FIT/116DEV gave new-cohort essential regret0.446518 versus initializer0.585805, but old-cohort regret regressed0.247001->0.320942; no promotion. Both local teacher and student visibly cut faces. EfficientDetLite2 matched227/491person targets vs181/491Lite0 on91DEV images; small-person90/304, unsuitable as complete face coverage. Current local crop bank has coverage gaps. Existing geometry guard is closed without reviewed full-face coverage. Prior proposed segmentation/pose/gaze hints were not sufficient crop-quality supervision.
Now collecting cached Terra source-percent person/face/product/text ROIs and all9ratios x2mode acceptable crop alternatives in one call/source. Independent Sol checks a bounded subset; teachers are weak references, not human gold. Four pilot images show most proposals safe against their own boxes, but one Sol proposal cuts a background face.32new synthetic sources20commerce(80%white,half4:3/3:4) plus12candids; broader real photographic cohort available. User permits speed/accuracy tradeoff if final crop edges within3pixels on normalized128map, but choosing a different valid composition is different from numeric approximation.
Choose an initial architecture and one fallback, explicit operations/tensor shapes, aspect/scale/position handling without stretching, multi-choice prediction, face/subject constraints, loss/optimizer/sampling plan and decisive controls. Compare directKcrop heads versus bank scoring versus small region/subject tokens, and say whether depth, pose, gaze, scanpath, segmentation should be runtime dependencies or offline teachers. Propose an experiment feasible with hundreds of source families; address identifiability, regression of old cohort and minimizing CPU allocations. Distinguish scientific recommendations from measured evidence. Keep answer concise and implementable."""

def request():
    fields=['recommended_architecture','operations_and_shapes','supervision_and_losses','optimizer_and_training','decisive_ablations','runtime_and_limits']
    schema=dict(type='object',properties={key:dict(type='string') for key in fields},required=fields,additionalProperties=False)
    return dict(model='gpt-6-astra',reasoning=dict(effort='medium'),store=False,tools=[],tool_choice='none',max_output_tokens=4096,
        input=[dict(role='developer',content='Give engineering recommendations, not invented measurements. No tools.'),dict(role='user',content=BRIEF)],
        text=dict(format=dict(type='json_schema',name='crop_architecture_advice',strict=True,schema=schema)))

def validate(response,task):
    parts=[]
    for item in response['output']:
        if item['type'] not in ['message','reasoning']:raise ValueError('unexpected tool output')
        for content in item.get('content',[]):
            if content['type']=='refusal':raise ValueError('refused consultation')
            if content['type']=='output_text':parts.append(content['text'])
    result=json.loads(''.join(parts));expected=set(request()['text']['format']['schema']['required'])
    if set(result)!=expected or any(not isinstance(v,str) or not v.strip() for v in result.values()):raise ValueError('complete architecture advice required')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['cache','budget','authorization','output']:p.add_argument(key,type=Path)
    a=p.parse_args();task=dict(id='subject-choice-v4-architecture',role='research-design',image=dict(sha256='text-only-no-image:'+sha(BRIEF.encode())))
    result=guarded(task,request(),validate,a.budget,a.cache,a.authorization,extend_astra=True)
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    if not result['annotation_valid']:raise SystemExit(1)

