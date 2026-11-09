#!/usr/bin/env python3
"""Freeze fixed selections, cached weak references and qualified evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('cohort',type=Path)
    args = p.parse_args()
    if (args.root/'SHA256SUMS.json').exists():
        raise ValueError('packet already frozen')
    if sha(args.cohort)!='a52144833a5824386577f63626e599f8dc602f8b77ee9d4d7f0c183574cebf70':
        raise ValueError('supplementary roles differ')
    plan = json.loads((args.root/'evaluation-plan.json').read_text())
    evaluation = json.loads((args.root/'evaluation.json').read_text())
    analysis = json.loads((args.root/'analysis.json').read_text())
    qualification = json.loads((args.root/'qualification.json').read_text())
    progress = json.loads((args.root/'references/progress.json').read_text())
    if (len(plan['models'])!=48 or evaluation['qualified_models']!=48
            or evaluation['evaluated_sources']!=15 or evaluation['unknown_sources']
            or analysis['valid_reference_boxes']!=23 or qualification['models']!=48
            or qualification['max_map_edge_difference']!=0 or progress['pid'] is not None
            or progress['completed']!=15 or analysis['model_selection_changed']):
        raise ValueError('complete fixed-model qualification and reference results required')
    if any(v['families']!=15 for c in analysis['paired_family_comparisons'].values() for v in c.values()):
        raise ValueError('uncertainty must report actual family count')
    sources = json.loads(args.cohort.read_text())['sources']
    lookup = {s['id']:s for s in sources}
    responses = list((args.root/'references/responses').glob('*.json'))
    if len(responses)!=15:
        raise ValueError('all fifteen full cached responses required')
    seen = set()
    for path in responses:
        row = json.loads(path.read_text())
        contract = row['contract']
        source = lookup[contract['source_id']]
        digest = hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        raw = args.root/'references/inputs'/(source['source']['provider_id']+'.rgb')
        if (digest!=path.stem or digest!=row['contract_sha256'] or row['postprocess_error']
                or row['token_cap_reached'] or not row['ended_with_eos']
                or sha(raw)!=contract['owned_input_sha256']
                or contract['source_plan_sha256']!=sha(args.cohort)):
            raise ValueError('reference contract or input bytes differ')
        seen.add(source['id'])
    if seen!=set(lookup):
        raise ValueError('reference source coverage differs')
    states = args.root/'selected-states'
    states.mkdir()
    copied = []
    for model in plan['models']:
        original = Path(model['checkpoint'])
        if sha(original)!=model['checkpoint_sha256']:
            raise ValueError('pre-reference checkpoint changed')
        name = f"{model['study']}-{model['arm']}-seed{model['seed']}.pt"
        shutil.copy2(original,states/name)
        copied.append(dict(original=str(original),copy='selected-states/'+name,sha256=sha(states/name),
            parent_packet_sha256=model['original_source_packet_sha256']))
    shutil.copy2(args.cohort,args.root/'cohort.json')
    paths = [Path('justfile'),Path('docs/experiments/SUPPLEMENTARY_EXTENT_EVALUATION.md'),
             Path('benchmarks/supplementary_extents_2026-11-09.md')]
    paths += [Path('tools/composition')/n for n in ['artifact_identity.py','probe_florence_regions.py',
        'plan_supplementary_extent_evaluation.py','evaluate_supplementary_extents.py',
        'analyze_supplementary_extents.py','analyze_extent_finetune.py','freeze_supplementary_extents.py']]
    paths += [Path('training')/n for n in ['composition_extent.py','composition_extent_image.py',
        'crop_fit.py','crop_ranker.py','crop_regions.py','local_rank_distillation.py','model_v3ds.py']]
    producers = {}
    for path in paths:
        target = args.root/'source'/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        producers[str(path)] = subprocess.check_output(['git','log','-1','--format=%H','--',str(path)],text=True).strip()
    (args.root/'logs').mkdir()
    for path in Path('/home/lilith/tmp').glob('zensally-supplementary-extent*20261109.log'):
        shutil.copy2(path,args.root/'logs'/path.name)
    manifest = dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        source_producers=producers,selected_states=copied,pre_reference_plan_sha256=sha(args.root/'evaluation-plan.json'),
        source_packet_sha256='20c5c0b330b12357b5e76cfba85f356992b0ca7ee35eea59e3594e6596512d04',
        teacher_packet_sha256='462488a1a7893e6653a1c82e602aeb3362c5a0c8c87cf1a9315adefc66644a9c',
        pretrained_focal_source_packet_sha256='378ec058d53ade02815ca355186578e295b435286e26f51e34f851cf528fc225',
        parent_dependencies='original descriptors, encoder initializers and FocalNet source/weights remain in the fully verified parent packets',
        evaluated_models=48,supplementary_development_sources=15,weak_native_reference_boxes=23,
        training_eligible=False,source_roles_changed=False,new_test_inference=False,paid_model_api_calls=0,
        model_promoted=False,face_coverage_certified=False,human_artistic_quality_measured=False,diagnostic_maps_saved=False)
    (args.root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries = [dict(path=str(f.relative_to(args.root)),bytes=f.stat().st_size,sha256=sha(f))
               for f in sorted(args.root.rglob('*')) if f.is_file()]
    sums = args.root/'SHA256SUMS.json'
    with sums.open('x') as stream:
        stream.write(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(members=len(entries),payload_bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha(sums))))


if __name__ == '__main__':
    main()
