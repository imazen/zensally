#!/usr/bin/env python3
"""Bind source-preview observations and report coverage without label admission."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

from artifact_identity import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    args = p.parse_args()
    source_path = args.root/'sources.json'
    data = json.loads(source_path.read_text())
    browser = json.loads((args.root/'browser-unsplash-content.json').read_text())
    with (args.root/'visual-tags.psv').open(newline='') as stream:
        tags = list(csv.DictReader(stream,delimiter='|'))
    if (len(tags)!=119 or browser['decoded']!=119 or browser['exceptions']
            or {int(t['index']) for t in tags}!=set(range(119))):
        raise ValueError('all119 reviewed source indices and browser checks required')
    enums = dict(scene={'product','prepared_food','people','animal','plant_fungus','scenery',
        'architecture_interior','decorative_still_life','apparent_render'},
        people={'none_apparent','partial_body','one','two_to_four','five_plus','uncertain'},
        distance={'none','close','medium','distant','mixed','uncertain'},
        commerce_style={'none','white_light_studio','colored_dark_studio','in_context','uncertain'},
        living_subject={'none','dog','bird','wildlife','insect','flora_fungi','uncertain'},
        rendering={'photo_apparent','uncertain','illustration_or_render_apparent'})
    bound = []
    for tag in tags:
        source = data['sources'][int(tag['index'])]
        if (source['index']!=int(tag['index']) or source['role'] not in ['fit','dev']
                or sha(args.root/source['preview'])!=source['preview_sha256']):
            raise ValueError('review/source preview identity differs')
        if any(tag[key] not in values for key,values in enums.items()) or not tag['note']:
            raise ValueError('invalid or incomplete content observation')
        bound.append(dict(tag,id=source['id'],role=source['role'],family_id=source['family_id'],
            rgb_sha256=source['rgb_sha256'],preview_sha256=source['preview_sha256'],
            source_url=source['source']['source_url'],photographer=source['source']['photographer'],
            reviewer='Codex visual source-preview inspection in the active conversation',
            human_rated=False,training_eligible=False,face_coverage_certified=False))
    coverage = {}
    for role in ['fit','dev','all']:
        rows = [r for r in bound if role=='all' or r['role']==role]
        coverage[role] = dict(sources=len(rows),families=len({r['family_id'] for r in rows}),
            counts={key:dict(Counter(r[key] for r in rows)) for key in enums})
    ratios = [s['native_size'][0]/s['native_size'][1] for s in data['sources']]
    result = dict(observations=bound,coverage=coverage,
        native_aspects=dict(minimum=min(ratios),maximum=max(ratios),median=statistics.median(ratios),
            at_least_3_to_1=sum(r>=3 for r in ratios),at_most_1_to_3=sum(r<=1/3 for r in ratios)),
        sources_sha256=sha(source_path),raw_tags_sha256=sha(args.root/'visual-tags.psv'),
        preview_long_edge=320,observed_absence_is_not_certified=True,
        interpretation='broad machine content observations; no face completeness, crop quality or training supervision',
        source_roles_changed=False,training_admission_changed=False,test_sources=0,paid_model_api_calls=0)
    with (args.root/'content-review.json').open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(coverage=coverage,native_aspects=result['native_aspects']),indent=2))


if __name__ == '__main__':
    main()
