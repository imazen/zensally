#!/usr/bin/env python3
"""Bind coarse, authored photo-subject queries to frozen fit/development sources."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from artifact_identity import sha


# Indices name the immutable content review, never detector outcomes or TEST.
# These are authored weak-query subjects, not verified labels or complete faces.
PHRASES = {
    0:'The seated reader.',2:'The food jar.',3:'The food jar.',
    4:'The wooden spoons with ingredients.',5:'The bowl of nuts.',6:'The drink bottles.',
    7:'The bottled drink.',8:'The smartphone.',9:'The walking person.',11:'The beverage bottle.',
    12:'The dog.',13:'The pancakes and jam jar.',14:'The burger.',17:'The pedestrians.',
    18:'The red bottle.',19:'The laptop and coffee cup.',20:'The essential-oil bottles.',
    22:'The headphones.',23:'The seated reader.',24:'The sunglasses.',25:'The candle jar.',
    26:'The dragonfly.',27:'The swimmer.',28:'The camera.',29:'The smartwatch.',31:'The fox.',
    32:'The skincare packages.',34:'The instant camera.',35:'The hanging boots.',37:'The shoes.',
    38:'The candle jar.',39:'The reclining person.',40:'The sunglasses.',41:'The seated people.',
    42:'The seated people.',43:'The reader on the windowsill.',44:'The shoe.',
    46:'The perfume bottle.',48:'The reader at the counter.',49:'The people at the beach cafe.',
    50:'The perfume bottle.',51:'The sunglasses.',52:'The bath-bomb cosmetics.',
    53:'The honey jar and food package.',54:'The sandals.',57:'The skincare bottle.',
    58:'The skincare bottle.',59:'The skincare mask package.',60:'The dropper bottle.',
    61:'The person behind the candles.',67:'The cosmetic containers.',68:'The standing person.',
    70:'The perfume bottle.',71:'The person in the passage.',73:'The beverage carton.',
    75:'The skincare products.',76:'The parrot.',77:'The earbuds and their case.',
    79:'The smartphone and bottles.',83:'The headset and camera accessories.',
    85:'The people by the waterfront.',86:'The dog.',
    88:'The skincare and makeup products.',89:'The bodywash bottle.',90:'The coffee cup and cookies.',
    91:'The red shoe.',92:'The brownie on the plate.',93:'The cookies on the plate.',
    94:'The vintage camera.',95:'The cream tubes.',96:'The dropper bottle.',97:'The camera.',
    98:'The perfume bottle.',99:'The wine glass.',100:'The leather wallet.',
    103:'The latte.',104:'The pumpkins and gourds.',105:'The pumpkins and gourds.',
    107:'The people on the bridge.',109:'The bartender and customer.',110:'The deer.',
    111:'The person with the phone.',113:'The leather backpack.',114:'The shoes.',
    115:'The perfume bottles.',117:'The person wearing a hat.',118:'The person photographing the city.',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['registry','review','output']:
        p.add_argument(name,type=Path)
    p.add_argument('--people-only',action='store_true')
    args = p.parse_args()
    if sha(args.registry)!='6f8d318cc870248dcc2400a235fcbaa5f7f0edf47d575d965bf5c72b5684d2f6':
        raise ValueError('photographic roles differ')
    review = json.loads(args.review.read_text())
    if sha(args.review)!='2f20aa9d3d75caa8d397c9d425eab5fe6e32144b7f9713fe2ac38ffab4d618bb':
        raise ValueError('frozen source-content review differs')
    records = [r for r in review['observations'] if r['scene'] in ['product','people','animal','prepared_food']]
    if len(records)!=87 or {int(r['index']) for r in records}!=set(PHRASES):
        raise ValueError('authored query coverage differs from frozen subject scope')
    lookup = {s['id']:s for s in json.loads(args.registry.read_text())['sources'] if s['role'] in ['fit','dev']}
    sources = []
    for row in records:
        source = lookup[row['id']]
        if source['rgb_sha256']!=row['rgb_sha256'] or source['role']!=row['role'] or source['family_id']!=row['family_id']:
            raise ValueError('authored subject/source identity differs')
        sources.append(dict(source,query=PHRASES[int(row['index'])],query_category=row['scene'],
            query_authorship='Codex coarse subject phrase from frozen source-preview observations',
            query_index=int(row['index']),training_eligible=False,face_coverage_certified=False))
    counts = Counter(s['role'] for s in sources)
    if counts!=dict(fit=72,dev=15) or {s['family_id'] for s in sources if s['role']=='fit'} & {s['family_id'] for s in sources if s['role']=='dev'}:
        raise ValueError('subject-query family-role contract differs')
    if args.people_only:
        sources = [dict(s,query='The people.') for s in sources if s['query_category']=='people']
        counts = Counter(s['role'] for s in sources)
        if len(sources)!=20 or counts!=dict(fit=18,dev=2):
            raise ValueError('generic query must cover all twenty registered people scenes')
    result = dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        sources=sources,registry_sha256=sha(args.registry),content_review_sha256=sha(args.review),
        fit_sources=counts['fit'],development_sources=counts['dev'],query_subjects=len(sources),
        other_subjects_unknown=119-len(sources),generic_people_followup=args.people_only,
        source_roles_changed=False,teacher_labels_admitted=False,face_coverage_certified=False,
        artistic_quality_labels=False,new_test_inference=False,paid_model_api_calls=0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(sources=len(sources),by_role=dict(counts),by_category=dict(Counter(s['query_category'] for s in sources)),
        plan_sha256=sha(args.output))))


if __name__ == '__main__':
    main()
