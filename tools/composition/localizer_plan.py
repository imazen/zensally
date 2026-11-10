#!/usr/bin/env python3
"""Freeze primary-subject queries before full synthetic-corpus local inference."""
import argparse,json
from pathlib import Path

QUERIES={
 'outerwear-ridge-wide':'a rain jacket.', 'bootmaker-tall':'a leather boot.',
 'skincare-product-and-face':'a serum bottle.', 'adaptive-travel-portrait':'a backpack.',
 'ceramics-square':'a ceramic mug.', 'ceramics-wide':'a ceramic mug.',
 'garden-table-family-wide':'a garden table. a garden bench.', 'pet-bed-tall':'a dog bed. a dog.',
 'coffee-maker-two-workers':'an espresso machine.', 'floor-lamp-portrait':'a floor lamp.',
 'cycling-depth-wide':'a bicycle.', 'jewelry-profile-tall':'an earring. a jewelry box.',
 'luggage-family-wide':'a suitcase.', 'leather-bag-square':'a leather handbag.',
 'picnic-hamper-group-wide':'a picnic basket.', 'trail-layering-tall':'a jacket. a backpack. a dog.',
 'serum-and-carton':'a serum bottle. a skincare carton.', 'leather-tote-model':'a leather handbag.',
 'headphones-case':'headphones. a headphones case.', 'linen-shirt-model':'a shirt.',
 'coffee-machine':'a coffee machine.', 'running-shoes':'running shoes.',
 'hand-cream-use':'a hand cream tube.', 'carry-on-suitcase':'a suitcase.',
 'dog-bed':'a dog bed. a dog.', 'table-lamp':'a table lamp.',
 'sneaker-fit-wide':'sneakers.', 'skincare-set-wide':'skincare bottles. skincare jars. skincare packaging.',
 'floor-lamp-tall':'a floor lamp.', 'outdoor-pack-tall':'a backpack.',
 'jewelry-box-square':'jewelry. a jewelry box.', 'tableware-square':'dishes. a bowl. plates.',
 'garden-furniture-context':'a garden table. a garden bench.', 'travel-case-context':'a suitcase.',
 'coffee-bar-context':'a coffee machine.', 'jacket-context':'a jacket. a backpack. a dog.',
 'birthday-cake-kitchen':'a birthday cake.', 'birthday-grandparent':'a birthday cake.',
 'birthday-garden':'a birthday cake.', 'birthday-blowing-candles':'a birthday cake.',
 'birthday-living-room':'a birthday cake.', 'birthday-park-picnic':'a birthday cake.',
 'vacation-beach':'a person.', 'vacation-hill-view':'a person.',
 'vacation-museum-cafe':'a person. an ice cream cone.', 'vacation-campsite':'a person. a tent.',
 'vacation-city-square':'a person.', 'vacation-rainy-window':'a person. a board game.',
 'speaker-pair-white':'speakers. a charging case.', 'tea-bottles-white':'bottles. a carton.',
 'watch-face-context':'a wristwatch. a watch box.', 'dog-harness-white':'a dog. a dog harness.',
 'selfie-solo-bus':'a person.', 'selfie-couple-night':'a person.',
 'selfie-family-hike':'a person. a dog.', 'selfie-mirror-outfit':'a person.',
}
p=argparse.ArgumentParser(description=__doc__);p.add_argument('first_gate',type=Path);p.add_argument('output',type=Path);a=p.parse_args();ids={s['id'] for s in json.loads((a.first_gate/'inputs.json').read_text())['images'] if s['synthetic']}
if ids!=QUERIES.keys():raise ValueError('primary query coverage differs from frozen 56 sources')
with a.output.open('x') as f:json.dump({'schema':1,'queries':QUERIES,'scope':'primary product/story extents from authored source briefs; text-conditioned weak supervision, missing or partial detections do not certify absence/completeness','threshold':.3,'text_threshold':.25,'human_rated':False,'face_coverage_certified':False},f,indent=2);f.write('\n')
print(json.dumps({'queries':len(QUERIES)}))
