#!/usr/bin/env python3
"""Eight additional sources using the remaining approved image-generation budget."""
import argparse,json
from pathlib import Path
from sunburst_commerce_v2 import COMMERCE,WHITE

PRODUCTS=[
 ('speaker-pair-white','960x720','NERALVIQUE',0,0,'audio',True,
  'Two complete compact portable speakers, one upright and one laid on its side to reveal buttons and fabric weave, with a small branded charging case. NERALVIQUE wordmark printed clearly on the main speaker grille tab and box. Matte sand and dark olive materials, believable rubber feet and soft studio contact shadows. Full object outlines fit within the source, asymmetric placement and a useful gap between products.'),
 ('tea-bottles-white','720x960','CEVRANEL',0,0,'beverage-packaging',True,
  'Three complete premium cold-brew tea glass bottles at slightly different depths beside an ivory carton, one bottle visibly taller. CEVRANEL navy serif wordmark clearly printed on paper labels and the carton, restrained original line-leaf emblem, honey-colored liquid and believable glass reflections. Retain every bottle cap and base, not a flat label mockup. Products occupy different vertical positions with white breathing room.'),
 ('watch-face-context','1440x480','TAVROVELLE',1,0,'watches',False,
  'Context: a bright ordinary cafe table by a window. An older adult with warm brown skin, close-cropped gray hair and round glasses leans naturally at the table, complete face and head visible, while bringing a wrist with a complete premium analog watch closer to the camera. TAVROVELLE wordmark on the watch dial is useful physical branding, a matching complete presentation box rests lower on the table with larger readable branding. Natural skin, metal reflections and leather strap. Panoramic environmental composition with face, watch and box separated horizontally; do not squeeze everything into one tiny center cluster.'),
 ('dog-harness-white','480x1440','NERALVIQUE',1,1,'pet-products',True,
  'Full-body studio photograph of a stocky adult woman with light freckled skin and short auburn hair, standing beside a medium-sized brown dog wearing an ochre technical walking harness. The complete human head, shoes, dog ears/paws and full harness are visible with white margins. NERALVIQUE wordmark on the harness side patch faces camera and is readable. She holds a plausible leash slack, dressed in everyday charcoal trousers and a blue shirt. Natural person/dog proportions, human face high and dog face much lower.'),
]
SELFIES=[
 ('selfie-solo-bus','720x960',1,
  'An adult woman with dark skin and short natural hair takes a slightly awkward arm-length front-camera selfie on an ordinary city bus. Complete main facial features visible, face near the upper side rather than perfectly centered, shoulder and forearm in the lower foreground. Empty seat backs, rainy windows and small blurry commuters give depth. Flat mixed daylight/interior light, normal skin texture and mild phone noise. Casual jacket, tired half-smile, ordinary imperfect framing. No cinematic lighting or beauty filter.'),
 ('selfie-couple-night','960x720',2,
  'Two adult friends, one with light olive skin and short wavy dark hair and the other with deep brown skin and round glasses, take an arm-length selfie on a busy evening street after dinner. Their full faces are close but at different heights, a little off-center, with normal expression lines and slight hand/camera motion. Warm street lamps and storefront light, mild shadow noise, smaller unrelated people well behind. Casual clothes, uneven exposure, modest front-camera wide-angle perspective, not a staged lifestyle advertisement.'),
 ('selfie-family-hike','720x960',4,
  'A vacation selfie on an ordinary wooded walking trail: an adult with a gray beard close to the phone, another adult with long dark hair a little farther away, and two school-age children lower and behind at distinct distances. All four primary faces fully visible, some looking at the screen and one distracted. Daypacks, crumpled rain jackets, leafy trail context and a dog farther down the path. Slight phone tilt, bright patchy overcast light and honest skin texture, ordinary family snapshot rather than polished campaign work.'),
 ('selfie-mirror-outfit','960x720',1,
  'One young adult with short black curls and warm medium skin takes a casual outfit mirror selfie in a small lived-in bedroom. The phone is held to the side below eye level so it does not cover the complete face. Most of the body, outfit hem and shoes appear inside the mirror, with an ordinary bed, laundry chair and window in the surrounding room. Slightly dusty mirror, mixed window and ceiling light, natural proportions, mild phone sharpening, relaxed unposed expression. One person and their physical mirror reflection, not an extra separate subject.'),
]

def plan():
 jobs=[]
 for identity,size,brand,people,animals,product,white,scene in PRODUCTS:
  jobs.append(dict(id=identity,size=size,prompt=COMMERCE+(WHITE if white else '')+scene,
   scenario_family='products-selfies-v3-'+identity,people_requested=people,animals_requested=animals,
   product=product,brand_requested=brand,background_requested='white' if white else 'context',
   crop_challenge='Keep meaningful product outline, physical text and all main faces across requested ratios.',synthetic=True,training_eligible=False))
 for identity,size,people,scene in SELFIES:
  jobs.append(dict(id=identity,size=size,prompt='Create one photorealistic amateur smartphone selfie in a normal camera-photo aspect ratio, with believable imperfect everyday detail. Single coherent photograph, no collage, no text overlay, no watermark. Small anatomy and object imperfections need no cosmetic polishing; prioritize composition, plausible subject scales and intact main faces.\n'+scene,
   scenario_family='products-selfies-v3-'+identity,people_requested=people,animals_requested=1 if identity=='selfie-family-hike' else 0,
   product='amateur-smartphone-selfie',brand_requested=None,background_requested='context',
   crop_challenge='Near faces, unequal face sizes/heights, foreground phone/arm and background people.',synthetic=True,training_eligible=False))
 return jobs

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
 with a.output.open('x') as f:json.dump(plan(),f,indent=2);f.write('\n')
