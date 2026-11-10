#!/usr/bin/env python3
"""Four varied subject-choice witnesses before the larger authorized batch."""
import argparse
import json
from pathlib import Path

from sunburst_commerce_v2 import COMMERCE,WHITE


def plan():
    scenes=[
        ('two-models-tote-white','1440x480','NERALVIQUE',2,True,
         'A full-product studio fashion photograph of two adult friends standing well apart across a white seamless studio. One older woman with silver curls and deep brown skin carries the complete featured structured ochre leather tote; another younger adult with light freckled skin stands farther away holding a small matching card case. The featured tote has a readable debossed NERALVIQUE wordmark, realistic seams and handles, and its complete silhouette is visible. Both complete faces and full bodies fit naturally in the source with room around the edges; faces at different heights and products lower. The tote-carrying woman is the clear primary subject, with the friend secondary. Premium useful catalogue image, plausible ordinary proportions, soft contact shadows.'),
        ('person-speakers-white','720x960','CEVRANEL',1,True,
         'An adult man with warm olive skin and short dark curls sits beside a clean white display plinth on a seamless white background. His complete face is in the upper part of the frame; the entire featured pair of compact premium home speakers and their small product carton rest lower on the plinth. CEVRANEL wordmark is printed on a physically plausible fabric grille tab and clearly on the carton. The man lightly touches one speaker and looks down toward it. White shirt, natural skin texture and fabric weave, complete product corners and bases, quiet soft contact shadows. Product and face are separated vertically, without a central tiny cluster.'),
        ('friends-camera-context','960x720','TAVROVELLE',3,False,
         'A premium camera-bag marketing photograph at an outdoor market cafe. A foreground adult with dark brown skin and short natural hair carries the entire featured compact canvas camera bag, with TAVROVELLE readable on the front leather patch. Two friends at different depths behind are secondary, one seated and one standing, with all three complete faces visible. The foreground person and branded bag are the primary story, the friends supply context and need not all appear in every later crop. Natural morning light, ordinary expressions, worn wood table, believable canvas grain and strap construction. Keep the full primary bag outline, strap attachment points and all source faces inside the photograph.'),
        ('birthday-choice-candid','720x960',None,6,False,
         'Photorealistic ordinary amateur smartphone birthday photograph in a modest lived-in kitchen. An older adult with medium brown skin and gray hair is the clear birthday guest of honor, leaning toward a small cake at the near side of the table; two laughing adults stand nearby and three relatives sit or stand farther back, at varied heights and distances. All six visible faces are complete in the source, but widely spread so later tall or wide crops will sometimes have to choose a coherent person or subgroup. Mismatched cups, supermarket cake packaging, crumpled gift wrap, natural mixed window and ceiling light, a slightly tilted phone, mild phone HDR and noise. Unposed everyday moment, realistic skin and clothing, no cinematic treatment, text overlays or collage.'),
    ]
    jobs=[]
    for identity,size,brand,people,white,scene in scenes:
        prefix=(COMMERCE+(WHITE if white else '')) if brand else 'Create one photorealistic amateur smartphone candid, not a posed advertising scene. '
        jobs.append(dict(id=identity,size=size,prompt=prefix+scene,scenario_family='subject-choice-v4-'+identity,
            people_requested=people,animals_requested=0,product='featured-product-with-people' if brand else 'amateur-birthday',
            brand_requested=brand,background_requested='white' if white else 'context',
            crop_challenge='Choose a coherent primary person/subgroup; retain whole intersected faces and requested product details.',
            synthetic=True,training_eligible=False))
    return jobs


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    with a.output.open('x') as stream:stream.write(json.dumps(plan(),indent=2)+'\n')
