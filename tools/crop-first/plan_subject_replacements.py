#!/usr/bin/env python3
"""Three owner-directed concept/product-geometry replacements with inherited roles."""
import argparse
import json
from pathlib import Path
from sunburst_generate import payload

def plan(source):
    scenes={
        'skincare-person':('720x960','CEVRANEL',
            'Create one photorealistic premium ecommerce skincare photograph on a white seamless studio background. An adult model in a simple cream cotton shirt stands naturally behind a white display plinth. Both hands rest below the visible plinth edge. The complete featured amber dropper bottle and matching ivory carton stand in the foreground. CEVRANEL is clearly readable on both, with the restrained product title Daily Moisture Serum and plausible small cosmetic packaging text. The packaging is about everyday hydration, with no brightening, whitening, before-and-after or treatment message. Natural skin texture, normal proportions, understated expression and soft professional light. Keep the complete face, all bottle and carton corners, bottle base and dropper cap inside the frame. Product and person are separated vertically enough to exercise later crop choices. A believable product campaign with quiet art direction and no conspicuous casting gimmick.'),
        'watch-mature':('720x960','NERALVIQUE',
            'Create one photorealistic premium wristwatch ecommerce portrait on a white seamless studio background. A mature adult man in a gray fine-knit sweater and white cotton shirt sits at a white table, his complete face visible in the upper part of the photograph. His left forearm rests naturally across the foreground tabletop, palm down with relaxed fingers, and the cuff ends above the watch. The featured analog NERALVIQUE watch is worn conventionally around his wrist. Show the watch as a distinct manufactured object sitting on top of the skin: a complete raised metal case, clearly separate case edges and lugs, and two visible brown leather strap segments that emerge from the lugs and visibly curve around the wrist. Include a small contact shadow beneath the raised case and enough side view to see the encircling band. Correct wrist/hand/forearm connections and perspective. The readable watch dial and external leather band are the product focus; the wearer provides natural context. Frame the whole visible case and strap portions, wrist and complete face with breathing room. Useful restrained luxury catalogue photography.'),
        'handbag-square':('832x832','NERALVIQUE',
            'Create one photorealistic premium handbag catalogue photograph on a clean white seamless studio background. An adult wearing a cream blouse and charcoal trousers stands in a relaxed ordinary pose, holding a complete structured plum leather handbag low near the waist by its handles with one hand. The other arm rests naturally at the side. Keep the complete face, natural hands, every bag corner, base, handle and strap attachment inside the square frame. Readable NERALVIQUE foil wordmark on the bag front, plausible stitching and leather grain, quiet soft contact shadows, natural expression and commercially believable styling. The complete bag is the clear featured product; the person gives scale and context. The photograph should feel like a normal carefully made ecommerce image rather than a casting statement.')
    }
    jobs=[]
    for identity,(size,brand,prompt) in scenes.items():
        old=json.loads((source/(identity+'.json')).read_text())
        job=dict(id=identity+'-v2',size=size,prompt=prompt,scenario_family=old['scenario_family'],registered_role=old['registered_role'],
            source_id='sunburst-v4:'+identity+'-v2',replaces_source_id='sunburst-v4:'+identity,replaced_png_sha256=old['png_sha256'],
            people_requested=1,animals_requested=0,brand_requested=brand,background_requested='white',product=old['product'],
            synthetic=True,training_eligible=False,crop_challenge='Believable product/person framing; retain whole selected faces and the complete featured product when catalogue purpose requires it.',
            coordinator_prompt_review='Codex reviewed concept, product/person associations, natural pose and external product geometry; output still requires full-size visual review.')
        payload(job);jobs.append(job)
    return jobs

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    with a.output.open('x') as stream:stream.write(json.dumps(plan(a.source),indent=2)+'\n')
    print(json.dumps(dict(replacements=3,old_sources_retained=True,source_families_and_roles_inherited=True)))

