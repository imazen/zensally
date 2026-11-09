#!/usr/bin/env python3
"""Complete the four pilots to20 commerce and12 varied smartphone sources."""
import argparse
import hashlib
import json
from pathlib import Path
from sunburst_commerce_v2 import COMMERCE,WHITE
from sunburst_subject_pilot import plan as pilots


def plan():
    # Fourteen white and three context sources complement the pilot's2/1 split.
    commerce=[
        ('trainer-shoes','960x720',True,1,'NERALVIQUE','An athletic adult woman with short dark hair stands beside a low white plinth. A complete pair of premium cream training shoes is the featured product; readable NERALVIQUE woven tongue labels face the camera. One shoe worn and one fully displayed, visible soles and toes, her complete face higher and off-center.'),
        ('travel-case','720x960',True,2,'TAVROVELLE','Two adult travelers at different heights beside one complete compact teal rolling suitcase with TAVROVELLE embossed by the handle. Primary older East Asian traveler touches the handle; secondary friend is separated laterally. Complete wheels and top handle, complete faces, plausible bag construction.'),
        ('headphone-sisters','960x720',True,2,'CEVRANEL','Two adult sisters standing at different depths, primary subject wearing premium over-ear headphones with a small readable CEVRANEL headband wordmark, second pair fully displayed on a low plinth. Complete faces and headphone silhouettes, natural smiles, widely separated visual anchors.'),
        ('watch-mature','720x960',True,1,'NERALVIQUE','An older adult with pale skin and white hair holds a wrist at chest height, displaying a complete premium analog wristwatch with NERALVIQUE dial text. Complete face, fingers, wrist and watch case within frame, close enough that branding is useful, visible clothing texture.'),
        ('backpack-parent','960x720',True,2,'TAVROVELLE','An adult parent and teenage child stand apart. Primary parent holds a complete rust-colored technical backpack low at one side; TAVROVELLE patch, zips, straps and all corners visible. Both complete faces, quiet premium studio lighting, parent and product clearly primary.'),
        ('skincare-person','720x960',True,1,'CEVRANEL','Adult woman with vitiligo beside a white plinth carrying a complete amber skincare bottle and matching carton, readable fictional CEVRANEL label with plausible small ingredients typography. Complete face, all bottle/carton edges, primary product in lower third, authentic skin texture.'),
        ('coffee-kit','960x720',True,1,'CEVRANEL','A middle-aged man with a beard stands beside a complete compact espresso machine, matching grinder and branded coffee bag on a white counter. Readable CEVRANEL wordmark on machine and bag. Entire product group fits, complete face offset to left; wide separation between human and machine.'),
        ('sunglasses-friends','720x960',True,2,'NERALVIQUE','Primary adult with tightly curled hair wears premium sunglasses; a second adult sits farther back. A complete spare pair and NERALVIQUE spectacle case lie on a low plinth. Both complete faces, eyewear fits naturally, foreground glasses and case form a clear product story.'),
        ('jacket-wide','1440x480',True,3,'TAVROVELLE','Three adults of different builds spread across a white studio wearing coordinated lightweight jackets. Primary central older woman wears the featured full TAVROVELLE jacket with readable chest patch, full hem and sleeves visible. Two secondary models occupy separate lateral zones. Complete faces and bodies, believable fabric folds.'),
        ('lamp-tall','480x1440',True,1,'CEVRANEL','Complete premium floor lamp from shade to base with small CEVRANEL plaque on its stem, an adult person seated beside it looking upward. Whole face inside frame, complete lamp silhouette and cord, large vertical separation between shade and base. Quiet white studio.'),
        ('handbag-square','832x832',True,1,'NERALVIQUE','An adult woman in a wheelchair with her complete face visible, presenting a complete structured violet handbag on her lap. NERALVIQUE metal wordmark on front, handles and base fully visible. Respectful natural product catalogue image, realistic leather grain and clothing.'),
        ('bicycle-wide','1440x480',True,1,'TAVROVELLE','A complete premium city bicycle side-on with TAVROVELLE readable on frame tube; an adult rider stands near the rear wheel, complete face and hands visible. Entire wheels, basket and handlebar fit in source. Product primary, rider secondary, broad white seamless studio.'),
        ('carryon-tall','480x1440',True,1,'TAVROVELLE','A tall adult model stands naturally beside a complete small hard-shell carry-on suitcase. TAVROVELLE label visible on case, complete wheels and extended handle. Entire face and product fit, generous plain white vertical space, plausible luggage proportions.'),
        ('pet-carrier-square','832x832',True,1,'NERALVIQUE','Adult person with a complete face visible beside a fully visible premium fabric pet carrier with readable NERALVIQUE front label. A small alert terrier looks out through the open mesh door; whole dog face and ears inside source. Complete carrier handle, base and door, natural contact shadows.'),
        ('weekender-station','1440x480',False,2,'TAVROVELLE','Travel campaign in an airy railway concourse. Primary older adult carries a complete TAVROVELLE leather weekender, readable patch and whole bag silhouette. Secondary companion stands well away near a timetable. Complete faces in source, candid interaction and natural window light, luggage remains the primary product.'),
        ('speaker-home','480x1440',False,1,'CEVRANEL','Premium home-audio marketing photograph in a believable small living room. Complete CEVRANEL floorstanding speaker low in frame, adult listener with a complete face seated farther back, soft window light, realistic rug and books. Readable brand plaque, complete speaker base and top, vertical room context.'),
        ('tote-garden','1152x576',False,2,'NERALVIQUE','Premium garden-market marketing shot. Primary adult woman with dark skin and gray braids carries a complete NERALVIQUE canvas tote filled with a few flowers; readable wordmark, handles and bag base visible. A secondary vendor farther away, complete faces, natural ordinary expressions, tactile fabric and warm daylight.'),
    ]
    candid=[
        ('beach-selfie','960x720',2,0,'A slightly tilted close front-camera selfie of two adult friends at a windy public beach, primary person close at left and second farther at right, complete faces and hair inside frame, ordinary sunburn and flyaway hair, cloudy ocean behind.'),
        ('museum-vacation','720x960',4,0,'A parent photographs three relatives during a museum vacation. Four visible complete faces at different distances, primary older visitor foreground lower-left, others behind near a sculpture. Ordinary mixed light, mild phone motion blur, no art-document framing.'),
        ('family-picnic','960x720',5,1,'Unposed family picnic in a public park with five people of different ages and one dog. Primary adult helping a child open a lunch container at front, others spread around the blanket. Complete visible faces and dog head, ordinary food wrappers and uneven midday shade.'),
        ('kitchen-selfie','720x960',3,0,'A messy front-camera kitchen selfie of three adult housemates making breakfast. Primary phone-holder in foreground, two people at different depths behind, all faces complete in original. Ordinary phone HDR, fluorescent and window light, cereal boxes and a dish rack.'),
        ('birthday-outdoors','960x720',7,0,'An amateur backyard birthday phone photograph with seven relatives spread across a picnic table. Primary middle-aged guest laughing beside a cake near left, small background faces at right, complete faces in original, plastic plates, cheap bunting, patchy sun and shade.'),
        ('train-window-selfie','720x960',2,0,'A casual travel selfie of two adult partners seated on a train, primary person nearer the phone and the other partly farther back, both complete faces inside original. One hand holds a paper coffee cup, blurry countryside through window, believable phone camera noise.'),
        ('hilltop-vacation','960x720',4,0,'Four adult friends at different heights on a windy hilltop, primary person foreground with hiking pole, three friends spread behind against broad scenery. Complete faces in original, casual mismatched outdoor clothes, ordinary smartphone snapshot rather than professional landscape.'),
        ('sofa-cat-selfie','720x960',1,1,'An ordinary person taking a relaxed selfie on a worn sofa with a cat sitting beside their shoulder. Complete human face, complete cat head and ears, off-center composition, indoor evening light and mild phone grain, books and laundry in background.'),
        ('pool-holiday','960x720',6,0,'A casual hotel-pool vacation snapshot of six relatives, primary adult in foreground with a child near the edge, four secondary people spread behind at different distances. Whole visible faces in original, natural ordinary bodies, towels and plastic chairs, harsh phone-camera sunlight.'),
        ('market-selfie','720x960',3,0,'Three adult friends taking an informal front-camera selfie at a busy outdoor market, primary person foreground right, two friends at differing depths, all three complete faces and some small bystanders in original. Mild lens distortion, everyday clothing, believable stalls and clutter.'),
        ('grandparent-garden','960x720',3,1,'A grandparent sitting in a modest garden with two younger adult relatives and a medium dog. Primary grandparent near front looking toward camera, others farther apart, all complete visible faces and dog head. Amateur phone snapshot, slightly crooked horizon and realistic uneven light.'),
    ]
    jobs=[]
    for identity,size,white,people,brand,scene in commerce:
        jobs.append(dict(id=identity,size=size,prompt=COMMERCE+(WHITE if white else '')+scene,
            people_requested=people,animals_requested=int(identity=='pet-carrier-square'),brand_requested=brand,
            background_requested='white' if white else 'context',product=identity))
    for identity,size,people,animals,scene in candid:
        jobs.append(dict(id=identity,size=size,prompt='One photorealistic average-person amateur smartphone photograph, natural everyday imperfections, no cinematic treatment, collage or text overlay. '+scene,
            people_requested=people,animals_requested=animals,brand_requested=None,background_requested='context',product='smartphone-candid'))
    for job in jobs:
        family='subject-choice-v4-'+job['id'];bucket=int(hashlib.sha256(family.encode()).hexdigest()[:8],16)%5
        job.update(scenario_family=family,source_id='sunburst-v4:'+job['id'],registered_role='dev' if bucket==0 else 'fit',
            synthetic=True,training_eligible=False,crop_challenge='Choose a coherent primary person or subgroup while preserving whole intersected faces and purposeful product composition.')
    combined=pilots()+jobs;commerce_all=[j for j in combined if j['brand_requested']]
    assert len(jobs)==28 and len(commerce_all)==20 and sum(j['background_requested']=='white' for j in commerce_all)==16
    assert sum(j['size'] in ['960x720','720x960'] for j in commerce_all)==10
    assert len(combined)==32 and len({j['id'] for j in combined})==32
    from sunburst_generate import payload
    for job in jobs:payload(job)
    return jobs


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    with a.output.open('x') as stream:stream.write(json.dumps(plan(),indent=2)+'\n')
    print(json.dumps(dict(new_images=28,total_with_pilot=32,commerce=20,white_commerce=16,commerce_4_by_3_or_3_by_4=10,candids=12)))
