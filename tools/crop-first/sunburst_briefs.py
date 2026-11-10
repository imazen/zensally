#!/usr/bin/env python3
"""Authored ecommerce photo briefs; synthetic exploration, not admitted labels."""
import argparse
import json
from pathlib import Path

COMMON = '''Use case: photorealistic-natural / premium ecommerce campaign photography.
Deliver one coherent photograph, suitable as the source for product, hero, or thumbnail layouts. Original fictitious people and unbranded products; not a collage, illustration, contact sheet, CGI showroom, or webpage mockup. Photorealistic skin with pores and natural age detail, believable hands and product geometry, tactile fabrics and materials, restrained commercial color grading. Compose for the requested native aspect ratio with every person's face/head inside the source frame and natural breathing room. Keep the featured product recognizable and its relationship to people visually clear. Preserve intentional environmental context and useful negative space; do not put every subject in the center or force blank space into every photograph. No text overlays, watermarks, crop marks, interface elements, or invented logos.
'''

SCENES = [
    ('outerwear-ridge-wide', '3072x1024', 3, 0, 'outerwear',
     'Three people across width and depth; product identity versus scenery; distant face.',
     '''Asset: panoramic outdoor apparel hero photograph.
Scene: a coastal hiking ridge after rain, sea haze and layered grassland, overcast daylight with a narrow warm break in the clouds.
Subjects: an older woman with short silver hair wearing a dark navy technical rain shell stands in the nearer left area fastening her cuff. A tall dark-skinned man in an olive shell stands farther to the right looking toward her. A third adult in a rust-colored shell walks on the trail in the middle distance, clearly a smaller person rather than a miniature duplicate. All three heads remain distinct, with realistic relative sizes and unhurried gestures. Show the near garments' seam tape, matte woven texture, zippers, and subtle water beads.
Composition: an eye-level environmental campaign photograph with the people separated naturally by the trail; the landscape links them and provides calm sky for possible hero copy. Retain feet and the trail's direction. Do not manufacture a dramatic event or pack the figures shoulder-to-shoulder.'''),
    ('bootmaker-tall', '1024x3072', 1, 0, 'footwear',
     'Face, hands, and featured boot separated vertically; tight crop may lose the story.',
     '''Asset: vertical luxury footwear craftsmanship campaign photograph.
Scene: a quiet bootmaking workshop with tall north-facing windows, worn oak workbench and softly receding tools.
Subject: an older male artisan with a salt-and-pepper beard stands behind the workbench, naturally looking down while holding one finished chestnut leather hiking boot. His whole head sits in the upper part of the composition, his hands and the complete boot are in the middle/lower part, with a second matching boot on the bench. The featured boot has visible hand stitching, brass eyelets, waxed laces, a rugged sole and softly burnished leather, with anatomically credible fingers supporting it.
Composition: environmental portrait, approximately 50mm photographic perspective; include the apron, lower bench and enough workshop depth to explain the craft. Soft side light, controlled leather highlights, gentle grain. The boot is the commercial subject while the artisan's expression and hands give it meaning.'''),
    ('skincare-product-and-face', '1536x1152', 1, 0, 'skincare',
     'Small foreground bottle and face are jointly important.',
     '''Asset: high-end skincare product-in-use campaign photograph.
Scene: a warm limestone bathroom vanity, soft window light, understated towels and a softly blurred mirror edge.
Subject: a middle-aged dark-skinned woman with short natural curls and subtle gray strands wears a simple cream robe. In a relaxed three-quarter portrait, she holds a small frosted amber serum bottle beside, but not in front of, her cheek. Her full face is visible. The bottle is unbranded, with a fine brushed-metal dropper collar and a blank ivory label; a second closed bottle stands lower on the counter. Render real pores, fine expression lines, translucent glass, liquid refraction and gentle contact shadows.
Composition: medium-close commercial photograph with a clear face/product relationship and some quiet stone surface; keep both the held bottle and the face in focus enough to read their forms. No over-retouched plastic skin, decorative typography or cosmetic before-and-after panels.'''),
    ('adaptive-travel-portrait', '1152x1536', 1, 0, 'travel-accessories',
     'Person, wheelchair, backpack and hands should remain meaningful together.',
     '''Asset: inclusive premium travel-accessory lifestyle photograph.
Scene: a sunlit contemporary hotel entrance with pale stone, glazing and a few softly receding plants.
Subject: a stylish adult wheelchair user with warm brown skin and short dark hair pauses to adjust the shoulder strap of a compact charcoal travel backpack. The backpack rests against their lap with its full silhouette visible, tactile recycled fabric, neat zipper pulls and a small blank leather patch. Their hand naturally grips the strap. Show their complete head, torso, chair wheels and footrest, with a small carry-on case beside the chair.
Composition: vertical, eye-level environmental portrait with relaxed confident expression, natural body proportions, detailed woven clothing and soft late-afternoon light. Commercial emphasis on independent everyday travel and thoughtful product design; avoid a staged medical setting, exaggerated sentimentality or cropping the mobility context out of the source.'''),
    ('ceramics-square', '1024x1024', 2, 0, 'ceramics',
     'Price/aspect probe with the same authored prompt as ceramics-wide; shared scenario family.',
     '''Asset: artisanal tableware ecommerce lifestyle photograph.
Scene: a bright pottery showroom with plaster walls, one long timber display table and a broad window.
Subjects: a young adult potter in a textured linen shirt gently offers a handmade speckled ceramic mug toward the camera, holding its handle naturally. Their full face and the entire mug are visible, with the mug sufficiently large to show glaze variation, a thin irregular rim, tiny pinholes and a believable handle. A second adult visitor stands farther back by the window, a clearly smaller complete face and softly readable figure. Show a small cluster of matching bowls on the table without obscuring the featured mug.
Composition: a loose, natural editorial product photograph rather than a centered catalog cutout; preserve the relationship between the potter, the mug, the visitor and the room. Soft daylight, warm neutral materials, realistic ceramic highlights and gentle background falloff. Adapt the available environmental space naturally to the canvas.'''),
    ('ceramics-wide', '2048x1024', 2, 0, 'ceramics',
     'Same-prompt aspect pricing probe; distant visitor cannot silently disappear.', None),
    ('garden-table-family-wide', '3072x1024', 4, 1, 'outdoor-furniture',
     'Separated faces and dog; wide table is the product, not merely faces.',
     '''Asset: panoramic premium outdoor-furniture campaign photograph.
Scene: a planted urban courtyard with a broad teak dining table, linen cushions, pale paving and dappled late-afternoon shade.
Subjects: four family members with varied visible ages and skin tones gather naturally around the table: an older man at one end, a middle-aged woman carrying a ceramic serving bowl, a young adult seated at the far side and a school-age child standing near the other end. All four complete heads are visible and spatially separated. A medium-sized dog rests beside the near bench with its face unobscured. Their ordinary gestures and expressions feel candid, not like a synchronized pose.
Composition: a wide environmental photograph showing the full table silhouette, its joinery and grain, chair spacing and believable family use. The furniture is the saleable subject and the people explain scale and use. Give the end figures breathing room and keep the courtyard architecture coherent. No in-image lettering.'''),
    ('pet-bed-tall', '1024x3072', 1, 1, 'pet-products',
     'Human face near top, dog face and full bed lower down.',
     '''Asset: vertical premium pet-bed lifestyle photograph.
Scene: an airy living room beside a tall window, a wool rug, warm oak floor and quiet greenery.
Subjects: an older woman wearing a softly textured cardigan kneels beside a large sage-green woven dog bed, smiling gently toward a mature golden retriever resting in it. Her complete head is higher in the frame; the dog's complete face, ears and front paws sit substantially lower. Show the whole oval bed, its raised bolster, removable fabric cover and fine stitched seams. One hand rests naturally on the bolster without hiding the dog.
Composition: vertical environmental photograph at a low conversational viewpoint, generous room above and below the two faces, soft morning light and realistic fur detail. The bed remains the primary commercial object while the human/animal relationship supplies the story. Avoid giant eyes, a toy-like dog or implausibly tiny furniture.'''),
    ('coffee-maker-two-workers', '1536x1152', 2, 0, 'coffee-equipment',
     'Two faces at different depths, reflective machine and hand action.',
     '''Asset: premium home-and-small-business espresso-equipment campaign photograph.
Scene: a refined neighborhood coffee bar with walnut cabinetry, textured plaster and a window casting soft raking light.
Subjects: two adult baristas, one nearer the camera wearing an understated olive apron and one farther behind the counter. The nearer barista naturally locks a portafilter into an unbranded brushed-stainless espresso machine while glancing toward their colleague. Both complete faces remain visible. Show the machine's full main body, steam wand, cup tray, credible knobs and a small white ceramic cup; preserve believable metal reflections rather than mirror-like distortion.
Composition: a three-quarter environmental product photograph, with the machine and working hand clearly readable and the second person smaller but recognizable as a person. Include useful counter context and some quiet wall area. Natural skin, fabric weave, fine coffee dust and restrained highlights.'''),
    ('floor-lamp-portrait', '1152x1536', 1, 0, 'home-lighting',
     'Tall product, face, adjusting hand and base cannot be collapsed into one face crop.',
     '''Asset: premium lighting ecommerce lifestyle portrait.
Scene: an inviting modern reading corner with a low upholstered chair, pale textured wall, a book and a tall floor lamp.
Subject: an adult with dark curly hair wearing a soft charcoal knit stands beside the lamp and adjusts its small brass switch with one hand. Their whole face is visible in profile-to-three-quarter view. The lamp has a warm ivory linen shade, a slender brushed-brass stem and a weighted dark stone base, all fully visible inside the source. Show the hand contacting the actual switch at a plausible height.
Composition: a vertical full-body environmental photograph with natural headroom, feet, lamp shade and base included. Mix soft window light with warm lamp glow, keeping materials tactile and proportions believable. The lamp is the product; the person gives scale and demonstrates use. No text, catalog diagram or duplicated lamp.'''),
    ('cycling-depth-wide', '3072x1024', 4, 0, 'cycling',
     'Near cyclist and three smaller background people; bicycle wheels and helmets.',
     '''Asset: panoramic premium commuter-cycle apparel and equipment photograph.
Scene: a broad park path beside a river in early autumn, soft bright overcast weather and layered trees.
Subjects: four adult cyclists at believable staggered distances. A nearer rider in a deep-blue technical jacket pauses at the side, one hand naturally on the handlebar of a complete unbranded city bicycle. A second rider is farther to the opposite side; two companions are smaller on the receding path. Each wears a realistic helmet with their visible face kept inside the frame; no helmet becomes an extra head. Include the near bicycle's full wheels, frame triangle, saddle and naturally aligned fork.
Composition: coherent environmental campaign photography with depth, differentiated subject scales and room around the near rider; a 35mm-like viewpoint without extreme distortion. Crisp garment texture, believable spokes and tire contact shadows. Let trees and river remain useful context rather than an empty artificial backdrop.'''),
    ('jewelry-profile-tall', '1024x3072', 1, 0, 'jewelry',
     'Full profile face and tiny valuable earring, hands and presentation box.',
     '''Asset: vertical fine-jewelry ecommerce campaign photograph.
Scene: a calm dressing area with a softly lit neutral wall and a narrow walnut vanity.
Subject: an adult woman with natural fine expression lines turns in a relaxed three-quarter/profile view while checking a small sculptural gold earring. Show her complete face and head, a natural hand just below the ear without covering the earring, and a slim velvet presentation box held lower near the waist. The earring has a quiet hammered surface and a tiny green stone; the box is unbranded with a blank interior. Render actual skin texture, loose hair strands, fingernails and controlled gold highlights.
Composition: a deliberately tall, elegant environmental portrait, with the face, jewelry and hands distributed vertically and breathing room around the silhouette. Soft directional daylight, realistic proportions and restrained retouching. Avoid floating earrings, extra fingers, fantasy gemstone sparkle or typography.'''),
    ('luggage-family-wide', '2048x1024', 4, 0, 'luggage',
     'Four separated faces, different heights, multiple relevant suitcases.',
     '''Asset: premium luggage ecommerce travel photograph.
Scene: a quiet contemporary railway-station concourse with tall glazing, warm stone and soft morning light, no readable signage.
Subjects: two adults and two school-age children stand naturally at varied positions around a coordinated set of unbranded graphite hard-shell luggage. One adult extends a telescoping handle; another looks toward the smaller child while the older child rests a hand on a compact carry-on. All four complete heads are visible at believable different heights. Show the cases' ribbing, clean corners, small spinner wheels, zippers and tactile handles, with natural hand contact.
Composition: an eye-level wide lifestyle photograph showing the product set and relationships, enough concourse to communicate travel, and modest negative space that could take page copy. Avoid a rigid lineup, obvious stock-photo grins, missing wheels, airport-security drama or invented brand lettering.'''),
    ('leather-bag-square', '1024x1024', 0, 0, 'bags',
     'Dark low-contrast product, handles and closure matter more than bright background.',
     '''Asset: premium leather-goods product detail photograph for an ecommerce category thumbnail and product page.
Scene: a charcoal stone shelf against a softly textured warm-gray studio background, with a quiet sweep of window light.
Subject: one complete unbranded dark espresso leather tote at a three-quarter angle, both handles visible, soft but structured side panels, fine saddle stitching and a small brushed-brass closure. Beside it, place a neatly rolled matching strap and one folded piece of natural linen to establish material scale without distracting. The entire bag silhouette and base remain inside the frame.
Composition: a sophisticated square product photograph with purposeful asymmetric balance and generous but not excessive breathing room, tactile leather grain, believable creasing and subtle contact shadow. Use controlled grazing highlights so the dark product reads against the dark setting; do not turn it into a white-background cutout. No human figure, letters or logo.'''),
    ('picnic-hamper-group-wide', '3072x1024', 6, 0, 'picnic-and-food',
     'Six faces, foreground product and background faces in separate clusters.',
     '''Asset: panoramic premium picnic-hamper and outdoor-tableware campaign photograph.
Scene: a riverside meadow with a low picnic setting, a linen blanket, late-summer trees and soft golden side light.
Subjects: six people spanning young adult through older adult ages, varied visible skin tones and everyday summer clothing, gathered in two loose conversational clusters rather than one centered row. A wicker picnic hamper with leather straps, two ceramic serving plates and clear glass bottles occupy the nearer foreground. One person reaches naturally to pass a bowl; two people stand farther back near the tree line, smaller complete faces but clearly part of the group. Keep all six heads inside the source frame.
Composition: a single plausible, relaxed commercial lifestyle photograph with the hamper and tableware clearly identifiable, genuine distance relationships and ample environmental context. Detailed wicker, linen, food surfaces, natural hands and realistic shadow direction. Avoid a banquet table, collage, duplicated faces or banner text.'''),
    ('trail-layering-tall', '1024x3072', 1, 1, 'outdoor-apparel',
     'Full-body apparel, head, boots and a smaller dog in a vertical environment.',
     '''Asset: vertical premium outdoor-apparel and daypack campaign photograph.
Scene: a steep woodland trail with a receding canopy, damp rock and soft shafts of morning light.
Subject: a middle-aged hiker with short silver-streaked hair wears an unbranded moss-green layered jacket, dark hiking trousers, sturdy boots and a compact ochre daypack. Show the complete person from head to boots, one hand adjusting a chest strap while looking back naturally. A small dark dog with an unobscured face stands farther down the trail, below and behind the hiker. Garment seams, buckles, woven straps and the boot soles should be physically plausible.
Composition: a tall environmental photograph with grounded scale, breathing room above the head and below the boots, and a readable trail linking person and dog. The products should feel genuinely used, with light wear and subtle moisture; keep natural skin and fabric texture. No fantasy wilderness, extreme wide-angle distortion or text.'''),
]


def plan():
    jobs = []
    ceramics = SCENES[4][-1]
    for identity, size, people, animals, product, challenge, scene in SCENES:
        jobs.append({'id': identity, 'size': size, 'prompt': COMMON + (scene or ceramics),
                     'scenario_family': 'ceramics-aspect-probe' if identity.startswith('ceramics-') else identity,
                     'people_requested': people, 'animals_requested': animals,
                     'product': product, 'crop_challenge': challenge,
                     'synthetic': True, 'training_eligible': False})
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    with a.output.open('x') as f:
        json.dump(plan(), f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
