#!/usr/bin/env python3
"""Eight source-grounded approximate region witnesses; not human/face certification."""
import argparse,json
from pathlib import Path

# Percentages were marked by Codex after viewing each full source and its crop.
# Broad face groups are conservative bounds, not instance segmentation.
REGIONS={
 'linen-shirt-model':[('face','visible head',[33,0,57,24]),('product','complete shirt sleeves and hem',[14,19,86,88]),('brand','shirt label',[58,38,66,41])],
 'leather-tote-model':[('face','visible head',[23,0,64,32]),('product','tote body handles and strap',[25,38,92,96]),('brand','tote wordmark',[50,64,77,68])],
 'dog-harness-white':[('face','human head',[22,7,57,21]),('pet','whole dog including head and paws',[10,58,99,93]),('product','harness and wordmark',[44,66,94,83])],
 'coffee-bar-context':[('face','near head',[17,0,37,43]),('face','far head',[44,2,54,25]),('product','complete espresso machine and cup shelf',[53,15,97,91]),('brand','machine wordmark',[64,32,79,37])],
 'garden-furniture-context':[('face','left two heads',[20,0,43,27]),('face','right two heads',[63,3,81,35]),('pet','dog head',[27,59,36,83]),('product','table and both benches',[10,34,92,98]),('brand','table and bench wordmarks',[49,44,70,72])],
 'birthday-cake-kitchen':[('face','all five primary head clusters',[0,9,100,67]),('story','birthday cake and candles',[52,59,84,81])],
 'selfie-family-hike':[('face','near adult head',[3,34,41,67]),('face','back adult head',[40,23,60,41]),('face','rear child head',[74,33,88,45]),('face','near child head',[52,57,79,84]),('pet','distant dog',[69,27,74,34])],
 'vacation-museum-cafe':[('face','near child head',[4,49,46,84]),('face','standing adult head visible in source',[15,0,38,17]),('face','seated child head',[57,35,73,51]),('face','seated adult head',[82,26,100,44]),('face','background crowd band',[35,24,100,40]),('story','ice cream and hand',[31,78,48,96])],
}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('first_gate',type=Path);p.add_argument('output',type=Path);a=p.parse_args();lookup={s['id']:s for s in json.loads((a.first_gate/'inputs.json').read_text())['images']};rows=[]
 for id,regions in REGIONS.items():
  s=lookup[id];rows.append({'id':id,'rgb_sha256':s['rgb_sha256'],'width':s['width'],'height':s['height'],'regions':[{'kind':kind,'description':description,'box_percent':box} for kind,description,box in regions],'face_coverage_certified':False,'human_rated':False,'training_eligible':False})
 with a.output.open('x') as f:json.dump({'schema':1,'annotation_origin':'Codex visual source/crop inspection; approximate conservative percentage boxes, unreviewed','profile':'whole-product commerce and candid-story witnesses; not an exhaustive acceptable-crop label set','sources':rows},f,indent=2);f.write('\n')
 print(json.dumps({'sources':len(rows),'regions':sum(len(r['regions']) for r in rows)}))
if __name__=='__main__':main()
