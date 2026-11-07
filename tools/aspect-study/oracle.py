#!/usr/bin/env python3
"""Independent read-only PNG/mask oracle. Pillow never writes study images."""
import argparse,csv,hashlib,json
from pathlib import Path
from PIL import Image
p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path);p.add_argument('manifest',type=Path);p.add_argument('gallery',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
records={}
for r in csv.DictReader(a.manifest.open(),delimiter='\t'):
 with Image.open(a.dataset/r['mask']) as img:
  raw=img.convert('L'); data=raw.tobytes()
  binary=bytes(255 if v>=128 else 0 for v in data)
  records[r['id']]={'w':raw.width,'h':raw.height,'foreground':binary.count(255),'sha256':hashlib.sha256(binary).hexdigest()}
 assert records[r['id']]['foreground']>0
verified_masks=set()
for path in a.gallery.rglob("mask-hashes.tsv"):
 for r in csv.DictReader(path.open(),delimiter="\t"):
  assert records[r["id"]]["sha256"]==r["sha256_binary_mask"],(path,r["id"])
  assert (records[r["id"]]["w"],records[r["id"]]["h"])==(int(r["width"]),int(r["height"]))
  verified_masks.add(r["id"])
assert len(verified_masks)==5019,f"Incomplete native mask oracle: {len(verified_masks)}/5019"
images=0;masks=0
for path in a.gallery.rglob('*.png'):
 with Image.open(path) as img:
  img.load();assert img.width>0 and img.height>0
  sidecar=path.with_suffix(".rgb-sha256")
  assert sidecar.exists(),f"Missing encoder pixel hash: {path}"
  assert hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()==sidecar.read_text().strip(),f"PNG pixel mismatch: {path}"
  if path.name=='mask.png' and path.parent.name.endswith('-natural'):
   ident=path.parent.name.removesuffix('-natural');r=records[ident]
   mask_path=a.dataset/'DUTS-TE-Mask'/f'{ident}.png'
   with Image.open(mask_path) as original:
    source=original.convert('L').tobytes()
   rgb=img.convert('RGB').tobytes();assert img.size==(256,256)
   for y in range(256):
    sy=(y*(r['h']-1)+127)//255
    for x in range(256):
     sx=(x*(r['w']-1)+127)//255
     expected=(255,0,0) if source[sy*r['w']+sx]>=128 else (0,0,255)
     assert tuple(rgb[(y*256+x)*3:][:3])==expected,(path,x,y)
   masks+=1
  images+=1
result={'source_masks':len(records),'verified_native_binary_masks':len(verified_masks),'pngs_read_back':images,'natural_mask_grids_verified':masks,'minimum_source_foreground_pixels':min(r['foreground'] for r in records.values()),'natural_aspect_at_least_4':sum(max(r['w']/r['h'],r['h']/r['w'])>=4 for r in records.values()),'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'pillow_writes_images':False}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
