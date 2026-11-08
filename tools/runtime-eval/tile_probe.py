#!/usr/bin/env python3
"""Reference-only five-view face guard: full source plus four overlapping 75% views."""
import math
from types import SimpleNamespace
import numpy as np
import mediapipe as mp
import mediapipe_probe as probe

original_make=probe.make

def iou(a,b):
 x0,y0=max(a.origin_x,b.origin_x),max(a.origin_y,b.origin_y)
 x1,y1=min(a.origin_x+a.width,b.origin_x+b.width),min(a.origin_y+a.height,b.origin_y+b.height)
 area=max(0,x1-x0)*max(0,y1-y0);union=a.width*a.height+b.width*b.height-area
 return area/union if union>0 else 0

def make(asset):
 runner,base,kind=original_make(asset)
 if kind!='faces':raise ValueError('only face detection supports this fixed view policy')
 def call(image):
  pixels=image.numpy_view();h,w=pixels.shape[:2];detections=list(base(image).detections)
  cw,ch=math.ceil(w*.75),math.ceil(h*.75)
  for x,y in [(0,0),(w-cw,0),(0,h-ch),(w-cw,h-ch)]:
   view=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(pixels[y:y+ch,x:x+cw]))
   for d in base(view).detections:
    b=d.bounding_box
    detections.append(SimpleNamespace(bounding_box=SimpleNamespace(origin_x=b.origin_x+x,origin_y=b.origin_y+y,width=b.width,height=b.height),categories=d.categories))
  selected=[]
  for d in sorted(detections,key=lambda d:-d.categories[0].score):
   if all(iou(d.bounding_box,s.bounding_box)<=.3 for s in selected):selected.append(d)
  return SimpleNamespace(detections=selected)
 return runner,call,kind

probe.make=make
if __name__=='__main__':probe.main()
