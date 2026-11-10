"""Fixed constrained-bank eligibility, independent of a neural runtime."""
import numpy as np


def eligible(task,width,height,mode):
    p,q=map(int,task['ratio'].split(':'));maximum=min(width//p,height//q)
    indices=[j for j,c in enumerate(task['candidates']) if
        (c['pixel_box'][2]-c['pixel_box'][0]==maximum*p if mode=='minimal'
         else c['pixel_box'][2]-c['pixel_box'][0]<maximum*p)]
    if not indices and mode=='maximal':indices=list(range(len(task['candidates'])))
    if indices:
        retention=max(task['candidates'][j]['retained_importance'] for j in indices)
        indices=[j for j in indices if task['candidates'][j]['retained_importance']>=retention-.05]
    mask=np.zeros(128,np.uint8);mask[indices]=1
    return mask
