"""Fixed-capacity source-region geometry for controlled crop-head research."""
import math

CAPACITY=8
FEATURES=CAPACITY*5+7


def descriptor(boxes):
    normalized=[]
    for box in boxes:
        if len(box)!=4 or not all(type(v) in [int,float] and math.isfinite(v) for v in box):raise ValueError('finite normalized xyxy regions required')
        if not(0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):raise ValueError('positive in-source regions required')
        normalized.append(tuple(float(v) for v in box))
    unique=sorted(set(normalized),key=lambda b:(-(b[2]-b[0])*(b[3]-b[1]),*b))
    vector=[]
    for box in unique[:CAPACITY]:vector.extend([*box,1.])
    vector.extend([0.]*(CAPACITY*5-len(vector)))
    if unique:
        vector.extend([min(b[0] for b in unique),min(b[1] for b in unique),max(b[2] for b in unique),max(b[3] for b in unique),
            sum((b[0]+b[2])/2 for b in unique)/len(unique),sum((b[1]+b[3])/2 for b in unique)/len(unique),math.log1p(len(unique))])
    else:vector.extend([0.]*7)
    return vector,dict(input_regions=len(boxes),unique_regions=len(unique),retained_slots=min(CAPACITY,len(unique)),omitted_slots=max(0,len(unique)-CAPACITY))
