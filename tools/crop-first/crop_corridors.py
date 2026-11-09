"""Exact equal-aspect crop intervals; geometry does not establish artistic quality."""
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'composition'))
from subject_policy import face_relation


def contains(outer,inner):
    return outer[0]<=inner[0]<inner[2]<=outer[2] and outer[1]<=inner[1]<inner[3]<=outer[3]


def audit(inner,outer,width,height,ratio,faces):
    p,q=map(int,ratio.split(':'));g=math.gcd(p,q);p,q=p//g,q//g
    for box in [inner,outer,*faces]:
        if len(box)!=4 or any(type(v) is not int for v in box) or not(0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):raise ValueError('positive in-source integer regions required')
    nested=contains(outer,inner);aspect=all((b[2]-b[0])*q==(b[3]-b[1])*p for b in [inner,outer])
    transitions=[i for i,f in enumerate(faces) if not(contains(inner,f) or face_relation(outer,f)=='excluded')]
    return dict(nested=nested,exact_aspect=aspect,faces_crossing_crop_interval=transitions,geometrically_admissible=nested and aspect and not transitions,
        artistic_quality_verified=False,face_coverage_certified=False)


def domains(inner,outer,ratio):
    p,q=map(int,ratio.split(':'));g=math.gcd(p,q);p,q=p//g,q//g
    if not contains(outer,inner) or any((b[2]-b[0])*q!=(b[3]-b[1])*p for b in [inner,outer]):raise ValueError('nested equal-aspect endpoints required')
    for k in range((inner[2]-inner[0])//p,(outer[2]-outer[0])//p+1):
        cw,ch=k*p,k*q;x0,x1=max(outer[0],inner[2]-cw),min(inner[0],outer[2]-cw);y0,y1=max(outer[1],inner[3]-ch),min(inner[1],outer[3]-ch)
        if x0<=x1 and y0<=y1:yield dict(k=k,width=cw,height=ch,x=[x0,x1],y=[y0,y1])


def member(box,inner,outer,ratio):
    p,q=map(int,ratio.split(':'))
    return contains(outer,box) and contains(box,inner) and (box[2]-box[0])*q==(box[3]-box[1])*p


def samples(inner,outer,ratio):
    rows=list(domains(inner,outer,ratio));output=[]
    for t in [0.,.25,.5,.75,1.]:
        target=round(rows[0]['k']*(1-t)+rows[-1]['k']*t);row=min(rows,key=lambda r:(abs(r['k']-target),r['k']))
        x=max(row['x'][0],min(row['x'][1],round(inner[0]*(1-t)+outer[0]*t)));y=max(row['y'][0],min(row['y'][1],round(inner[1]*(1-t)+outer[1]*t)))
        output.append([x,y,x+row['width'],y+row['height']])
    return output
