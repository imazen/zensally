"""Exact native refinement: choose a whole head, preserve its interval membership."""
import math
import sys
from pathlib import Path
from subject_origin_projection import nearest
from subject_policy import face_relation
from audit_crop_feasibility import interval
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from crop_corridors import audit,contains


def refine(width,height,outer,inner,ratio,heads):
    p,q=map(int,ratio.split(':'));g=math.gcd(p,q);p,q=p//g,q//g
    if not contains(outer,inner) or any((b[2]-b[0])*q!=(b[3]-b[1])*p for b in [outer,inner]) or (inner[2]-inner[0])*(inner[3]-inner[1])>=(outer[2]-outer[0])*(outer[3]-outer[1]):raise ValueError('strict nested matching-aspect parent required')
    if not heads:return dict(outer=list(outer),inner=list(inner),status='no-proposals-unchecked',geometry_checked=False,inner_enlarged=False)
    projected=nearest(width,height,outer,heads)
    if projected is None:return dict(outer=None,inner=None,status='no-head-safe-outer-origin',geometry_checked=True,inner_enlarged=False)
    retained=[head for head in heads if face_relation(projected,head)=='contained'];cw,ch=projected[2]-projected[0],projected[3]-projected[1];local=[[h[0]-projected[0],h[1]-projected[1],h[2]-projected[0],h[3]-projected[1]] for h in retained];needed=max((max(h[2] for h in local)-min(h[0] for h in local)+p-1)//p,(max(h[3] for h in local)-min(h[1] for h in local)+q-1)//q);original_k=(inner[2]-inner[0])//p;k=max(original_k,needed)
    if k>=cw//p:return dict(outer=projected,inner=None,status='no-strict-child-retaining-outer-heads',geometry_checked=True,inner_enlarged=k>original_k)
    feasible=interval(cw,ch,(p,q),local,k)
    if feasible is None:raise ValueError('computed minimum retained-head extent must fit projected outer')
    lx,ly=feasible[:2];hx=min([cw-k*p]+[h[0] for h in local]);hy=min([ch-k*q]+[h[1] for h in local]);x=max(lx,min(hx,inner[0]-projected[0]));y=max(ly,min(hy,inner[1]-projected[1]));child=[x+projected[0],y+projected[1],x+projected[0]+k*p,y+projected[1]+k*q];result=audit(child,projected,width,height,ratio,heads)
    if not result['geometrically_admissible']:raise ValueError('projected pair violates existing continuous interval guard')
    return dict(outer=projected,inner=child,status='provided-head-interval-safe',geometry_checked=True,inner_enlarged=k>original_k)
