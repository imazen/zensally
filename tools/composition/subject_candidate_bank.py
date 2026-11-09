"""Bounded mode-specific crop banks; optional oracle-ROI anchors are explicit."""
import math

def _grid(width,height,p,q,k,side):
    cw,ch=p*k,q*k
    for iy in range(side):
        for ix in range(side):
            x=round((width-cw)*ix/max(1,side-1));y=round((height-ch)*iy/max(1,side-1))
            yield (x,y,x+cw,y+ch)

def _minimal(width,height,p,q,k,capacity):
    cw,ch=p*k,q*k;dx,dy=width-cw,height-ch
    major_x=dx/width>=dy/height;minor=dy if major_x else dx;major=dx if major_x else dy
    minor_values=sorted({0,round(minor/2),minor});count=max(1,capacity//len(minor_values))
    for small in minor_values:
        for i in range(count):
            large=round(major*i/max(1,count-1));x,y=(large,small) if major_x else (small,large)
            yield (x,y,x+cw,y+ch)

def generate(width,height,ratio,mode,*,capacity=128,rois=None):
    if mode not in ['minimal','maximal'] or capacity not in [128,256] or min(width,height)<1:raise ValueError('registered bank geometry required')
    p,q=ratio;divisor=math.gcd(p,q);p,q=p//divisor,q//divisor;maximum=min(width//p,height//q)
    if maximum<1:raise ValueError('integer target aspect cannot fit source')
    result=[];seen=set()
    def add(box):
        if box not in seen:result.append(box);seen.add(box)
    grid_capacity=capacity if rois is None else capacity//2
    if mode=='minimal':
        for box in _minimal(width,height,p,q,maximum,grid_capacity):add(box)
    else:
        scales=[1.-i*.1 for i in range(8)] if capacity==128 else [1.-i*.05 for i in range(16)]
        if rois is not None:scales=[1.,.85,.7,.55,.4,.3,.2]
        side=max(2,math.isqrt(grid_capacity//len(scales)))
        for scale in scales:
            for box in _grid(width,height,p,q,max(1,round(maximum*scale)),side):
                if len(result)<grid_capacity:add(box)
    if rois is not None:
        anchors=[]
        for roi in sorted(rois,key=lambda r:(r['priority']!='primary',r['kind'] not in ['person','product','animal'],r['id'])):
            if roi['kind'] not in ['person','product','animal','face_head']:continue
            b=roi['box'];left,top,right,bottom=[b[k]*s/100 for k,s in zip(['left','top','right','bottom'],[width,height,width,height])]
            needed=max(1,math.ceil(max((right-left)/p,(bottom-top)/q)))
            sizes=[maximum] if mode=='minimal' else sorted({min(maximum,math.ceil(needed*factor)) for factor in [1.,1.3,1.7,2.2]})
            candidates=[]
            for k in sizes:
                cw,ch=p*k,q*k
                for ax,ay in [(.5,.5),(1/3,.5),(2/3,.5),(.5,1/3),(.5,2/3)]:
                    x=max(0,min(width-cw,round((left+right)/2-cw*ax)));y=max(0,min(height-ch,round((top+bottom)/2-ch*ay)))
                    candidates.append((x,y,x+cw,y+ch))
            anchors.append(candidates)
        # Round-robin keeps several subjects represented before refining any one.
        for offset in range(max(map(len,anchors),default=0)):
            for group in anchors:
                if offset<len(group) and len(result)<capacity:add(group[offset])
    if not result or len(result)>capacity:raise ValueError('bounded nonempty bank required')
    if any(not(0<=l<r<=width and 0<=t<b<=height and (r-l)*q==(b-t)*p) for l,t,r,b in result):raise ValueError('native bank geometry differs')
    return result

