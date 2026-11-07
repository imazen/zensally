"""Bounded annotation proposals; no face-coverage or aesthetic certification."""
import math


def candidate_pool(width,height,mode,hints=()):
    if type(width) is not int or type(height) is not int or min(width,height)<=0:raise ValueError('positive integer dimensions required')
    if mode not in ['minimal','maximal']:raise ValueError('unknown crop mode')
    rw,rh=(9,16) if mode=='minimal' else (1,1)
    units=min(width//rw,height//rh)
    if units<1:raise ValueError('source too small for exact integer aspect')
    records=[];seen={}
    def add(scale,x,y,origin,reason,center=False):
        if any(not math.isfinite(v) for v in [scale,x,y]) or not .5<=scale<=1 or not 0<=x<=1 or not 0<=y<=1:raise ValueError('invalid proposal geometry')
        if mode=='minimal' and scale!=1:raise ValueError('Minimal proposals must have maximum area')
        k=max(1,math.floor(units*scale));w,h=k*rw,k*rh
        ox=min(width-w,max(0,round(x*width-w/2 if center else (width-w)*x)))
        oy=min(height-h,max(0,round(y*height-h/2 if center else (height-h)*y)))
        box=(ox,oy,w,h)
        if box in seen:
            records[seen[box]]['generators'].append(origin);return
        seen[box]=len(records)
        records.append({'id':f'C{len(records)+1:02}','x':ox,'y':oy,'w':w,'h':h,'generators':[origin],'reason':reason})
    if mode=='minimal':
        horizontal=width*rh>height*rw
        for p in [.5,.35,.65,.2,.8]:
            add(1,p if horizontal else .5,.5 if horizontal else p,'interior-baseline','Center and neighboring maximum-area placements')
    else:
        for scale in [1,.875,.75,.625,.5]:add(scale,.5,.5,'center-scale-baseline','Alternative context and tighter compositions')
        for x in [.35,.65]:add(.75,x,.5,'interior-baseline','Neighboring placement at the same scale')
    if len(hints)>8:raise ValueError('at most eight bounded vision proposals per task')
    for hint in hints:add(hint['scale'],hint['cx'],hint['cy'],'vision-proposal',hint['reason'],center=True)
    return records
