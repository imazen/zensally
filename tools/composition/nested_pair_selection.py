"""Research-only selectors for existing matching-aspect candidate banks."""
import math


def strict_nested(inner, outer):
    return (outer[0]<=inner[0] and outer[1]<=inner[1] and outer[2]>=inner[2]
            and outer[3]>=inner[3] and
            (inner[2]-inner[0])*(inner[3]-inner[1]) < (outer[2]-outer[0])*(outer[3]-outer[1]))


def select(outer, inner, outer_scores, inner_scores, outer_mask, inner_mask, policy):
    if policy not in ['independent','outer-first','joint-sum','maximal-first']:raise ValueError('registered policy required')
    for boxes,scores,mask in [(outer,outer_scores,outer_mask),(inner,inner_scores,inner_mask)]:
        if not len(boxes)==len(scores)==len(mask) or not all(math.isfinite(v) for v in scores):raise ValueError('matching finite scores required')
        for box,eligible in zip(boxes,mask):
            if eligible and (len(box)!=4 or any(type(v) is not int for v in box) or box[0]>=box[2] or box[1]>=box[3]):raise ValueError('eligible integer crop geometry required')
    oi=[k for k,v in enumerate(outer_mask) if v];ii=[k for k,v in enumerate(inner_mask) if v]
    if not oi or not ii:raise ValueError('both registered banks require eligible candidates')
    raw_outer=max(oi,key=lambda k:(outer_scores[k],-k));raw_inner=max(ii,key=lambda k:(inner_scores[k],-k))
    if policy=='independent':return raw_outer,raw_inner
    if policy=='outer-first':
        choices=[k for k in ii if strict_nested(inner[k],outer[raw_outer])]
        return raw_outer,max(choices,key=lambda k:(inner_scores[k],-k)) if choices else None
    if policy=='maximal-first':
        choices=[k for k in oi if strict_nested(inner[raw_inner],outer[k])]
        return (max(choices,key=lambda k:(outer_scores[k],-k)),raw_inner) if choices else (raw_outer,None)
    choices=((x,y) for x in oi for y in ii if strict_nested(inner[y],outer[x]))
    # Model scores are binary32 values represented exactly as Python binary64.
    # Declare binary64 pair sums and index ties as this policy's arithmetic.
    return max(choices,key=lambda xy:(outer_scores[xy[0]]+inner_scores[xy[1]],-xy[0],-xy[1]),default=(raw_outer,None))
