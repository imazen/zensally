"""Nearest integer crop origin preserving one whole supplied head and cutting none."""
from subject_policy import face_relation,valid_box


def padded_heads(width,height,heads,percent):
    if percent not in [0,10,25] or type(percent) is not int:raise ValueError('registered integer padding percent required')
    result=[]
    for head in heads:
        if not valid_box(head) or any(type(v) is not int for v in head) or not(0<=head[0]<head[2]<=width and 0<=head[1]<head[3]<=height):raise ValueError('integer in-source supplied head required')
        dx=((head[2]-head[0])*percent+99)//100+2 if percent else 0
        dy=((head[3]-head[1])*percent+99)//100+2 if percent else 0
        result.append([max(0,head[0]-dx),max(0,head[1]-dy),min(width,head[2]+dx),min(height,head[3]+dy)])
    return result


def union(intervals):
    output=[]
    for left,right in sorted(intervals):
        if left>right:continue
        if output and left<=output[-1][1]+1:output[-1]=(output[-1][0],max(right,output[-1][1]))
        else:output.append((left,right))
    return output


def difference(allowed,forbidden):
    output=[];j=0
    for left,right in union(allowed):
        while j<len(forbidden) and forbidden[j][1]<left:j+=1
        cursor=left;k=j
        while k<len(forbidden) and forbidden[k][0]<=right:
            first,last=forbidden[k]
            if cursor<first:output.append((cursor,min(right,first-1)))
            cursor=max(cursor,last+1)
            if cursor>right:break
            k+=1
        if cursor<=right:output.append((cursor,right))
    return output


def nearest(width,height,box,heads):
    if type(width) is not int or type(height) is not int or min(width,height)<1 or not valid_box(box) or any(type(v) is not int for v in box):raise ValueError('positive integer source/crop geometry required')
    if not(0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):raise ValueError('crop outside source')
    for head in heads:
        if not valid_box(head) or any(type(v) is not int for v in head) or not(0<=head[0]<head[2]<=width and 0<=head[1]<head[3]<=height):raise ValueError('integer in-source supplied head required')
    cw,ch=box[2]-box[0],box[3]-box[1];mx,my=width-cw,height-ch
    boundaries={0,my+1}
    for left,top,right,bottom in heads:
        boundaries.update(max(0,min(my+1,value)) for value in [top-ch+1,bottom-ch,top+1,bottom])
    boundaries=sorted(boundaries);best=None;result=None
    def clipped(first,last):return max(0,first),min(mx,last)
    for first,last in zip(boundaries,boundaries[1:]):
        y=max(first,min(last-1,box[1]));allowed=[];forbidden=[]
        for left,top,right,bottom in heads:
            if y+ch<=top or y>=bottom:continue
            overlap=(left-cw+1,right-1)
            if y<=top and y+ch>=bottom and right-left<=cw:
                allowed.append(clipped(right-cw,left))
                forbidden.extend([clipped(overlap[0],right-cw-1),clipped(left+1,overlap[1])])
            else:forbidden.append(clipped(*overlap))
        for low,high in difference(allowed,union(forbidden)):
            x=max(low,min(high,box[0]));key=(((x-box[0])*height)**2+((y-box[1])*width)**2,y,x)
            if best is None or key<best:best=key;result=[x,y,x+cw,y+ch]
    if result is not None:
        relation=[face_relation(result,head) for head in heads]
        if 'contained' not in relation or 'cut' in relation:raise ValueError('origin sweep violated supplied-head geometry')
    return result
