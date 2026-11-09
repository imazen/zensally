"""Owned crop/contain pixels with strided input and independent lossless readback."""
import hashlib
import math
from pathlib import Path
import subprocess
from PIL import Image


def render(raw,width,height,stride,native_size,box,ratio,stem,resizer,encoder,*,unit=192):
    if min(width,height)<1 or stride<width*3 or len(raw)<stride*(height-1)+width*3:raise ValueError('bounded RGB rows and stride required')
    if type(unit) is not int or unit<1:raise ValueError('positive integer ratio unit required')
    nw,nh=native_size;p,q=map(int,ratio.split(':'));ow,oh=unit*p,unit*q
    if box is None:
        scale=min(ow/width,oh/height);rw,rh=max(1,round(width*scale)),max(1,round(height*scale));bounds=[0,0,width,height]
    else:
        if len(box)!=4 or not(0<=box[0]<box[2]<=nw and 0<=box[1]<box[3]<=nh):raise ValueError('bounded native crop required')
        x0,y0,x1,y1=box;bounds=[max(0,math.floor(x0*width/nw)),max(0,math.floor(y0*height/nh)),min(width,math.ceil(x1*width/nw)),min(height,math.ceil(y1*height/nh))];rw,rh=ow,oh
    x0,y0,x1,y1=bounds;config=','.join(map(str,[x0,y0,x1-x0,y1-y0,rw,rh]))
    resized=subprocess.check_output([str(resizer),str(width),str(height),str(stride),config],input=raw)
    if len(resized)!=rw*rh*3:raise ValueError('owned crop byte length differs')
    if box is None:
        pixels=bytearray(b'\xff')*(ow*oh*3);left,top=(ow-rw)//2,(oh-rh)//2
        for y in range(rh):pixels[((top+y)*ow+left)*3:((top+y)*ow+left+rw)*3]=resized[y*rw*3:(y+1)*rw*3]
        resized=bytes(pixels)
    stem=Path(stem);path=stem.with_suffix('.rgb');png=stem.with_suffix('.png');path.write_bytes(resized)
    subprocess.run([str(encoder),'--from-rgb',str(ow),str(oh),str(path),str(png)],check=True,stdout=subprocess.PIPE)
    with Image.open(png) as im:
        im.load()
        if im.size!=(ow,oh) or im.convert('RGB').tobytes()!=resized:raise ValueError('independent encoded readback differs')
    return dict(path=str(png),sha256=hashlib.sha256(png.read_bytes()).hexdigest(),width=ow,height=oh,working_source_bounds=bounds,rgb_sha256=hashlib.sha256(resized).hexdigest())
