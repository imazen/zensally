// A viewBox selects coordinates, but does not clip its letterbox gutters.
let zensallyCropClipSerial=0;
function clipCropImage(svg,image,bounds){
 const ns='http://www.w3.org/2000/svg',id='zensally-crop-clip-'+(++zensallyCropClipSerial);
 const defs=document.createElementNS(ns,'defs'),clip=document.createElementNS(ns,'clipPath');
 clip.id=id;clip.setAttribute('clipPathUnits','userSpaceOnUse');
 const rect=document.createElementNS(ns,'rect');
 for(const[k,v]of Object.entries({x:bounds[0],y:bounds[1],width:bounds[2]-bounds[0],height:bounds[3]-bounds[1]}))rect.setAttribute(k,v);
 clip.append(rect);defs.append(clip);svg.prepend(defs);
 image.setAttribute('clip-path','url(#'+id+')');
}
