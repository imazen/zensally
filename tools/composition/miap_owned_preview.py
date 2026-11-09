"""Shared owned preview/readback for MIAP downloads; independent readers verify."""
import json
import subprocess

from PIL import Image
from artifact_identity import sha


def prepare_preview(directory,pipeline,png_reader,source,download,boxes,*,allow_grayscale=False):
    original=directory/'source.jpg'
    with Image.open(original) as image:
        header=dict(size=list(image.size),orientation=image.getexif().get(274,1),mode=image.mode)
        modes={'RGB','L'} if allow_grayscale else {'RGB'}
        if image.format!='JPEG' or image.mode not in modes or header['orientation']!=1 or image.width*image.height>32000000:
            raise ValueError('source header outside registered RGB/orientation/size scope')
    result=subprocess.run([str(pipeline),str(original),str(directory/'preview.png'),'768'],capture_output=True,text=True)
    (directory/'pipeline.log').write_text(result.stdout+result.stderr);result.check_returncode();geometry=json.loads(result.stdout)
    result=subprocess.run([str(png_reader),str(directory/'preview.png'),str(directory/'preview.rgb')],capture_output=True,text=True)
    (directory/'reader.log').write_text(result.stdout+result.stderr);result.check_returncode()
    with Image.open(directory/'preview.png') as image:
        if list(image.size)!=[geometry['width'],geometry['height']] or image.convert('RGB').tobytes()!=(directory/'preview.rgb').read_bytes():
            raise ValueError('independent PNG samples/dimensions differ')
    if header['size']!=[geometry['source_width'],geometry['source_height']]:raise ValueError('independent JPEG and owned source dimensions differ')
    if len(boxes)!=source['box_inventory']['box_records']:raise ValueError('source-bound annotation count differs')
    return dict(id=source['id'],source=source,download=download,header=header,geometry=geometry,boxes=boxes,
        **{name+'_sha256':sha(directory/name) for name in ['preview.png','preview.rgb']},
        independent_png_rgb_exact=True,training_eligible=False,face_coverage_certified=False,
        original_md5_not_applicable_to_cvdf_derivative=True,annotation_alignment_review_complete=False)
