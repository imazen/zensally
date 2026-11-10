#!/usr/bin/env python3
"""Evaluation-only MediaPipe reference runner on hashed, owned-decoded RGB."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision,BaseOptions


def make(asset):
    base=BaseOptions(model_asset_path=asset['path'],delegate=BaseOptions.Delegate.CPU)
    name=asset['id']
    if 'blaze_face' in name:
        runner=vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(base_options=base,min_detection_confidence=.5))
        return runner,runner.detect,'faces'
    if any(s in name for s in ['efficientdet','ssd_mobilenet']):
        runner=vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(base_options=base,score_threshold=.3,max_results=50))
        return runner,runner.detect,'objects'
    if 'pose_landmarker' in name:
        runner=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=base,num_poses=8))
        return runner,runner.detect,'pose'
    if 'face_landmarker' in name:
        runner=vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=base,num_faces=16))
        return runner,runner.detect,'landmarks'
    runner=vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(base_options=base,output_confidence_masks=True))
    return runner,runner.segment,'masks'


def load(row):
    b=Path(row['rgb']).read_bytes()
    if hashlib.sha256(b).hexdigest()!=row['rgb_sha256']:raise ValueError('source RGB changed')
    return mp.Image(image_format=mp.ImageFormat.SRGB,data=np.frombuffer(b,np.uint8).reshape(row['height'],row['width'],3).copy())


def summarize(result,kind,row,output):
    if kind in ['faces','objects']:
        out=[]
        for d in result.detections:
            b=d.bounding_box
            out.append({'xywh':[b.origin_x,b.origin_y,b.width,b.height],
                        'categories':[{'name':c.category_name,'index':c.index,'score':c.score} for c in d.categories]})
        return {'detections':out}
    if kind in ['pose','landmarks']:
        sets=result.pose_landmarks if kind=='pose' else result.face_landmarks
        return {'landmarks':[[{'x':p.x,'y':p.y,'z':p.z,'visibility':p.visibility,'presence':p.presence} for p in ps] for ps in sets]}
    masks=[]
    for i,m in enumerate(result.confidence_masks):
        data=m.numpy_view()
        if data.ndim==3 and data.shape[2]==1:data=data[:,:,0]
        if data.ndim!=2:raise ValueError("expected a single-channel confidence mask")
        finite=np.isfinite(data)
        if not finite.all():raise ValueError('nonfinite mask')
        yy,xx=np.nonzero(data>=.5)
        box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None
        # Raw gray maps are evaluation artifacts, never training pixels.
        filename=row['id']+'-mask-'+str(i)+'.pgm'
        with (output/filename).open('xb') as f:
            f.write(f'P5\n{data.shape[1]} {data.shape[0]}\n255\n'.encode());f.write((np.clip(data,0,1)*255).astype(np.uint8).tobytes())
        masks.append({'file':filename,'shape':list(data.shape),'min':float(data.min()),'max':float(data.max()),'mean':float(data.mean()),'fraction_ge_half':float((data>=.5).mean()),'box_ge_half':box})
    return {'masks':masks}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['evaluate','bench','profile']);p.add_argument('manifest',type=Path);p.add_argument('model');p.add_argument('output',type=Path);p.add_argument('--repeats',type=int,default=30);a=p.parse_args()
    if not 1<=a.repeats<=10000:raise ValueError('bounded repeats')
    doc=json.loads(a.manifest.read_text());asset=next(x for x in doc['models'] if x['id']==a.model)
    if hashlib.sha256(Path(asset['path']).read_bytes()).hexdigest()!=asset['sha256']:raise ValueError('model changed')
    a.output.mkdir(parents=True,exist_ok=False)
    runner,call,kind=make(asset)
    try:
        if a.mode=='evaluate':
            rows=[]
            for row in doc['images']:
                result=call(load(row));rows.append({'id':row['id'],'result':summarize(result,kind,row,a.output)})
            (a.output/'results.json').write_text(json.dumps({'model':a.model,'kind':kind,'rows':rows},indent=2)+'\n')
            print(json.dumps({'model':a.model,'evaluated':len(rows)}))
        else:
            images=[load(r) for r in doc['images'] if r['benchmark']]
            for image in images:call(image)
            start=time.perf_counter_ns()
            for _ in range(a.repeats):
                for image in images:call(image)
            elapsed=time.perf_counter_ns()-start
            result={'inference_ns':elapsed/(a.repeats*len(images)),'calls':a.repeats*len(images),'images':len(images),'kind':kind,'model':a.model,'timing_scope':'warm CPU MediaPipe Task on preloaded native RGB, includes task preprocessing/postprocessing; excludes load/hash, image construction, feature summarization and file IO','manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest()}
            (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('RESULT '+json.dumps(result))
    finally:runner.close()

if __name__=='__main__':main()
