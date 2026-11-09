#!/usr/bin/env python3
"""Scalar descriptors from the already pinned local gaze-teacher backbone."""
import argparse
import atexit
import json
import os
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from artifact_identity import sha
from audit_face_corroboration import verify_member
from attention import image
from teacher_models import load
from distill_attention_linear import write


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inputs','teachers','cache','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(a.inputs/'SHA256SUMS.json')!='86aacbfb10ce299649029e4805efc73073fb542feb4b2f2401cc0d9cb8c32ef3' or sha(a.cache)!='3b85e5ed376b92d62d38e72fdf432d665b7aa62c944c6a874a9c83fc064773c4':raise ValueError('pinned sources differ')
    verify_member(a.inputs,'inputs.json');doc=json.loads((a.inputs/'inputs.json').read_text());members=doc['teacher_members']
    if len(members)!=226:raise ValueError('complete pinned teacher inventory required')
    for name,digest in members.items():
        if sha(Path(name))!=digest:raise ValueError('pinned teacher member changed')
    c=torch.load(a.cache,map_location='cpu',weights_only=False);by_id={r['id']:r for r in doc['images']};sources=c['sources']
    for source in sources:
        row=by_id[source['id']]
        if any(row[k]!=source[k] for k in ['id','role','family_id','rgb_sha256','native_size']) or source['role'] not in ['fit','dev'] or row['gaze']['width']!=448 or row['gaze']['height']!=448:raise ValueError('teacher source identity/shape differs')
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',dict(input_manifest_sha256=sha(a.inputs/'SHA256SUMS.json'),source_cache_sha256=sha(a.cache),teacher_members=members,model='pinned gaze-teacher DINOv2 ViT-B/14 backbone; gaze head bypassed',preprocessing='existing owned448-square gaze RGB; ImageNet normalization',descriptors=['768 global class token','12288 spatial4x4 average-pooled patch descriptors'],diagnostic_maps_saved=False,api_calls=0,test_inference=False,new_downloads=False,model_promoted=False));status=dict(pid=os.getpid(),stage='semantic-features',completed=0,required=72)
    def progress(**kw):status.update(kw,checked_unix=time.time());write(a.output/'PROGRESS.json',status)
    def failed():
        if status['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failed)
    original=torch.hub.download_url_to_file
    def no_download(*args,**kwargs):raise ValueError('teacher downloads are disabled; pinned local cache required')
    torch.hub.download_url_to_file=no_download
    try:model=load('gaze',a.teachers).cuda()
    finally:torch.hub.download_url_to_file=original
    global_values=[];spatial_values=[]
    for i,source in enumerate(sources):
        row=by_id[source['id']];value=image(row,'gaze','cuda')/255;mean=value.new_tensor([.485,.456,.406])[None,:,None,None];std=value.new_tensor([.229,.224,.225])[None,:,None,None];result=model.backbone.model.forward_features((value-mean)/std);global_value=result['x_norm_clstoken'];patch=result['x_norm_patchtokens']
        if global_value.shape!=(1,768) or patch.shape!=(1,1024,768):raise ValueError('pinned DINO feature shape differs')
        spatial=F.adaptive_avg_pool2d(patch.reshape(1,32,32,768).permute(0,3,1,2),(4,4)).flatten(1)
        if not torch.isfinite(global_value).all() or not torch.isfinite(spatial).all():raise ValueError('nonfinite semantic descriptor')
        global_values.append(global_value[0].cpu());spatial_values.append(spatial[0].cpu());progress(completed=i+1,id=source['id'])
    result=dict(schema='pinned-semantic-pair-descriptors-v1',sources=sources,global_features=torch.stack(global_values),spatial_features=torch.stack(spatial_values),protocol_sha256=sha(a.output/'protocol.json'));torch.save(result,a.output/'features.pt');report=dict(sources=72,global_dimensions=768,spatial_dimensions=12288,features_sha256=sha(a.output/'features.pt'),features_bytes=(a.output/'features.pt').stat().st_size,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved(),api_calls=0,test_inference=False,model_promoted=False);write(a.output/'summary.json',report);progress(pid=None,stage='complete',completed=72);print(json.dumps(report))


if __name__=='__main__':main()
