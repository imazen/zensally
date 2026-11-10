#!/usr/bin/env python3
"""Independently verify three-image processor blocks and actual A/B swaps."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoProcessor
from critic_probe import sha


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['processor','root','output']:p.add_argument(key,type=Path)
    a=p.parse_args();torch.set_num_threads(1)
    processor=AutoProcessor.from_pretrained(a.processor,local_files_only=True)
    rows=[]
    for path in sorted(a.root.glob('*-AB.json'))+sorted(a.root.glob('*-BA.json')):
        response=json.loads(path.read_text());task=response['input'];arrays=[]
        for row in task['images']:
            source=Path(row['rgb'])
            if sha(source)!=row['sha256']:raise ValueError('stored image bytes differ')
            arrays.append(np.frombuffer(source.read_bytes(),dtype=np.uint8).reshape(row['height'],row['width'],3).copy())
        content=[{'type':'image'} for _ in arrays]+[{'type':'text','text':task['prompt']}]
        prompt=processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
        batch=processor(text=[prompt],images=arrays,do_resize=False,return_tensors='pt')
        if batch.input_ids.shape[1]!=response['input_tokens'] or batch.image_grid_thw.tolist()!=response['image_grid_thw']:raise ValueError('actual inference tokens/grid do not replay')
        image_prompt=processor.apply_chat_template([dict(role='user',content=[{'type':'image'},{'type':'text','text':'Inspect.'}])],tokenize=False,add_generation_prompt=True)
        offset=0;blocks=[]
        for array,grid in zip(arrays,batch.image_grid_thw):
            independent=processor(text=[image_prompt],images=[array],do_resize=False,return_tensors='pt')
            count=int(grid.prod());block=batch.pixel_values[offset:offset+count]
            if not torch.equal(block,independent.pixel_values) or not torch.equal(grid,independent.image_grid_thw[0]):raise ValueError('combined image block differs from independent single-image processing')
            blocks.append(digest(block));offset+=count
        if offset!=len(batch.pixel_values):raise ValueError('extra or missing image blocks')
        rows.append(dict(id=task['id'],task=task['task'],order=task['order'],pixel_blocks_sha256=blocks,input_tokens=int(batch.input_ids.shape[1]),single_image_block_checks=3))
    groups={}
    for row in rows:groups.setdefault(row['task'],{})[row['order']]=row
    if len(rows)!=12 or len(groups)!=6 or any(set(g)!={'AB','BA'} for g in groups.values()):raise ValueError('complete12 responses/six swaps required')
    for group in groups.values():
        ab,ba=[group[k]['pixel_blocks_sha256'] for k in ['AB','BA']]
        if ab[0]!=ba[0] or ab[1]!=ba[2] or ab[2]!=ba[1] or ab[1]==ab[2]:raise ValueError('source stable and two distinct crops must actually swap pixel blocks')
    result=dict(requests=12,pairs=6,independent_single_image_block_checks=36,all_blocks_exact=True,all_sources_stable=True,all_distinct_crops_swapped=True,rows=rows,api_calls=0,model_inference=False)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
