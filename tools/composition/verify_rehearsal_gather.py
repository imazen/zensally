#!/usr/bin/env python3
"""Verify lowering preserves all original graph initializers and public tensors."""
import argparse
import json
from pathlib import Path
import onnx
import numpy as np
from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('raw',type=Path);p.add_argument('gather',type=Path);a=p.parse_args();rows=[]
    for name in ['encoder','baseline-query','candidate-query']:
        before=onnx.load(str(a.raw/(name+'.onnx')));after=onnx.load(str(a.gather/(name+'.onnx')))
        for key in ['input','output']:
            if [v.SerializeToString() for v in getattr(before.graph,key)]!=[v.SerializeToString() for v in getattr(after.graph,key)]:raise ValueError('graph public tensors changed')
        original={v.name:onnx.numpy_helper.to_array(v) for v in before.graph.initializer};updated={v.name:onnx.numpy_helper.to_array(v) for v in after.graph.initializer}
        if any(key not in updated or value.dtype!=updated[key].dtype or value.shape!=updated[key].shape or value.tobytes()!=updated[key].tobytes() for key,value in original.items()):raise ValueError('original initializer changed')
        grids_before=sum(n.op_type=='GridSample' for n in before.graph.node);grids_after=sum(n.op_type=='GridSample' for n in after.graph.node)
        if (grids_before,grids_after)!=((0,0) if name=='encoder' else (2,0)):raise ValueError('exact two-sampler replacement differs')
        rows.append(dict(graph=name,original_initializers=len(original),all_original_initializer_bytes_exact=True,raw_sha256=sha(a.raw/(name+'.onnx')),gather_sha256=sha(a.gather/(name+'.onnx')),grid_sample_before=grids_before,grid_sample_after=grids_after))
    result=dict(rows=rows,api_calls=0,test_inference=False,model_promoted=False)
    with (a.gather/'weight-retention.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
