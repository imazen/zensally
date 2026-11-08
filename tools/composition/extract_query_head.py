#!/usr/bin/env python3
"""Export the existing small dense head for the owned ZNPR baker, unchanged."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx

from artifact_identity import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    model=onnx.load(args.query)
    values={v.name:onnx.numpy_helper.to_array(v) for v in model.graph.initializer}
    norms=[node for node in model.graph.node if node.op_type=='LayerNormalization']
    if len(norms)!=1:raise ValueError('expected one fixed LayerNorm')
    norm=norms[0]
    attributes={a.name:onnx.helper.get_attribute_value(a) for a in norm.attribute}
    if attributes.get('axis',-1)!=-1:raise ValueError('unexpected normalization axis')
    gamma,beta=[values[name] for name in norm.input[1:3]]
    if gamma.shape!=(62,) or beta.shape!=(62,):raise ValueError('expected 62-feature normalization')
    layers=[]
    previous=norm.output[0]
    for in_dim,out_dim,activation in [(62,64,'relu'),(64,32,'relu'),(32,1,'identity')]:
        products=[n for n in model.graph.node if n.op_type=='MatMul' and n.input[0]==previous and n.input[1] in values]
        if len(products)!=1:raise ValueError('unexpected dense chain')
        product=products[0];weight=values[product.input[1]]
        adds=[n for n in model.graph.node if n.op_type=='Add' and product.output[0] in n.input]
        if len(adds)!=1:raise ValueError('expected one dense bias add')
        add=adds[0];bias_name=next(name for name in add.input if name!=product.output[0]);bias=values[bias_name]
        if weight.shape!=(in_dim,out_dim) or bias.shape!=(out_dim,):raise ValueError('unexpected dense shapes')
        layers.append(dict(in_dim=in_dim,out_dim=out_dim,activation=activation,dtype='f32',
                           weights=weight.astype(np.float32).ravel().tolist(),biases=bias.astype(np.float32).tolist()))
        previous=add.output[0]
        if activation=='relu':
            relus=[n for n in model.graph.node if n.op_type=='Relu' and n.input[0]==previous]
            if len(relus)!=1:raise ValueError('expected hidden ReLU')
            previous=relus[0].output[0]
    request=dict(schema_hash=0x5A534351303031,flags=0,scaler_mean=[0.]*62,scaler_scale=[1.]*62,
                 layers=layers,zerobias_tau=0.,compressed=False,optimize=False)
    result=dict(format='zensally-query-head-v1',gamma=gamma.tolist(),beta=beta.tolist(),
                epsilon=attributes.get('epsilon',1e-5),bake=request,
                source_query_sha256=sha(args.query),onnx_version=onnx.__version__,
                descriptor_order='inside17,outside17,difference17,geometry11',
                dense_weight_order='input-major, output-contiguous, copied from ONNX MatMul constants',
                learned_parameter_values_changed=False,face_coverage_certified=False)
    with (args.output/'head.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    onnx.utils.extract_model(str(args.query),str(args.output/'descriptors.onnx'),
        [v.name for v in model.graph.input],[norm.input[0]],check_model=True)
    identity=dict(head_json_sha256=sha(args.output/'head.json'),descriptor_model_sha256=sha(args.output/'descriptors.onnx'),
        original_query_sha256=sha(args.query),dense_parameters=sum(len(l['weights'])+len(l['biases']) for l in layers),
        normalization_parameters=len(gamma)+len(beta),face_coverage_certified=False)
    (args.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps(identity))


if __name__=='__main__':main()
