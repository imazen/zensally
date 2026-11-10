import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/composition'))
from cached_focalnet import CachedContext


def tensor(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def stub_graphs(root):
    analysis = helper.make_graph([
        helper.make_node('ReduceMean', ['image'], ['features'], axes=[1,2,3], keepdims=0),
        helper.make_node('Constant', [], ['importance'],
            value=numpy_helper.from_array(np.zeros((1,1,64,64),np.float32))),
    ], 'analysis', [tensor('image',[1,3,256,256])],
        [tensor('features',[1]),tensor('importance',[1,1,64,64])])
    query = helper.make_graph([
        helper.make_node('ReduceMean',['boxes'],['means'],axes=[2],keepdims=0),
        helper.make_node('ReduceMean',['content'],['geometry'],axes=[1],keepdims=0),
        helper.make_node('Add',['means','features'],['values']),
        helper.make_node('Add',['values','geometry'],['crop_scores']),
    ], 'query', [tensor('boxes',[1,128,4]),tensor('content',[1,4]),tensor('features',[1])],
        [tensor('crop_scores',[1,128])])
    for name, graph in [('analysis',analysis),('query',query)]:
        model = helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)],ir_version=8)
        onnx.checker.check_model(model)
        onnx.save(model,root/(name+'.onnx'))


class CacheTests(unittest.TestCase):
    def test_strided_source_and_reused_buffers_reject_stale_tokens(self):
        scratch = Path.home()/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            stub_graphs(root)
            context = CachedContext(root)
            boxes = np.zeros((1,128,4),np.float32)
            with self.assertRaises(ValueError): context.score(boxes,None)
            image = np.ones((1,3,256,512),np.float32)[...,::2]
            self.assertFalse(image.flags.c_contiguous)
            content = np.array([[0,0,1,1]],np.float32)
            old = context.analyze(image,content,'a')
            np.testing.assert_array_equal(context.score(boxes,old),np.full((1,128),1.5,np.float32))
            buffer = context.scores.ctypes.data
            current = context.analyze(image*2,content,'a')
            with self.assertRaises(ValueError): context.score(boxes,old)
            with self.assertRaises(ValueError): context.score(boxes,(current[0],'foreign'))
            np.testing.assert_array_equal(context.score(boxes,current),np.full((1,128),2.5,np.float32))
            self.assertEqual(context.scores.ctypes.data,buffer)
            invalid = boxes.copy()
            invalid[0,0,0] = float('nan')
            with self.assertRaises(ValueError): context.score(invalid,current)
            with self.assertRaises(ValueError): context.analyze(image,content*0,'bad')
            with self.assertRaises(ValueError): context.score(boxes,current)


if __name__ == '__main__':
    unittest.main()
