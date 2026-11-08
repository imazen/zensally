import sys
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/composition'))
from split_focalnet import frontier


class SplitTests(unittest.TestCase):
    def test_frontier_keeps_geometry_out_of_analysis(self):
        inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1]) for name in ['image', 'boxes', 'content']]
        nodes = [helper.make_node('Relu', ['image'], ['features']),
                 helper.make_node('Sigmoid', ['features'], ['importance']),
                 helper.make_node('Add', ['boxes', 'content'], ['geometry']),
                 helper.make_node('MatMul', ['features', 'geometry'], ['crop_scores'])]
        graph = helper.make_graph(nodes, 'cut', inputs, [])
        self.assertEqual(frontier(graph), ['features'])
        # Direct image use by a query-dependent node also crosses the frontier.
        graph.node.append(helper.make_node('Add', ['image', 'content'], ['mixed']))
        self.assertEqual(frontier(graph), ['features', 'image'])


if __name__ == '__main__':
    unittest.main()
