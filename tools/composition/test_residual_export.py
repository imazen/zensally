import unittest
import torch
from export_residual_native import layer


class ExportLayout(unittest.TestCase):
    def test_baker_input_major_layout_with_rectangular_dense_weights(self):
        encoded=layer(torch.tensor([[1.,2.,3.],[4.,5.,6.]]),torch.tensor([7.,8.]),'identity')
        inputs=[2.,3.,4.];actual=[encoded['biases'][j]+sum(inputs[i]*encoded['weights'][i*2+j] for i in range(3)) for j in range(2)]
        self.assertEqual(actual,[27.,55.])


if __name__=='__main__':unittest.main()
