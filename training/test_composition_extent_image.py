import unittest
import torch
from torch import nn
from composition_extent import ExtentHead,extent_objective
from composition_extent_image import ImageExtentModel

class TinyRanker(nn.Module):
 def __init__(self):super().__init__();self.project=nn.Conv2d(3,16,1)
 def encode(self,image):
  feature=self.project(image);return feature,nn.functional.avg_pool2d(feature,2)

class ImageExtentTests(unittest.TestCase):
 def test_extent_gradient_reaches_shared_encoder_from_strided_pixels(self):
  torch.manual_seed(31);model=ImageExtentModel(TinyRanker(),ExtentHead(),torch.zeros(512),torch.ones(512));image=torch.randn(2,3,16,16)[:,:,::2,::2];geometry=torch.zeros(2,4);box=model(image,geometry);loss=extent_objective(box,torch.tensor([[.1,.2,.8,.9],[0.,0.,1.,1.]]));loss.backward();self.assertGreater(float(model.ranker.project.weight.grad.abs().sum()),0);torch.testing.assert_close(box,model(image.contiguous(),geometry),rtol=1e-6,atol=1e-6)
if __name__=='__main__':unittest.main()
