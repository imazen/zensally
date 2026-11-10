import unittest
import torch
from composition_extent import ExtentHead,box_metrics

class ExtentTests(unittest.TestCase):
 def test_partition_decoding_stays_visible_and_nonempty_at_saturation(self):
  model=ExtentHead()
  with torch.no_grad():
   for p in model.parameters():p.zero_()
   model.layers[-1].bias.copy_(torch.tensor([10000.,-10000.,-10000.,-10000.,-10000.,10000.]))
  result=model(torch.zeros(2,516));self.assertTrue(bool((result[:,:2]>=0).all()&(result[:,2:]<=1).all()&(result[:,2:]>result[:,:2]).all()))
 def test_box_metrics_and_image_gradient(self):
  torch.manual_seed(31);model=ExtentHead();x=torch.randn(2,516,requires_grad=True);target=torch.tensor([[.1,.2,.8,.9],[0,0,1,1.]]);result=model(x);(result-target).square().mean().backward();self.assertGreater(float(x.grad.abs().sum()),0);metrics,_=box_metrics(target,target);self.assertEqual(metrics['mean_iou'],1.);self.assertEqual(metrics['edge_mae'],0.)
if __name__=='__main__':unittest.main()
