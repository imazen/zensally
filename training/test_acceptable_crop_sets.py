import unittest
import torch
from acceptable_crop_sets import acceptable_set_loss

class AcceptableSets(unittest.TestCase):
    def test_unlisted_candidate_has_no_loss_or_gradient(self):
        masks=torch.tensor([[True,False,False]]),torch.tensor([[False,True,False]])
        a=torch.tensor([[1.,0.,-10.]],requires_grad=True);b=torch.tensor([[1.,0.,1e6]],requires_grad=True)
        la,_=acceptable_set_loss(a,*masks);lb,_=acceptable_set_loss(b,*masks)
        self.assertEqual(la.item(),lb.item());lb.sum().backward();self.assertEqual(b.grad[0,2].item(),0)
    def test_alternative_order_is_irrelevant(self):
        scores=torch.tensor([[1.,2.,3.]])
        a,_=acceptable_set_loss(scores,torch.tensor([[True,True,False]]),torch.tensor([[False,False,True]]))
        b,_=acceptable_set_loss(scores[:,[1,0,2]],torch.tensor([[True,True,False]]),torch.tensor([[False,False,True]]))
        self.assertEqual(a.item(),b.item())
    def test_empty_reference_row_has_zero_finite_gradient(self):
        scores=torch.randn(2,3,requires_grad=True);mask=torch.zeros(2,3,dtype=torch.bool)
        loss,valid=acceptable_set_loss(scores,mask,~mask);loss.sum().backward()
        self.assertFalse(valid.any());self.assertTrue(torch.isfinite(scores.grad).all());self.assertEqual(scores.grad.abs().sum().item(),0)

if __name__=='__main__':unittest.main()

