import unittest
import torch
from candidate_cut_head import balanced_source_loss


class CutLossTests(unittest.TestCase):
    def test_unknown_scores_have_zero_gradient_and_do_not_affect_loss(self):
        p=torch.tensor([[True,False,False],[False,False,False]]);n=torch.tensor([[False,True,False],[False,False,False]])
        a=torch.tensor([[1.,-1.,100.],[3.,4.,5.]],requires_grad=True);b=a.detach().clone();b[:,2]=-100.;b[1]=999.
        loss,valid=balanced_source_loss(a,p,n);other,_=balanced_source_loss(b,p,n)
        self.assertTrue(torch.equal(loss,other));self.assertEqual(valid.tolist(),[True,False]);loss.sum().backward();self.assertEqual(float(a.grad[0,2]),0.);self.assertTrue(torch.equal(a.grad[1],torch.zeros(3)))

    def test_duplicating_one_known_class_does_not_reweight_it(self):
        a=torch.tensor([[2.,-1.]]);p=torch.tensor([[True,False]]);n=~p
        b=torch.tensor([[2.,2.,-1.]]);bp=torch.tensor([[True,True,False]])
        self.assertTrue(torch.equal(balanced_source_loss(a,p,n)[0],balanced_source_loss(b,bp,~bp)[0]))


if __name__=='__main__':unittest.main()
