import unittest
import torch
from direct_crop import DirectCropHead,decode,geometry,loss


class DirectCrops(unittest.TestCase):
    def test_native_bounds_exact_aspect_and_maximum_minimal_extent(self):
        torch.manual_seed(31)
        sizes=[[1024,683],[257,129],[129,257],[720,960]];ratios=['1:1','3:1','1:3','6:1','1:6','9:16']
        context=geometry(sizes,ratios);origins=torch.randn(4,12,2,2)*30;scales=torch.randn(4,12,2)*30
        _,native=decode(origins,scales,context,integer=True)
        for i,(w,h) in enumerate(sizes):
            for q in range(12):
                p,r=map(int,ratios[q//2].split(':'))
                for box in native[i,q].tolist():
                    l,t,right,b=box
                    self.assertTrue(0<=l<right<=w and 0<=t<b<=h)
                    self.assertEqual((right-l)*r,(b-t)*p)
                    if q%2==0:self.assertEqual((right-l)/p,min(w//p,h//r))

    def test_alternative_order_does_not_change_loss_and_single_target_collapses(self):
        model=DirectCropHead(4);pred=torch.rand(2,2,2,4,requires_grad=True);target=torch.rand(2,2,2,4)
        valid=torch.ones(2,2,2,dtype=torch.bool);offer=torch.zeros(2,2);known=torch.ones(2,2,dtype=torch.bool);mode=torch.tensor([0,1])
        left,_=loss(model,pred,offer,target,valid,offer,known,mode)
        right,_=loss(model,pred,offer,target.flip(2),valid,offer,known,mode)
        self.assertEqual(left.item(),right.item())
        unknown=torch.zeros_like(valid);value,_=loss(model,pred,offer,target,unknown,offer,torch.zeros_like(known),mode)
        value.backward();self.assertEqual(pred.grad.abs().sum().item(),0)

    def test_warmed_adamw_does_not_advance_inactive_regression_heads(self):
        torch.manual_seed(31);model=DirectCropHead(4);optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
        visual=torch.randn(2,4);context=geometry([[20,30],[30,20]],['1:1']);target=torch.rand(2,2,2,4)
        def step(valid):
            optimizer.zero_grad(set_to_none=True);origins,scales,offer=model(visual,context['query']);pred,_=decode(origins,scales,context)
            value,active=loss(model,pred,offer,target,valid,torch.ones(2,2),torch.ones(2,2,dtype=torch.bool),context['mode'])
            value.backward();model.clear_inactive_grads(**active);optimizer.step()
        step(torch.ones(2,2,2,dtype=torch.bool))
        before={name:p.detach().clone() for name,p in model.named_parameters() if name.startswith(('origins.','scales.'))}
        counters={name:optimizer.state[p]['step'].clone() for name,p in model.named_parameters() if name in before}
        step(torch.zeros(2,2,2,dtype=torch.bool))
        for name,p in model.named_parameters():
            if name in before:
                self.assertTrue(torch.equal(before[name],p));self.assertTrue(torch.equal(counters[name],optimizer.state[p]['step']))
        valid=torch.ones(2,2,2,dtype=torch.bool);valid[:,1]=False
        step(valid)
        for name,p in model.named_parameters():
            if name.startswith('scales.'):
                self.assertTrue(torch.equal(before[name],p));self.assertTrue(torch.equal(counters[name],optimizer.state[p]['step']))


if __name__=='__main__':unittest.main()
