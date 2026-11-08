import unittest
import torch
from crop_regions import area_pool,RegionCropRanker


class RegionTests(unittest.TestCase):
    def test_fractional_cells_match_independent_overlap_and_gradients(self):
        torch.manual_seed(19)
        f=torch.randn(2,3,5,7,dtype=torch.float64,requires_grad=True)
        r=torch.tensor([[[0,0,1,1],[.13,.21,.79,.81],[.5,0,.501,1],[0,0,0,1]]]*2,dtype=torch.float64)
        actual=area_pool(f,r)
        expected=[]
        for b in range(2):
            rows=[]
            for x0,y0,x1,y1 in r[b].tolist():
                weights=torch.tensor([[max(0,min(x1,(x+1)/7)-max(x0,x/7))*max(0,min(y1,(y+1)/5)-max(y0,y/5)) for x in range(7)] for y in range(5)],dtype=f.dtype)
                rows.append((f[b]*weights).sum((-2,-1))/weights.sum() if weights.sum()>0 else f[b].sum((-2,-1))*0)
            expected.append(torch.stack(rows))
        expected=torch.stack(expected)
        torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
        ga=torch.autograd.grad(actual.square().sum(),f,retain_graph=True)[0]
        ge=torch.autograd.grad(expected.square().sum(),f)[0]
        torch.testing.assert_close(ga,ge,rtol=1e-10,atol=1e-10)

    def test_heads_backpropagate_and_hard_face_mask_survives(self):
        torch.set_num_threads(1);torch.manual_seed(11)
        m=RegionCropRanker();im=torch.rand(2,3,64,96)
        boxes=torch.tensor([[[0,0,1,1],[.5,.5,1,1]]]*2,dtype=torch.float32)
        faces=torch.tensor([[1,.1,.1,.2,.2,1]]*2)
        rank,accept=m(im,boxes,torch.ones(2),torch.ones(2,2),torch.zeros(2,2),faces,torch.ones(2,2,dtype=torch.bool),torch.tensor([[0,0,1,1]]*2))
        self.assertTrue(torch.isneginf(rank[:,1]).all());self.assertTrue(torch.isneginf(accept[:,1]).all())
        (rank[:,0].sum()+accept[:,0].sum()).backward()
        for module in [m.stem[0],m.project8,m.project16,m.scorer[0]]:
            self.assertTrue(torch.isfinite(module.weight.grad).all());self.assertGreater(module.weight.grad.abs().sum(),0)

if __name__=='__main__':unittest.main()
