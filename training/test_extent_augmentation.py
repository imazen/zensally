import unittest

import torch

from extent_augmentation import transform


class ExtentAugmentationTests(unittest.TestCase):
    def test_content_reflection_keeps_asymmetric_padding_and_target_coordinates(self):
        # Three content columns with unequal left/right padding expose a
        # whole-canvas flip: it would move the content two columns right.
        backing = torch.zeros(1,3,12,18)
        image = backing[:,:,::2,::2]
        image[:,:,2:4,2:5] = torch.tensor([.2,.4,.8])[None,None,None]
        original = image.clone()
        target = torch.tensor([[1/3,0.,1.,1.]])
        content = torch.tensor([[2/9,2/6,5/9,4/6]])
        output,box = transform(image,target,content,torch.tensor([True]),
                               torch.ones(1,1,1,1),torch.ones(1,1,1,1))
        expected = torch.zeros_like(image)
        expected[:,:,2:4,2:5] = torch.tensor([.8,.4,.2])[None,None,None]
        torch.testing.assert_close(output,expected,rtol=0,atol=1e-7)
        torch.testing.assert_close(box,torch.tensor([[0.,0.,2/3,1.]]),rtol=0,atol=1e-7)
        torch.testing.assert_close(image,original,rtol=0,atol=0)
        packed,_ = transform(image.contiguous(),target,content,torch.tensor([True]),
                             torch.ones(1,1,1,1),torch.ones(1,1,1,1))
        torch.testing.assert_close(output,packed,rtol=0,atol=0)

    def test_color_transform_clamps_content_and_preserves_black_padding(self):
        image = torch.zeros(1,3,2,3)
        image[:,:,:,1] = 1
        target = torch.tensor([[0.,0.,1.,1.]])
        output,box = transform(image,target,torch.tensor([[1/3,0.,2/3,1.]]),
            torch.tensor([False]),torch.full((1,1,1,1),1.2),torch.full((1,1,1,1),1.2))
        torch.testing.assert_close(output,image,rtol=0,atol=0)
        torch.testing.assert_close(box,target,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
