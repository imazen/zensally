"""Analytic constant-prediction oracles for the candidate-retention objectives."""
import math
import unittest

import torch

from fit_photo_retention import objective

# The original loss deliberately makes FP32 class targets. Its backward scalar
# weights retain FP32 rounding even when this analytic fixture uses FP64 logits.
GRADIENT_TOLERANCE=4*torch.finfo(torch.float32).eps


class ConstantPosterior(unittest.TestCase):
    def derivative(self,p,balanced,padded=False):
        count=24 if padded else 20
        raw=torch.zeros(1,count,2,dtype=torch.float64)
        raw[...,0]=math.log(p/(1-p));raw.requires_grad_()
        positive=torch.zeros(1,count,dtype=torch.bool);positive[0,0]=True
        valid=torch.zeros_like(positive);valid[:,:20]=True
        if padded:positive[:,20:]=True  # Invalid positives cannot change the prior.
        objective(raw,positive,torch.full_like(positive,.5,dtype=torch.float64),valid,balanced=balanced).sum().backward()
        return float(raw.grad[...,0].sum()),raw.grad

    def test_balanced_constant_optimum_is_half(self):
        derivative,_=self.derivative(.5,True)
        self.assertAlmostEqual(derivative,0.,delta=GRADIENT_TOLERANCE)

    def test_natural_constant_optimum_is_observed_one_in_twenty(self):
        derivative,_=self.derivative(.05,False)
        self.assertAlmostEqual(derivative,0.,delta=GRADIENT_TOLERANCE)

    def test_half_probability_is_not_natural_optimum(self):
        derivative,_=self.derivative(.5,False)
        self.assertAlmostEqual(derivative,.45,delta=GRADIENT_TOLERANCE)

    def test_padding_has_zero_gradient_and_does_not_change_prior(self):
        derivative,gradient=self.derivative(.05,False,padded=True)
        self.assertAlmostEqual(derivative,0.,delta=GRADIENT_TOLERANCE)
        self.assertTrue(torch.equal(gradient[:,20:],torch.zeros_like(gradient[:,20:])))


if __name__=='__main__':unittest.main()
