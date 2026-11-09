import unittest

import numpy as np
import torch

from person_relationships import PersonHead, auxiliary_loss, statistics


def record(box,group=False):
    return dict(box=box,flags=dict(IsGroupOf=int(group)))


class PersonRelationships(unittest.TestCase):
    def test_individual_centers_and_group_union_have_distinct_meanings(self):
        target,valid=statistics([record([0,0,.2,.2]),record([.8,.8,1,1]),record([.05,.05,.95,.95],True)])
        np.testing.assert_allclose(target[:4],[0,0,1,1])
        np.testing.assert_allclose(target[4:8],[.5,.5,.8,.8],atol=1e-6)
        self.assertTrue(valid.all())
        self.assertGreater(target[8],target[9])

    def test_group_only_masks_unknown_individual_positions(self):
        target,valid=statistics([record([.1,.2,.7,.8],True)])
        np.testing.assert_array_equal(valid[4:8],np.zeros(4))
        self.assertEqual(target[8],0)
        self.assertGreater(target[9],0)

    def test_degenerate_or_nonfinite_geometry_is_rejected(self):
        for box in [[0,0,0,1],[0,0,float('nan'),1],[-.1,0,1,1]]:
            with self.assertRaises(ValueError):statistics([record(box)])

    def test_ordered_head_and_masked_auxiliary_gradients(self):
        torch.set_num_threads(2)
        model=PersonHead();features=(torch.randn(2,16,32,32),torch.randn(2,16,16,16))
        content=torch.tensor([[0.,0.,1.,1.]]*2);aspect=torch.ones(2)
        prediction=model(features,content,aspect)
        self.assertTrue(bool((prediction[:,2:4]>prediction[:,:2]).all()))
        target,valid=statistics([record([.1,.2,.7,.8],True)])
        target=torch.tensor(np.stack([target]*2));valid=torch.tensor(np.stack([valid]*2))
        prediction.retain_grad();auxiliary_loss(prediction,target,valid).mean().backward()
        self.assertTrue(torch.equal(prediction.grad[:,4:8],torch.zeros(2,4)))

    def test_warmed_inactive_head_adamw_state_stays_absent(self):
        # The production loop has disjoint head forwards with one shared optimizer.
        # Warm both heads first: a cold-only test cannot expose momentum drift.
        torch.set_num_threads(2);torch.manual_seed(31)
        auxiliary=PersonHead();rank=torch.nn.Linear(3,1)
        optimizer=torch.optim.AdamW([*auxiliary.parameters(),*rank.parameters()],lr=.01)
        content=torch.tensor([[0.,0.,1.,1.]]*2);aspect=torch.ones(2)
        for task in ['rank','auxiliary']:
            optimizer.zero_grad(set_to_none=True)
            value=rank(torch.ones(2,3)).sum() if task=='rank' else auxiliary(None,content,aspect,geometry=True).sum()
            value.backward();optimizer.step()
        before={p:(p.detach().clone(),{k:v.clone() if torch.is_tensor(v) else v for k,v in optimizer.state[p].items()}) for p in auxiliary.parameters()}
        optimizer.zero_grad(set_to_none=True);rank(torch.ones(2,3)).sum().backward();optimizer.step()
        for p,(weight,state) in before.items():
            self.assertIsNone(p.grad);self.assertTrue(torch.equal(p,weight))
            for key,value in state.items():self.assertTrue(torch.equal(optimizer.state[p][key],value))
        before={p:(p.detach().clone(),{k:v.clone() if torch.is_tensor(v) else v for k,v in optimizer.state[p].items()}) for p in rank.parameters()}
        optimizer.zero_grad(set_to_none=True);auxiliary(None,content,aspect,geometry=True).sum().backward();optimizer.step()
        for p,(weight,state) in before.items():
            self.assertIsNone(p.grad);self.assertTrue(torch.equal(p,weight))
            for key,value in state.items():self.assertTrue(torch.equal(optimizer.state[p][key],value))


if __name__=='__main__':unittest.main()
