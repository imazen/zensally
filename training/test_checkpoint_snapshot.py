import unittest
import torch
from checkpoint_snapshot import snapshot


class CheckpointSnapshot(unittest.TestCase):
    def test_warmed_optimizer_and_weights_remain_at_snapshot_step(self):
        torch.manual_seed(31);model=torch.nn.Linear(3,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.1)
        def step():
            optimizer.zero_grad(set_to_none=True);model(torch.ones(2,3)).square().mean().backward();optimizer.step()
        step();saved=snapshot(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()))
        expected={k:v.clone() for k,v in saved['model'].items()};step()
        for key,value in expected.items():self.assertTrue(torch.equal(value,saved['model'][key]))
        self.assertTrue(any(not torch.equal(v,saved['model'][k]) for k,v in model.state_dict().items()))
        for state in saved['optimizer']['state'].values():self.assertEqual(state['step'].item(),1)
        self.assertTrue(all(state['step'].item()==2 for state in optimizer.state_dict()['state'].values()))


if __name__=='__main__':unittest.main()
