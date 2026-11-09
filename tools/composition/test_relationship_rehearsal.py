import unittest
import torch
from fit_relationship_rehearsal import old_metrics


class Scores(torch.nn.Module):
    def forward(self,hidden,unused):return hidden[...,0]


class RehearsalMetric(unittest.TestCase):
    def test_family_balance_query_availability_and_eligibility(self):
        hidden=torch.zeros(3,22,2,1);hidden[:,:,0]=1;hidden[1,:,1]=2
        target=torch.zeros(3,22,2);target[:,:,1]=1
        valid=torch.zeros(3,22,dtype=torch.bool);valid[:,0]=True;valid[1,1]=True
        data=dict(hidden=hidden,targets=target,query_valid=valid,eligible=torch.ones(3,22,2,dtype=torch.bool),sources=[dict(family_id=f) for f in ['a','a','b']])
        score,choices=old_metrics(Scores(),data,[0,1,2]);self.assertEqual(score,.75)
        self.assertEqual(choices[:,0].tolist(),[0,1,0])
        data['eligible'][2,:,0]=False
        self.assertEqual(old_metrics(Scores(),data,[0,1,2])[0],.25)
        data['query_valid'][0]=False
        with self.assertRaises(ValueError):old_metrics(Scores(),data,[0,1,2])


if __name__=='__main__':unittest.main()
