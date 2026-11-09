import unittest
import torch
from person_risk_policy import person_count,adjusted_scores,hard_logits
from candidate_cut_routing import route


class PersonRiskTests(unittest.TestCase):
    def test_source_without_person_preserves_scores_and_hard_availability(self):
        scores=torch.randn(2,3,4,4,dtype=torch.float64);logits=torch.randn(2,3,2,4);applicable=torch.tensor([False,True])
        for strength in [0.,.25,.5,1.,2.,4.,8.]:self.assertTrue(torch.equal(adjusted_scores(scores,logits,applicable,strength)[0],scores[0]))
        logits[0]=100.;_,present=route(scores,torch.ones_like(scores,dtype=torch.bool),hard_logits(logits,applicable));self.assertTrue(present[0].all())

    def test_person_threshold_and_source_identity(self):
        source=dict(id='x',rgb_sha256='abc',working_size=[10,20]);image=dict(id='x',rgb_sha256='abc',width=10,height=20)
        detections=[dict(categories=[dict(name='person',score=.3)]),dict(categories=[dict(name='person',score=.299)]),dict(categories=[dict(name='cat',score=.99)])]
        self.assertEqual(person_count(source,image,detections),1)
        with self.assertRaises(ValueError):person_count(source,dict(image,rgb_sha256='def'),detections)


if __name__=='__main__':unittest.main()
