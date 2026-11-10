import unittest
import numpy as np
from composition_features import candidate_features,exit_t,NAMES

class FeaturesTests(unittest.TestCase):
 def fixtures(self):
  pose={'landmarks':[[{'x':.25,'y':.5,'visibility':1.,'presence':1.} for _ in range(33)]]}
  pose['landmarks'][0][13]['x']=.75
  gaze={'people':[{'head':[.1,.3,.3,.5],'target':[.8,.4],'in_frame_score':1.}]}
  scan={'paths':[[{'x':.2,'y':.4}]*3+[{'x':.8,'y':.4}]*3 for _ in range(4)]}
  return pose,gaze,scan
 def test_half_crop_breaks_action_and_attention_relation(self):
  whole,half=candidate_features([[0,0,1,1],[0,0,.5,1]],*self.fixtures())
  self.assertEqual(len(whole),len(NAMES));self.assertEqual(whole[2],1);self.assertEqual(half[10],.5);self.assertEqual(half[20],0);self.assertEqual(half[25],1);self.assertEqual(half[26],.5);self.assertEqual(half[27],.5);self.assertEqual(half[29],.4)
  self.assertAlmostEqual(half[21],.5);self.assertEqual(whole[20],1)
 def test_absent_geometry_is_not_certified_and_invalid_boxes_fail(self):
  pose,gaze,scan=self.fixtures();values=candidate_features([[0,0,1,1]],{'landmarks':[]},{'people':[]},scan)
  self.assertTrue((values[0,:24]==0).all());self.assertEqual(exit_t([.5,.5],[0,0],[0,0,1,1]),0)
  for box in [[.5,0,.5,1],[-.1,0,1,1],[0,0,float('nan'),1]]:
   with self.assertRaises(ValueError):candidate_features([box],pose,gaze,scan)
 def test_missing_joint_confidence_cannot_create_a_cut_penalty(self):
  pose,gaze,scan=self.fixtures()
  for p in pose['landmarks'][0]:p['presence']=None
  values=candidate_features([[0,0,.5,1]],pose,gaze,scan)[0]
  self.assertEqual(values[1],0);self.assertEqual(values[10],0)
if __name__=='__main__':unittest.main()
