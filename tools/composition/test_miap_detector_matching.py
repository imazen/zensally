import unittest

from score_miap_detectors import matches


def target(box,group=0):return dict(box=box,flags={'IsGroupOf':group})
def prediction(xywh,score=.8):return dict(xywh=xywh,categories=[dict(name='person',score=score)])


class Matching(unittest.TestCase):
    def test_one_prediction_cannot_match_two_targets(self):
        row=dict(result=dict(detections=[prediction([25,25,50,50])]))
        targets,used,_=matches(row,[target([.25,.25,.75,.75])]*2,100,100)
        self.assertEqual(len(targets),2);self.assertEqual(len(used),1)

    def test_group_regions_are_not_individual_targets(self):
        row=dict(result=dict(detections=[prediction([0,0,100,100])]))
        targets,used,_=matches(row,[target([0,0,1,1],group=1)],100,100)
        self.assertEqual(targets,[]);self.assertEqual(used,set())

    def test_non_square_source_uses_each_axis(self):
        row=dict(result=dict(detections=[prediction([50,25,100,50])]))
        _,used,_=matches(row,[target([.25,.25,.75,.75])],200,100)
        self.assertEqual(used,{0})

    def test_raw_outside_box_is_not_clipped_to_improve_overlap(self):
        row=dict(result=dict(detections=[prediction([-100,-100,300,300])]))
        _,used,_=matches(row,[target([0,0,1,1])],100,100)
        self.assertEqual(used,set())


if __name__=='__main__':unittest.main()
