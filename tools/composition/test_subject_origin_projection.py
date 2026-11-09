import random
import unittest
from subject_origin_projection import nearest
from subject_policy import face_relation


class OriginProjection(unittest.TestCase):
    def test_matches_exhaustive_native_grid_with_individual_person_selection(self):
        rng=random.Random(31)
        for _ in range(250):
            w,h=rng.randrange(4,18),rng.randrange(4,18);cw,ch=rng.randrange(1,w+1),rng.randrange(1,h+1)
            x,y=rng.randrange(w-cw+1),rng.randrange(h-ch+1);box=[x,y,x+cw,y+ch];heads=[]
            for _ in range(rng.randrange(6)):
                l,t=rng.randrange(w),rng.randrange(h);heads.append([l,t,rng.randrange(l+1,w+1),rng.randrange(t+1,h+1)])
            legal=[]
            for iy in range(h-ch+1):
                for ix in range(w-cw+1):
                    candidate=[ix,iy,ix+cw,iy+ch];states=[face_relation(candidate,head) for head in heads]
                    if 'contained' in states and 'cut' not in states:legal.append(((((ix-x)*h)**2+((iy-y)*w)**2,iy,ix),candidate))
            expected=min(legal)[1] if legal else None
            self.assertEqual(nearest(w,h,box,heads),expected)

    def test_selects_one_person_without_preserving_the_union(self):
        heads=[[1,1,3,3],[7,1,9,3]]
        self.assertEqual(nearest(10,5,[0,0,4,5],heads),[0,0,4,5])
        self.assertEqual(nearest(10,5,[6,0,10,5],heads),[6,0,10,5])

    def test_edge_touch_exclusion_and_order_independence(self):
        heads=[[1,1,3,3],[5,1,8,4]];box=[0,0,5,5]
        self.assertEqual(nearest(10,5,box,heads),box);self.assertEqual(nearest(10,5,box,list(reversed(heads))),box)
        self.assertIsNone(nearest(10,10,[0,0,2,2],[[0,0,10,10]]))
        self.assertIsNone(nearest(10,10,[0,0,2,2],[]))


if __name__=='__main__':unittest.main()
