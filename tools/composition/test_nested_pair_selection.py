import unittest
from nested_pair_selection import select,strict_nested


class NestedSelection(unittest.TestCase):
    def test_outer_first_preserves_raw_minimal_while_joint_sum_can_move_it(self):
        outer=[[0,0,8,8],[2,0,10,8]];inner=[[0,0,4,4],[6,0,10,4]]
        args=(outer,inner,[3.,2.],[0.,10.],[True,True],[True,True])
        self.assertEqual(select(*args,'independent'),(0,1))
        self.assertEqual(select(*args,'outer-first'),(0,0))
        self.assertEqual(select(*args,'joint-sum'),(1,1))
        self.assertEqual(select(*args,'maximal-first'),(1,1))

    def test_equal_extent_is_not_a_strict_pair_and_missing_inner_is_explicit(self):
        args=([[0,0,8,8]],[[0,0,8,8]],[1.],[2.],[True],[True])
        self.assertEqual(select(*args,'outer-first'),(0,None))
        self.assertEqual(select(*args,'joint-sum'),(0,None))
        self.assertEqual(select(*args,'maximal-first'),(0,None))

    def test_masked_zero_slots_are_not_crop_geometry_and_ties_use_first_index(self):
        args=([[0,0,8,8],[0,0,0,0]],[[1,1,3,3],[2,2,4,4],[0,0,0,0]],[1.,100.],[2.,2.,100.],[True,False],[True,True,False])
        self.assertEqual(select(*args,'outer-first'),(0,0))

    def test_all_policies_agree_with_independent_exhaustive_small_bank_oracle(self):
        outer=[[x,0,x+6,6] for x in range(5)];inner=[[x,y,x+2,y+2] for x in range(9) for y in range(5)]
        for seed in range(19):
            os=[float((k*17+seed)%11) for k in range(5)];ins=[float((k*13+seed)%17) for k in range(45)]
            om=[(k+seed)%3!=0 for k in range(5)];im=[(k+seed)%4!=0 for k in range(45)];raw_o=sorted([k for k in range(5) if om[k]],key=lambda k:(-os[k],k))[0];raw_i=sorted([k for k in range(45) if im[k]],key=lambda k:(-ins[k],k))[0]
            feasible=[(x,y) for x in range(5) for y in range(45) if om[x] and im[y] and all([inner[y][0]>=outer[x][0],inner[y][1]>=outer[x][1],inner[y][2]<=outer[x][2],inner[y][3]<=outer[x][3]])]
            expected=sorted(feasible,key=lambda xy:(-os[xy[0]]-ins[xy[1]],xy[0],xy[1]))[0] if feasible else (raw_o,None)
            self.assertEqual(select(outer,inner,os,ins,om,im,'joint-sum'),expected)
            children=sorted([y for x,y in feasible if x==raw_o],key=lambda y:(-ins[y],y))
            self.assertEqual(select(outer,inner,os,ins,om,im,'outer-first'),(raw_o,children[0] if children else None))
            self.assertEqual(select(outer,inner,os,ins,om,im,'independent'),(raw_o,raw_i))
            parents=sorted([x for x,y in feasible if y==raw_i],key=lambda x:(-os[x],x))
            self.assertEqual(select(outer,inner,os,ins,om,im,'maximal-first'),(parents[0],raw_i) if parents else (raw_o,None))


if __name__=='__main__':unittest.main()
