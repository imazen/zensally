import unittest
from proposals import candidate_pool


class ProposalTests(unittest.TestCase):
    def test_minimal_includes_center_and_neighbors_without_endpoint_pair(self):
        for w,h in [(4032,3024),(900,3200)]:
            pool=candidate_pool(w,h,'minimal');k=min(w//9,h//16);cw,ch=9*k,16*k
            self.assertGreaterEqual(len(pool),5)
            self.assertTrue(any(abs(c['x']-(w-cw)/2)<=.5 and abs(c['y']-(h-ch)/2)<=.5 for c in pool))
            for c in pool:
                self.assertEqual((c['w'],c['h']),(cw,ch))
                self.assertTrue(0<=c['x']<=w-cw and 0<=c['y']<=h-ch)
                if w>cw:self.assertTrue(0<c['x']<w-cw)
                if h>ch:self.assertTrue(0<c['y']<h-ch)

    def test_subject_proposal_supplies_missing_off_center_composition(self):
        base=candidate_pool(4000,3000,'minimal')
        # A left-side subject excluded by the center and all interior baselines.
        def retains(c):return c['x']<=250 and c['x']+c['w']>=450
        self.assertFalse(any(retains(c) for c in base))
        pool=candidate_pool(4000,3000,'minimal',[{'cx':.23,'cy':.5,'scale':1,'reason':'Retain the off-center subject'}])
        self.assertTrue(any(retains(c) and 'vision-proposal' in c['generators'] for c in pool))

    def test_duplicates_merge_and_maximal_can_retain_context(self):
        pool=candidate_pool(4000,3000,'maximal',[{'cx':.5,'cy':.5,'scale':1,'reason':'Whole basket context'}])
        self.assertEqual(len(pool),len({(c['x'],c['y'],c['w'],c['h']) for c in pool}))
        self.assertEqual(pool[0]['w'],3000)
        self.assertIn('vision-proposal',pool[0]['generators'])
        self.assertGreater(len({c['w'] for c in pool}),1)

    def test_invalid_hints_fail_instead_of_silently_relaxing_minimal(self):
        for hint in [{'cx':float('nan'),'cy':.5,'scale':1,'reason':''},{'cx':.5,'cy':.5,'scale':.75,'reason':''}]:
            with self.assertRaises(ValueError):candidate_pool(4000,3000,'minimal',[hint])

    def test_extreme_ratios_are_exact_and_minimal_area_is_maximum(self):
        import math
        for w,h in [(3000,4000),(9355,2481),(1080,5192),(900,239)]:
            for ratio in [(1,1),(9,16),(16,9),(1,4),(4,1),(1,8),(8,1),(2,16)]:
                p,q=ratio;g=math.gcd(p,q);p//=g;q//=g;k=min(w//p,h//q)
                for mode in ['minimal','maximal']:
                    for c in candidate_pool(w,h,mode,target_ratio=ratio):
                        self.assertEqual(c['w']*q,c['h']*p)
                        self.assertTrue(0<=c['x']<=w-c['w'] and 0<=c['y']<=h-c['h'])
                        if mode=='minimal':self.assertEqual((c['w'],c['h']),(k*p,k*q))


if __name__=='__main__':unittest.main()
