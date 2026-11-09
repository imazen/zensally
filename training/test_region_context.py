import unittest
from region_context import descriptor,FEATURES


class RegionContext(unittest.TestCase):
    def test_order_and_exact_duplicate_regions_do_not_change_descriptor(self):
        boxes=[[.1,.2,.3,.4],[.6,.1,.9,.8]]
        a,_=descriptor(boxes);b,count=descriptor([boxes[1],boxes[0],boxes[1]])
        self.assertEqual(a,b);self.assertEqual(count['input_regions'],3);self.assertEqual(count['unique_regions'],2)

    def test_capacity_omits_slots_but_preserves_all_region_count_and_extent(self):
        boxes=[[i/20,0,(i+1)/20,.2] for i in range(12)]
        vector,count=descriptor(boxes)
        self.assertEqual(len(vector),FEATURES);self.assertEqual(count['omitted_slots'],4)
        self.assertEqual(vector[-7:-3],[0.,0.,.6,.2])
        self.assertGreater(vector[-1],2)

    def test_empty_and_invalid_inputs(self):
        vector,count=descriptor([]);self.assertEqual(vector,[0.]*FEATURES);self.assertEqual(count['retained_slots'],0)
        for box in [[0,0,0,1],[-.1,0,1,1],[0,0,float('nan'),1]]:
            with self.assertRaises(ValueError):descriptor([box])


if __name__=='__main__':unittest.main()
