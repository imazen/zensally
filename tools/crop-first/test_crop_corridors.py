import unittest
from crop_corridors import audit,contains,domains,member,samples


class CropCorridors(unittest.TestCase):
    def test_domains_equal_all_nested_integer_crops_and_preserve_faces(self):
        inner,outer=[7,5,13,9],[1,1,19,13];faces=[[8,6,10,8],[0,14,2,16]]
        self.assertTrue(audit(inner,outer,20,16,'3:2',faces)['geometrically_admissible'])
        actual={(x,y,x+r['width'],y+r['height']) for r in domains(inner,outer,'3:2') for x in range(r['x'][0],r['x'][1]+1) for y in range(r['y'][0],r['y'][1]+1)}
        expected={(x,y,x+3*k,y+2*k) for k in range(1,7) for x in range(21-3*k) for y in range(17-2*k) if member([x,y,x+3*k,y+2*k],inner,outer,'3:2')}
        self.assertEqual(actual,expected)
        for box in actual:self.assertTrue(contains(box,faces[0]));self.assertLessEqual(box[3],faces[1][1])
        self.assertEqual(samples(inner,outer,'3:2')[0],inner);self.assertEqual(samples(inner,outer,'3:2')[-1],outer)

    def test_endpoints_can_be_safe_while_the_interval_crosses_a_face(self):
        inner,outer=[7,5,13,9],[1,1,19,13]
        result=audit(inner,outer,20,16,'3:2',[[2,2,5,4]])
        self.assertEqual(result['faces_crossing_crop_interval'],[0]);self.assertFalse(result['geometrically_admissible'])
        self.assertFalse(audit([7,5,14,9],outer,20,16,'3:2',[])['exact_aspect'])


if __name__=='__main__':unittest.main()
