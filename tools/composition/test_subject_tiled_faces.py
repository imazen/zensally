import unittest
from subject_tiled_faces import windows,extract,merge


class TiledFaces(unittest.TestCase):
    def test_overlapping_tiles_cover_source_and_extract_rows_exactly(self):
        w,h=7,5;rgb=bytes(range(w*h*3));boxes=windows(w,h)
        self.assertEqual(len(boxes),4)
        self.assertEqual({(x,y) for b in boxes for y in range(b[1],b[3]) for x in range(b[0],b[2])},{(x,y) for y in range(h) for x in range(w)})
        for b in boxes:
            expected=bytes(v for y in range(b[1],b[3]) for x in range(b[0],b[2]) for v in rgb[(y*w+x)*3:(y*w+x+1)*3])
            self.assertEqual(extract(rgb,w,h,b),expected)
        with self.assertRaises(ValueError):extract(rgb[:-1],w,h,boxes[0])

    def test_nms_retains_highest_confidence_and_distinct_people(self):
        rows=[dict(box=[.1,.1,.3,.3],confidence=.9,origin='full'),dict(box=[.11,.11,.31,.31],confidence=.8,origin='tile'),dict(box=[.7,.7,.8,.8],confidence=.7,origin='tile')]
        self.assertEqual(merge(rows),[rows[0],rows[2]])
        self.assertEqual(merge(rows[::-1]),merge(rows))


if __name__=='__main__':unittest.main()
