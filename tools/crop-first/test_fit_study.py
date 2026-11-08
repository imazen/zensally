import unittest
from fit_study import rectangles

class GeometryTests(unittest.TestCase):
    def test_integer_aspect_minimal_area_and_face_veto(self):
        source={'id':'fixture','width':1001,'height':701}
        zones={'rois':[{'id':'face','kind':'face_head','box':{'left':0,'top':0,'right':10,'bottom':10}}],
               'crops':[{'ratio':r,'mode':m,'action':'contain','options':[]} for r in ['1:3','3:1','1:6','6:1'] for m in ['minimal','maximal']]}
        tasks=rectangles(source,zones)
        self.assertEqual(len(tasks),8)
        for t in tasks:
            rw,rh=map(int,t['ratio'].split(':'));k=min(1001//rw,701//rh)
            for c in t['candidates']:
                x,y,w,h=c['xywh'];self.assertEqual(w*rh,h*rw);self.assertTrue(0<=x<=1001-w and 0<=y<=701-h)
                if t['mode']=='minimal':self.assertEqual((w,h),(rw*k,rh*k))
                self.assertEqual(bool(c['missed_proposed_faces']),not(x<=0 and y<=0 and x+w>=101 and y+h>=71))
    def test_duplicate_proposal_not_presented_twice(self):
        source={'id':'fixture','width':600,'height':600};box={'left':0,'top':0,'right':100,'bottom':100}
        z={'rois':[],'crops':[{'ratio':'1:1','mode':'minimal','action':'crop','options':[{'box':box},{'box':box}]}]}
        self.assertEqual(len(rectangles(source,z)[0]['candidates']),1)
if __name__=='__main__':unittest.main()
