import json,unittest
from review import validate

class ReviewTests(unittest.TestCase):
    def fixture(self):
        task={'id':'11000','tasks':[{'task_id':'one','candidates':[{'id':'A'},{'id':'B'}]}]}
        label={'source_id':'11000','tasks':[{'task_id':'one','preferred':None,'reason':'neither','candidates':[{'candidate_id':c,'acceptable':False,'face_cut':'no','confidence':'high','reason':'poor composition'} for c in ['A','B']]}]}
        return task,label
    def response(self,label):return {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(label)}]}]}
    def test_reject_all_does_not_invent_a_winner(self):
        task,label=self.fixture();self.assertIsNone(validate(self.response(label),task)['tasks'][0]['preferred'])
    def test_face_loss_cannot_be_positive(self):
        task,label=self.fixture();label['tasks'][0]['candidates'][0].update(acceptable=True,face_cut='yes')
        with self.assertRaisesRegex(ValueError,'face-cut'):validate(self.response(label),task)
    def test_unknown_winner_and_missing_candidates_rejected(self):
        task,label=self.fixture();label['tasks'][0]['preferred']='unknown'
        with self.assertRaises(ValueError):validate(self.response(label),task)
        label['tasks'][0]['preferred']=None;label['tasks'][0]['candidates'].pop()
        with self.assertRaises(ValueError):validate(self.response(label),task)
    def test_wrong_source_is_rejected(self):
        task,label=self.fixture();label['source_id']='11001'
        with self.assertRaises(ValueError):validate(self.response(label),task)
if __name__=='__main__':unittest.main()
