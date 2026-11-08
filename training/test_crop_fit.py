import hashlib,json,tempfile,unittest
from pathlib import Path
import torch
from crop_fit import CropDataset,TrainingModel,load_manifest,loss,MODEL_ARGS

class FittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def scratch(self):
        p=Path.home()/'tmp';p.mkdir(exist_ok=True);return tempfile.TemporaryDirectory(dir=p)
    def fixture(self,root):
        raw=bytes([20,40,60])*(256*256);p=root/'tensor.rgb';p.write_bytes(raw)
        source={'id':'11000','family_id':'1000','bucket':'train','use':'fit','historical_eval_family':False,'width':640,'height':480,'valid_rect':[0.,.125,1.,.875],'tensor_path':str(p),'tensor_sha256':hashlib.sha256(raw).hexdigest(),'training_eligible':True,'human_rated':False,'face_safety_certified':False,'face_context':[0,0,0,0,0,0],
                'candidates':[{'task_id':'one','box':[0.,0.,.5,1.],'aspect':2/3,'mode':0,'acceptable':True,'weight':1.},{'task_id':'one','box':[.5,0.,1.,1.],'aspect':2/3,'mode':0,'acceptable':False,'weight':1.}],'pairs':[{'preferred':0,'other':1,'weight':1.}]}
        data={'schema':1,'supervision':'noisy_vlm_rendered_pixels','sources':[source]};path=root/'manifest.json';path.write_text(json.dumps(data));return path,data
    def test_canonical_holdout_and_historical_exposure_rejected(self):
        with self.scratch() as name:
            p,d=self.fixture(Path(name))
            d['sources'][0]['bucket']='validate';p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'held-out'):load_manifest(p)
            d['sources'][0]['bucket']='train';d['sources'][0]['historical_eval_family']=True;p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'held-out'):load_manifest(p)
    def test_eval_cannot_supply_gradients_and_tensor_identity_is_checked(self):
        with self.scratch() as name:
            p,d=self.fixture(Path(name))
            with self.assertRaises(ValueError):CropDataset(p,'dev',training=True)
            Path(d['sources'][0]['tensor_path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'tensor'):CropDataset(p,'fit',training=True)
    def test_pairs_cannot_cross_ratio_mode_tasks(self):
        with self.scratch() as name:
            p,d=self.fixture(Path(name));d['sources'][0]['candidates'][1]['task_id']='different';p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'crosses'):CropDataset(p,'fit',training=True)
    def test_weak_training_logits_do_not_open_serving_face_guard(self):
        with self.scratch() as name:
            p,d=self.fixture(Path(name));dataset=CropDataset(p,'fit',training=True);batch={k:v.unsqueeze(0) for k,v in dataset[0].items() if isinstance(v,torch.Tensor)};model=TrainingModel();model.eval()
            raw=model(**{k:batch[k] for k in MODEL_ARGS});self.assertTrue(torch.isfinite(raw).all())
            score=model.ranker(batch['image'],batch['boxes'],batch['source_aspect'],batch['target_aspect'],batch['mode'],batch['face_context'],torch.ones(1,128,dtype=torch.bool),batch['valid_rect'])
            self.assertTrue(torch.isneginf(score[0]).all());value,_,_=loss(raw,batch);value.backward();self.assertTrue(torch.isfinite(value));self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))
    def test_reject_all_has_acceptability_gradient_without_ranking(self):
        raw=torch.zeros(1,2,2,requires_grad=True);batch={'targets':torch.zeros(1,2),'weights':torch.ones(1,2),'pairs':torch.zeros(1,1,2,dtype=torch.long),'pairweights':torch.zeros(1,1)}
        value,accept,rank=loss(raw,batch);self.assertAlmostEqual(float(rank),0.);value.backward();self.assertTrue((raw.grad[...,1]>0).all());self.assertTrue((raw.grad[...,0]==0).all())
if __name__=='__main__':unittest.main()
