"""Load only pinned local teacher code; exact checkpoint key coverage required."""
import importlib,importlib.util,sys,types
from pathlib import Path
import torch

def load(kind,root):
 torch.hub.set_dir(str(root/'torch-cache'))
 original=torch.hub.load
 if kind=='gaze':
  sys.path.insert(0,str(root/'gazelle'))
  def hub(repo,model,*args,**kwargs):
   if repo!='facebookresearch/dinov2' or model!='dinov2_vitb14':raise ValueError('unexpected teacher hub request')
   return original(str(root/'dinov2'),model,source='local',pretrained=True)
  torch.hub.load=hub
  try:
   from gazelle.model import get_gazelle_model
   model,_=get_gazelle_model('gazelle_dinov2_vitb14_inout')
  finally:torch.hub.load=original
  state=torch.load(root/'gazelle-vitb-inout.pt',map_location='cpu',weights_only=True)
  expected={k for k in model.state_dict() if not k.startswith('backbone')}
  if set(state)!=expected:raise ValueError('gaze checkpoint key coverage differs')
  model.load_gazelle_state_dict(state)
 else:
  package=types.ModuleType('pinned_legacy_vision');package.__path__=[str(root)];sys.modules[package.__name__]=package
  utils=types.ModuleType('pinned_legacy_vision.utils');utils.load_state_dict_from_url=torch.hub.load_state_dict_from_url;sys.modules[utils.__name__]=utils
  spec=importlib.util.spec_from_file_location('pinned_legacy_vision.densenet',root/'legacy_densenet.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  package=types.ModuleType('deepgaze_pytorch');package.__path__=[str(root/'DeepGaze/deepgaze_pytorch')];sys.modules['deepgaze_pytorch']=package
  def hub(repo,model,*args,**kwargs):
   if (repo,model)!=('pytorch/vision:v0.6.0','densenet201'):raise ValueError('unexpected teacher hub request')
   # The complete DeepGaze checkpoint supplies this encoder's weights.
   return module.densenet201(pretrained=False)
  torch.hub.load=hub
  try:model=importlib.import_module('deepgaze_pytorch.deepgaze3').DeepGazeIII(pretrained=False)
  finally:torch.hub.load=original
  model.load_state_dict(torch.load(root/'deepgaze3.pth',map_location='cpu',weights_only=True),strict=True)
 model.eval()
 for p in model.parameters():p.requires_grad=False
 return model
