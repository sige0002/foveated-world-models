"""Independent ridge readouts; fit only on training episodes, every 5000 steps."""
from pathlib import Path
import json
import numpy as np
import torch
from research.evaluate_robotwin_matched_state_probes import encode,select_ridge,predict,posterior_parity
from research.evaluate_robotwin_loss_ablation_fp32_v2 import digest

def run(model, validation, codes, split_path):
 path=Path('/media/sadasue/SSD1/foveated-world-models/runs/robotwin-loss-ablation-b4-20261114/evaluations/matched-state-probes-fp32-20261114/train2048-highest-fp32-features.pt')
 train=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
 split=json.loads(Path(split_path).read_text())['ids']
 assert all(r['episode'] in set(split['train']) for r in train['rows'])
 assert not {r['episode'] for r in train['rows']}&{r['episode'] for r in validation['rows']}
 assert train['precision']=='highest FP32; TF32 disabled'
 with torch.inference_mode():
  parity=posterior_parity(model,train); encoded=encode(model,train,2)
 state=train['state'].numpy(); target=train['delta'].numpy()
 vs=validation['states'][:,0,:16].numpy(); vt=validation['delta'][:,:16].numpy()
 unique=np.unique([r['episode'] for r in train['rows']]);rng=np.random.default_rng(42);rng.shuffle(unique);mapping={int(e):i%5 for i,e in enumerate(unique)}
 folds=np.array([mapping[int(r['episode'])] for r in train['rows']]);alphas=[1e-5,1e-3,.1,1.,10.]
 options={'state':(state,vs),'state_z':(np.concatenate([state,encoded['z'].numpy()],1),np.concatenate([vs,codes['mu'].flatten(1).numpy()],1)),'state_current_z':(np.concatenate([state,encoded['current_only_z'].numpy()],1),np.concatenate([vs,codes['current_only_mu'].flatten(1).numpy()],1))}
 result={'train_cache_sha256':digest(path),'parity':parity,'training_samples':len(state),'validation_samples':len(vs),'fits':{}}
 groups={'all':list(range(16)),**validation['state_groups']}
 for name,(x,vx) in options.items():
  fitted,selection=select_ridge(x,target,folds,alphas);prediction=predict(fitted,vx);err=(prediction-vt)**2
  result['fits'][name]={'selection':selection,'mse':{g:float(err[:,cols].mean()) for g,cols in groups.items()}}
 return result
