"""Reproduce the upstream Stage-1 state-target range on real RoboTwin samples.

Use PYTHONPATH pointing to the pinned upstream checkout and its dependencies.
No training, source modifications, or checkpoint downloads.
"""
import argparse
import json
from pathlib import Path
import torch
from latent_action_model.data_loader.lerobot_datamodule import LeRobotDataModule
from latent_action_model.core.utils.lam_decoder import StatePredictor
from latent_action_model.core.utils.utils import eef_reconstruction_loss

p=argparse.ArgumentParser()
p.add_argument('--data-root',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
torch.set_num_threads(2)
dm=LeRobotDataModule(data_root_dir=a.data_root,data_mix='robotwin_merged',
 num_frames=2,frame_dt_sec=1.6,batch_size=16,num_workers=0,
 video_backend='pyav',max_state_dim=32)
dm.setup('fit')
b=next(iter(dm.train_dataloader()))
delta=b['delta_proprio']; mask=b['state_mask'].any(1)
assert torch.equal(delta,b['states'][:,1]-b['states'][:,0])
assert (b['embodiment_ids']==1).all()
# Small hidden width tests the exact upstream forward function, not learned quality.
g=StatePredictor(latent_dim=8,code_dim=32,max_state_dim=32,num_embodiments=32)
with torch.no_grad():
 for param in g.parameters(): param.zero_()
 g.state_decoder.layer2.b.fill_(20)
 y=g(torch.zeros(16,1,32),b['states'][:,0],b['embodiment_ids'])
assert (y==1).all()
clipped=delta.clamp(-1,1)
floor=eef_reconstruction_loss(clipped,delta,mask,state_loss_type='l2')
examples=[]
for i in range(16):
 if (delta[i,[7,15]].abs()>1).any():
  examples.append(dict(episode=int(b['trajectory_ids'][i]),base=int(b['base_indices'][i]),
    gripper_start=b['states'][i,0,[7,15]].tolist(),gripper_end=b['states'][i,1,[7,15]].tolist(),
    gripper_delta=delta[i,[7,15]].tolist()))
report=dict(samples=16,examples=examples,target_max_abs=float(delta.abs().max()),
 output_with_preactivation20=float(y.max()),batch_state_l2_bound=float(floor),
 source_modules=dict(state_predictor=__import__(StatePredictor.__module__,fromlist=['']).__file__),
 note='Bound computed by projecting teacher to [-1,1]; not trained model error. g hidden width 8 for forward-path probe.')
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
