"""Metadata/NPZ-only measured endpoint displacements; no video decoding or fitting."""
import json,sys
from pathlib import Path
import numpy as np
import torch,yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from vendor.dataset.robomind import RoboMINDDataset,STATE_SCHEMA
from research.idm.cache_reference_queries import sha256
SELECT=[0,1,2,9,10,11,12,19]
def observed_state_mask(valid):
 if valid.shape[-1]!=15:raise ValueError('Expected video+14 original validity fields')
 parts=[]
 for start in (1,8):
  parts.extend((valid[...,start:start+3],np.repeat(valid[...,start+3:start+6].all(-1,keepdims=True),6,axis=-1),valid[...,start+6:start+7]))
 return np.concatenate(parts,-1)
def labels_from_endpoints(states,valid):
 observed=observed_state_mask(valid)&np.isfinite(states)
 state=np.where(observed,states,np.nan).astype(np.float32)
 mask=observed[0,SELECT]&observed[1,SELECT]
 delta=np.where(mask,state[1,SELECT]-state[0,SELECT],np.nan).astype(np.float32)
 structural=np.zeros((2,32),dtype=bool);structural[:,:20]=True
 return dict(endpoint_states20=torch.from_numpy(state.copy()),current_state20=torch.from_numpy(state[0].copy()),endpoint_state_observed_mask=torch.from_numpy(observed.copy()),current_state_observed_mask=torch.from_numpy(observed[0].copy()),delta8=torch.from_numpy(delta),delta8_mask=torch.from_numpy(mask),original_valid=torch.from_numpy(valid.copy()),structural_state_mask32=torch.from_numpy(structural))
def main():
 torch.set_num_threads(2);out=ROOT/'research/idm/observed-displacement-labels'
 if out.exists():raise FileExistsError(out)
 cfg=yaml.safe_load((ROOT/'runs/frozen-idm-step6501/resolved_config.yaml').read_text())
 sources={'train':(ROOT/'runs/reference-idm-with-prior6501','train'),'validation':(ROOT/'runs/reference-idm-with-prior6501','validation'),'observed96':(ROOT/'runs/research-unused-96-prior6501','validation')}
 ds={};manifests={}
 for split,key in [('train','train_manifest'),('validation','val_manifest')]:
  path=Path(cfg['data'][key]);manifests[split]=dict(path=str(path),sha256=sha256(path))
  ds[split]=RoboMINDDataset(path,horizon_seconds=cfg['data']['horizon_seconds'],tolerance_seconds=cfg['data']['tolerance_seconds'],require_observed_endpoints=True)
 extracted={};source_records={};array_hashes={};episode_sets={}
 for name,(cache,split) in sources.items():
  metadata=json.loads((cache/'cache_manifest.json').read_text());file=cache/f'{split}.pt';digest=sha256(file)
  assert digest==metadata['cache_sha256'][split]
  assert metadata['manifests'][split]['sha256']==manifests[split]['sha256']
  data=torch.load(file,weights_only=True);dataset=ds[split];collected=[];rows=[]
  for row in data['rows']:
   manifest_i,a,b,tail=dataset.pairs[row['index']];entry=dataset.rows[manifest_i];arrays=dataset._arrays(manifest_i);dt=float(arrays['timestamps'][b]-arrays['timestamps'][a])
   assert str(entry['episode_id'])==str(row['episode']) and abs(dt-row['dt'])<1e-9
   if 'endpoint_indices' in row:assert row['endpoint_indices']==[a,b]
   assert entry['state_schema']==STATE_SCHEMA
   states=arrays['states'][[a,b]];valid=arrays['original_valid'][[a,b]]
   assert states.shape==(2,20) and valid.shape==(2,15)
   assert valid.all(), 'Cached observed-endpoint pair no longer observed'
   item=labels_from_endpoints(states,valid)
   item['timestamps']=torch.from_numpy(arrays['timestamps'][[a,b]].copy());item['fill_age_seconds']=torch.from_numpy(arrays['fill_age_seconds'][[a,b]].copy())
   assert not (item['fill_age_seconds']>0).any(), 'Observed endpoint unexpectedly filled'
   path=dataset.path.parent/entry['array_file']
   if str(path) not in array_hashes:array_hashes[str(path)]=sha256(path)
   collected.append(item);rows.append(dict(row,endpoint_indices=[a,b],array_file=str(path),manifest_row=manifest_i,state_schema=STATE_SCHEMA))
  extracted[name]={k:torch.stack([v[k] for v in collected]) for k in collected[0]};extracted[name]['rows']=rows
  episode_sets[name]={str(r['episode']) for r in rows}
  source_records[name]=dict(cache=str(cache),split=split,cache_sha256=digest,cache_manifest_sha256=sha256(cache/'cache_manifest.json'))
  print(name,len(rows),flush=True)
 names=list(episode_sets)
 for i,a in enumerate(names):
  for b in names[i+1:]:assert not episode_sets[a]&episode_sets[b]
 for name,(cache,split) in sources.items():assert sha256(cache/f'{split}.pt')==source_records[name]['cache_sha256']
 for path,digest in array_hashes.items():assert sha256(Path(path))==digest
 out.mkdir()
 for name,values in extracted.items():torch.save(values,out/f'{name}.pt')
 info=Path('/media/sadasue/SSD1/foveated-world-models/datasets/RoboMIND-lerobot/franka_fr3_dual/meta/info.json');features=json.loads(info.read_text())['features']
 report=dict(source_caches=source_records,source_manifests=manifests,array_sha256=array_hashes,script_sha256=sha256(Path(__file__)),source_code_sha256={p:sha256(ROOT/p) for p in ('vendor/dataset/robomind.py','vendor/dataset/prepare_robomind.py','vendor/lawam/latent_action_model/data_loader/collate.py')},metadata_path=str(info),metadata_sha256=sha256(info),metadata_features={k:features[k] for k in ('observation.states.end_effector','observation.states.gripper')},state_schema=STATE_SCHEMA,indices=SELECT,label_names=['left_dx_m','left_dy_m','left_dz_m','left_dgripper_source_units','right_dx_m','right_dy_m','right_dz_m','right_dgripper_source_units'],counts={k:len(v['rows']) for k,v in extracted.items()},valid_delta_coordinates={k:int(v['delta8_mask'].sum()) for k,v in extracted.items()},all_endpoints_observed=True,episode_overlap=0,threads=2,video_decoding=False,fitting=False,output_sha256={p.name:sha256(p) for p in out.glob('*.pt')},limitations='Measured endpoint differences, not commands, velocities or joint angles. xyz meters in each arm source coordinate frame; shared frame not established. Gripper source values validated in[0,1], physical units/normalization semantics not established. Current20D includes xyz,first2 rotation columns and gripper per arm. Structural state mask is dimension presence, separate from observed validity. Missing values masked toNaN without new filling; selected sources require observed endpoints. observed96 has already been used for research evaluation, not untouched/external test.')
 (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('COMPLETE',report['counts'])
if __name__=='__main__':main()
