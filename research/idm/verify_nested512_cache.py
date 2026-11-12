"""Independent read-only validation of completed nested512 extraction; never restarts it."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from vendor.dataset.robomind import RoboMINDDataset
from research.idm.train_reference_heads import representation

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def selected_hash(state,prefix):
 h=hashlib.sha256();n=0
 for k in sorted(state):
  if k.startswith(prefix):
   v=state[k].detach().cpu().contiguous();h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes());n+=1
 assert n>0
 return h.hexdigest()
def cpu_finite(value):
 if isinstance(value,torch.Tensor):
  assert value.device.type=='cpu'
  if value.is_floating_point():assert torch.isfinite(value).all()
 elif isinstance(value,dict):
  for v in value.values():cpu_finite(v)
 elif isinstance(value,(tuple,list)):
  for v in value:cpu_finite(v)
@torch.inference_mode()
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cache',type=Path,default=ROOT/'runs/reference-idm42-nested512-prior10000');parser.add_argument('--output',type=Path,default=ROOT/'research/idm/nested512-independent-verification.json');args=parser.parse_args()
 if not (args.cache/'cache_manifest.json').exists():raise SystemExit('Extraction not complete: final cache_manifest.json absent. No input files modified; do not restart extraction.')
 if args.output.exists():raise FileExistsError(args.output)
 torch.set_num_threads(2);torch.set_float32_matmul_precision('medium');meta=json.loads((args.cache/'cache_manifest.json').read_text());protocol=json.loads((args.cache/'extraction_protocol.json').read_text());selection_path=Path(meta['selection_path']);selection=json.loads(selection_path.read_text());assert sha(selection_path)==meta['selection_sha256'];assert selection==protocol['selection']
 assert meta['extraction_source_sha256']==protocol['source_sha256']
 for path,digest in meta['extraction_source_sha256'].items():assert sha(path)==digest,path
 parent=Path(selection['parent']);pm=json.loads((parent/'cache_manifest.json').read_text());assert meta['nested_parent_cache_sha256']==pm['cache_sha256']==selection['parent_cache_sha256'];assert sha(selection['manifest'])==selection['manifest_sha256']
 for split,digest in meta['cache_sha256'].items():assert sha(args.cache/f'{split}.pt')==digest
 assert sha(args.cache/'validation.pt')==sha(parent/'validation.pt')==selection['parent_cache_sha256']['validation']
 retained=torch.load(parent/'train.pt',weights_only=True);expanded=torch.load(args.cache/'train.pt',weights_only=True);val=torch.load(args.cache/'validation.pt',weights_only=True);observedpath=ROOT/'runs/research-unused-96-prior6501/validation.pt';observedmeta=json.loads((observedpath.parent/'cache_manifest.json').read_text());assert sha(observedpath)==observedmeta['cache_sha256']['validation'];observed=torch.load(observedpath,weights_only=True)
 for data in (retained,expanded,val,observed):cpu_finite(data)
 assert len(val['rows'])==64 and len(observed['rows'])==96
 assert retained['rows']==selection['retained'] and len(retained['rows'])==128
 assert expanded['rows']==selection['retained']+selection['added'] and len(expanded['rows'])==512
 assert set(retained)==set(expanded)
 for key,value in retained.items():
  if isinstance(value,torch.Tensor):assert torch.equal(expanded[key][:128],value),key
  else:assert expanded[key][:128]==value,key
 train_e={str(r['episode']) for r in expanded['rows']};dev_e={str(r['episode']) for r in val['rows']};obs_e={str(r['episode']) for r in observed['rows']}
 assert len(train_e)==512 and len(dev_e)==64 and len(obs_e)==96 and not dev_e&obs_e and not train_e&(dev_e|obs_e)
 dataset=RoboMINDDataset(selection['manifest'],**selection['options']);npz_hashes={}
 for row in expanded['rows']:
  mi,a,b,_=dataset.pairs[row['index']];entry=dataset.rows[mi];arr=dataset._arrays(mi);assert str(entry['episode_id'])==row['episode'] and abs(float(arr['timestamps'][b]-arr['timestamps'][a])-row['dt'])<1e-9
  assert arr['original_valid'][[a,b]].all() and not (arr['fill_age_seconds'][[a,b]]>0).any()
  if 'endpoint_indices' in row:assert [a,b]==row['endpoint_indices']
  npz=dataset.path.parent/entry['array_file'];npz_hashes[str(npz)]=sha(npz)
 q=torch.load(args.cache/'additional_queries.pt',weights_only=True);cpu_finite(q);assert q['rows']==selection['added'];assert len(meta['chunks'])==24;seen=set();chunks=[]
 for chunk_i,item in enumerate(meta['chunks']):
  start=chunk_i*16;path=Path(item['path']);path=path if path.is_absolute() else ROOT/path;assert path.resolve()==(args.cache/f'chunk-{start:03d}.pt').resolve();assert path not in seen;seen.add(path);assert sha(path)==item['sha256'];chunk=torch.load(path,weights_only=True);cpu_finite(chunk);assert chunk['rows']==selection['added'][start:start+16]
  for key in ('current','future','baseline'):assert torch.equal(chunk[key],expanded[key][128+start:128+start+16]),(start,key)
  for key in ('query','reference'):assert torch.equal(chunk[key],q[key][start:start+16]),(start,key)
  chunks.append(dict(start=start,rows=16,hash_verified=True,joined_tensors_exact=True))
 assert len(list(args.cache.glob('chunk-*.pt')))==24
 export=meta['export'];calpath=ROOT/'research/idm/reference-query-full/train_calibration.pt';assert sha(calpath)==export['calibration_sha256'];cal=torch.load(calpath,weights_only=True);hp=Path(export['head_checkpoint']);assert sha(hp)==export['head_checkpoint_sha256'];hs=torch.load(hp,weights_only=True);assert hs['seed']==42 and hs['steps']==2000 and hs['variant']=='reference';assert selected_hash(hs['state_dict'],'head.')==export['head_state_sha256']
 head=nn.Sequential(nn.Linear(1024,256),nn.GELU(),nn.Linear(256,64)).eval().requires_grad_(False);head.load_state_dict({k[5:]:v for k,v in hs['state_dict'].items() if k.startswith('head.')},strict=True);del hs
 qold=torch.load(ROOT/'research/idm/reference-query-full/train.pt',weights_only=True);assert qold['rows']==retained['rows'];old_mu=head(representation(qold,cal,'reference')).chunk(2,-1)[0];x=representation(q,cal,'reference');new_mu=torch.cat([head(v).chunk(2,-1)[0] for v in x.split(128)])
 old_error=float((old_mu-retained['codes']).abs().max());new_error=float((new_mu-expanded['codes'][128:]).abs().max());assert torch.equal(old_mu,retained['codes']) and torch.equal(new_mu,expanded['codes'][128:]),(old_error,new_error)
 # Verify stored checkpoint tensors CPU-only without loading optimizer or instantiating vision.
 for path in (ROOT/'runs/frozen-idm-step2900/reference_weights.pt',ROOT/'runs/frozen-model-step10000/reference_weights.pt'):
  weights=torch.load(path,weights_only=True,mmap=True);cpu_finite(weights['state_dict'])
  if weights['global_step']==2900:
   assert selected_hash(weights['state_dict'],'lam.encoder.')==export['encoder_state_sha256'];vision_hash=selected_hash(weights['state_dict'],'lam.vision_encoder.')
  else:
   assert weights['global_step']==10000 and selected_hash(weights['state_dict'],'lam.vision_encoder.')==vision_hash
  del weights
 assert sha(args.cache/'train.pt')==meta['cache_sha256']['train'] and sha(args.cache/'validation.pt')==meta['cache_sha256']['validation']
 result=dict(all_checks_passed=True,retained128_all_fields_exact=True,additional384_selection_and_endpoint_alignment=True,chunks=chunks,additional_queries_sha256=sha(args.cache/'additional_queries.pt'),head_reproduction=dict(precision='float32 medium CPU2, B128',retained_max_abs_error=old_error,added_max_abs_error=new_error,all512_codes_exact=True),validation64_byte_hash_unchanged=True,train_episodes=512,validation_union_episodes=160,episode_overlap=0,all_observed_endpoints_no_fill=True,npz_sha256=npz_hashes,all_loaded_tensors_cpu=True,validation_source_sha256=sha(observedpath),manifest_sha256=sha(args.cache/'cache_manifest.json'),protocol_sha256=sha(args.cache/'extraction_protocol.json'),script_sha256=sha(Path(__file__)),ancestor_export_verification=export.get('verification'),frozen_head_training_pairs=meta.get('frozen_head_training_pairs'),ancestor_metadata_note='Inherited export.verification train128/validation64 and frozen_head_training_pairs128 describe the ancestor head export/training, NOT verification or head refit on512. This independent report verifies the expanded512 cache; head/calibration remain fixed.',limitations='No image/query re-extraction or decoder rerun. CPU placement of saved tensors and CPU-only extraction source are checked; this cannot retrospectively prove every hardware operation from tensor contents alone. No GPU queried or used by this audit.')
 args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('npz_sha256','chunks')},indent=2))
if __name__=='__main__':main()
