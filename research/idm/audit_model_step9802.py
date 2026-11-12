"""Immutable named9802 baseline audit on existing features; no images/GPU/training."""
import json,hashlib,sys,time,os
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
from research.idm.cache_reference_queries import sha256
from research.idm.cache_unused_episodes import state_hash
from latent_action_model.core.utils.lam_encoder import LAMEncoder
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
from latent_action_model.core.vq import VAEQuantizer
@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.set_float32_matmul_precision('medium');started=time.perf_counter()
 run=Path('/media/sadasue/SSD1/foveated-world-models/runs/stage1-gb10-main-20261112');named=run/'checkpoints/epoch=2-step=9802.ckpt';out=ROOT/'runs/frozen-model-step9802';resultpath=ROOT/'research/idm/model-step9802-baseline-audit.json'
 if out.exists() or resultpath.exists():raise FileExistsError('Use new outputs')
 cfg=yaml.safe_load((run/'resolved_config.yaml').read_text());oldcfg=yaml.safe_load((ROOT/'runs/frozen-idm-step6501/resolved_config.yaml').read_text());assert cfg['model']==oldcfg['model'] and cfg['expected_vision']==oldcfg['expected_vision']
 # Hold the same open descriptor for load and byte hash, independent of pathname changes.
 with named.open('rb') as handle:
  st=os.fstat(handle.fileno());loaded=torch.load(handle,map_location='cpu',weights_only=True);handle.seek(0);digest=hashlib.sha256()
  for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
  assert os.fstat(handle.fileno()).st_size==st.st_size and os.fstat(handle.fileno()).st_mtime_ns==st.st_mtime_ns
 named_hash=digest.hexdigest();assert loaded['global_step']==9802;state=loaded['state_dict'];del loaded
 old=torch.load(ROOT/'runs/frozen-idm-step6501/reference_weights.pt',weights_only=True,mmap=True)['state_dict'];vision=state_hash(state,'lam.vision_encoder.');assert vision==state_hash(old,'lam.vision_encoder.');del old
 out.mkdir();torch.save(dict(global_step=9802,state_dict=state),out/'reference_weights.pt');(out/'resolved_config.yaml').write_text(yaml.safe_dump(cfg));snapshot_hash=sha256(out/'reference_weights.pt');print('Snapshot saved; vision/config equal',flush=True)
 c=cfg['model'];enc=LAMEncoder(context_dim=c['dim'],input_dim=768,num_heads=c['num_heads'],num_queries=c['num_queries'],grid_hw=tuple(v//c['patch_size'] for v in c['image_hw']),ffn_expansion_factor=c['ffn_expansion_factor'],dropout=c['dropout'],code_dim=c['code_dim'],num_layers=c['enc_layers'],num_frames=c['num_frames'],modal_mask=c['enc_modal_mask'],add_state=False,max_state_dim=c['max_state_dim'],num_embodiments=c['num_embodiments']).eval().requires_grad_(False)
 vq=VAEQuantizer(code_dim=c['code_dim'],**c['vq_kwargs']).eval().requires_grad_(False)
 dec=LAMDecoder_v2(context_dim=c['dim'],input_dim=768,code_dim=c['code_dim'],num_queries=c['num_queries'],num_layers=c['dec_layers'],num_heads=c['num_heads'],dropout=c['dropout'],grid_hw=tuple(v//c['patch_size'] for v in c['image_hw']),ffn_expansion_factor=c['ffn_expansion_factor'],last_ln=c.get('decoder_last_ln',True)).eval().requires_grad_(False)
 for name,model in [('encoder',enc),('vq',vq),('decoder',dec)]:
  prefix='lam.'+name+'.';model.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
 del state
 sources={'validation':(ROOT/'runs/reference-idm-with-prior6501','validation'),'observed96':(ROOT/'runs/research-unused-96-prior6501','validation'),'train':(ROOT/'runs/reference-idm-with-prior6501','train')};source_records={};results={};output_hashes={}
 for name,(parent,split) in sources.items():
  pm=json.loads((parent/'cache_manifest.json').read_text());path=parent/f'{split}.pt';input_hash=sha256(path);assert input_hash==pm['cache_sha256'][split];data=torch.load(path,weights_only=True);codes=[]
  torch.set_float32_matmul_precision('medium')
  for i in range(len(data['rows'])):
   nodes=enc(torch.cat((data['current'][i:i+1],data['future'][i:i+1]),1));mu,_=vq._encode(nodes);codes.append(mu)
   if (i+1)%32==0:print(name,'encoder',i+1,flush=True)
  codes=torch.cat(codes);assert torch.isfinite(codes).all();torch.set_float32_matmul_precision('highest');priors=[];rows=[]
  for start in range(0,len(codes),2):
   sl=slice(start,start+2);h,y=data['current'][sl],data['future'][sl];pred=dec(h,codes[sl]);prior=dec(h,torch.zeros_like(codes[sl]));priors.append(prior)
   error=(pred-y).square().flatten(1).mean(1);zero=(prior-y).square().flatten(1).mean(1);oldzero=(data['baseline'][sl]-y).square().flatten(1).mean(1)
   for j in range(len(h)):rows.append(dict(source_row=data['rows'][start+j],original9802_mse=float(error[j]),zero9802_mse=float(zero[j]),zero6501_mse=float(oldzero[j])))
  baseline=torch.cat(priors);export=dict(current=data['current'],future=data['future'],codes=codes,baseline=baseline,rows=data['rows']);torch.save(export,out/f'{name}.pt');check=torch.load(out/f'{name}.pt',weights_only=True)
  for key in ('current','future'):assert torch.equal(check[key],data[key])
  assert check['rows']==data['rows'] and torch.isfinite(baseline).all();output_hashes[name]=sha256(out/f'{name}.pt')
  results[name]=dict(count=len(rows),original9802_mse=sum(r['original9802_mse'] for r in rows)/len(rows),zero9802_mse=sum(r['zero9802_mse'] for r in rows)/len(rows),zero6501_mse=sum(r['zero6501_mse'] for r in rows)/len(rows),rows=rows)
  source_records[name]=dict(path=str(path),sha256=input_hash,manifest_sha256=sha256(parent/'cache_manifest.json'),h_y_rows_preserved=True);assert sha256(path)==input_hash;print(name,'complete', {k:v for k,v in results[name].items() if k!='rows'},flush=True)
 metadata=dict(checkpoint_step=9802,named_checkpoint=str(named),named_checkpoint_sha256=named_hash,snapshot_sha256=snapshot_hash,model_config=c,config_sha256=sha256(run/'resolved_config.yaml'),vision_state_sha256=vision,vision_state_equals6501=True,model_config_equals6501=True,feature_normalization=dict(enabled=c['norm_latents'],type=c['norm_latents_type'],latent_layer=c['latent_layer_to_use'],verification='Identical model config and complete vision_encoder state including normalization buffers; sourcecode hashes recorded, no vision recomputation'),source_caches=source_records,cache_sha256=output_hashes,precision='Encoder/VAE medium float32 B1; decoder highest float32 B2; CPU4',conditioning_mean='Original named9802 true-future posterior mu; baseline=original9802 decoder(current,zero code)',main_run_summary=json.loads((run/'run_summary.json').read_text()),script_sha256=sha256(Path(__file__)),source_code_sha256={p:sha256(ROOT/p) for p in ('vendor/lawam/latent_action_model/core/lam_model.py','vendor/lawam/latent_action_model/core/vjepa_encoder.py','vendor/lawam/latent_action_model/core/utils/lam_encoder.py','vendor/lawam/latent_action_model/core/utils/lam_decoder.py')},seconds=time.perf_counter()-started,scope='Named9802 snapshot, not final10000; original sources unchanged. Observed96 is research-observed, not independent finaltest. Cached rows preserve source metadata, including any historical reference_mse fields; use this audit result for9802 errors.')
 (out/'cache_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n');resultpath.write_text(json.dumps(dict(metadata=metadata,results=results),indent=2)+'\n');print('COMPLETE',flush=True)
if __name__=='__main__':main()
