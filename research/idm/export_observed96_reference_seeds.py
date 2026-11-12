"""Reextract immutable2900 cached-feature queries and export completed43/44 means."""
import json,sys,time
from pathlib import Path
import torch,yaml
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
from latent_action_model.core.utils.lam_encoder import LAMEncoder
from research.idm.cache_reference_queries import sha256
from research.idm.cache_unused_episodes import state_hash
from research.idm.train_reference_heads import representation
@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.set_float32_matmul_precision('medium')
 source=ROOT/'runs/research-unused-96-prior6501';qout=ROOT/'research/idm/observed96-reference-queries2900';dest={seed:ROOT/f'runs/research-observed96-idm{seed}-prior6501' for seed in (43,44)}
 if qout.exists() or any(p.exists() for p in dest.values()):raise FileExistsError('New query/output directories required')
 sm=json.loads((source/'cache_manifest.json').read_text());input_hash=sha256(source/'validation.pt');assert input_hash==sm['cache_sha256']['validation'];data=torch.load(source/'validation.pt',weights_only=True);assert len(data['rows'])==96
 calpath=ROOT/'research/idm/reference-query-full/train_calibration.pt';calhash=sha256(calpath);assert calhash==sm['calibration_sha256'];cal=torch.load(calpath,weights_only=True)
 snap=ROOT/'runs/frozen-idm-step2900';cfg=yaml.safe_load((snap/'resolved_config.yaml').read_text())['model'];weightfile=snap/'reference_weights.pt';weighthash=sha256(weightfile);saved=torch.load(weightfile,weights_only=True,mmap=True);assert saved['global_step']==2900
 encoderhash=state_hash(saved['state_dict'],'lam.encoder.');assert encoderhash==sm['encoder_state_sha256']
 enc=LAMEncoder(context_dim=cfg['dim'],input_dim=768,num_heads=cfg['num_heads'],num_queries=cfg['num_queries'],grid_hw=tuple(v//cfg['patch_size'] for v in cfg['image_hw']),ffn_expansion_factor=cfg['ffn_expansion_factor'],dropout=cfg['dropout'],code_dim=cfg['code_dim'],num_layers=cfg['enc_layers'],num_frames=cfg['num_frames'],modal_mask=cfg['enc_modal_mask'],add_state=False,max_state_dim=cfg['max_state_dim'],num_embodiments=cfg['num_embodiments']).eval().requires_grad_(False)
 prefix='lam.encoder.';enc.load_state_dict({k[len(prefix):]:v for k,v in saved['state_dict'].items() if k.startswith(prefix)},strict=True);del saved
 heads={};exports={}
 for seed in (42,43,44):
  mpath=ROOT/f'research/idm/reference-codes-seed{seed}/cache_manifest.json';meta=json.loads(mpath.read_text());export=meta['export'];assert export['seed']==seed and export['steps']==2000 and export['variant']=='reference';assert export['encoder_state_sha256']==encoderhash and export['calibration_sha256']==calhash
  hp=Path(export['head_checkpoint']);assert sha256(hp)==export['head_checkpoint_sha256'];ck=torch.load(hp,weights_only=True);assert ck['seed']==seed and ck['steps']==2000 and ck['variant']=='reference';assert state_hash(ck['state_dict'],'head.')==export['head_state_sha256']
  head=nn.Sequential(nn.Linear(1024,256),nn.GELU(),nn.Linear(256,64)).eval().requires_grad_(False);head.load_state_dict({k[len('head.'):]:v for k,v in ck['state_dict'].items() if k.startswith('head.')},strict=True);heads[seed]=head;exports[seed]=dict(export,source_head_manifest_sha256=sha256(mpath));del ck
 capture={};original=enc.out_proj.forward
 def forward(q):capture['query']=q.clone();return original(q)
 enc.out_proj.forward=forward;queries=[];references=[];start=time.perf_counter()
 for i in range(96):
  x,y=data['current'][i:i+1],data['future'][i:i+1];enc(torch.cat((x,y),1));queries.append(capture['query']);enc(torch.cat((x,x),1));references.append(capture['query'])
  if (i+1)%16==0:print('Extracted',i+1,'/96',flush=True)
 enc.out_proj.forward=original;q=dict(query=torch.cat(queries),reference=torch.cat(references),rows=data['rows']);normalized=representation(q,cal,'reference')
 def codes(seed):return torch.cat([heads[seed](normalized[i:i+64]).chunk(2,-1)[0] for i in range(0,96,64)])
 mu42=codes(42);maxerr=float((mu42-data['codes']).abs().max());print('HEAD42_MAX_ERROR',maxerr,flush=True)
 qout.mkdir();torch.save(q,qout/'validation.pt')
 qm=dict(source_cache=str(source),source_cache_sha256=input_hash,source_manifest_sha256=sha256(source/'cache_manifest.json'),weight_file_sha256=weighthash,encoder_state_sha256=encoderhash,calibration_sha256=calhash,head42_checkpoint_sha256=exports[42]['head_checkpoint_sha256'],head42_reproduction_max_error=maxerr,query_sha256=sha256(qout/'validation.pt'),precision='float32 medium; encoderB1,headB64/B32,CPU4',count=96,rows_exactly_preserved=True,script_sha256=sha256(Path(__file__)),seconds=time.perf_counter()-start,scope='Research-observed96 episodes; no longer research-unused or independent final test; no image decoding or refit')
 (qout/'manifest.json').write_text(json.dumps(qm,indent=2)+'\n')
 if maxerr>1e-5:raise RuntimeError('Head42 reproduction exceeded1e-5. Stop; do not export43/44.')
 report={}
 for seed,path in dest.items():
  mu=codes(seed);assert mu.shape==data['codes'].shape and torch.isfinite(mu).all();value=dict(data,codes=mu);path.mkdir();torch.save(value,path/'validation.pt');check=torch.load(path/'validation.pt',weights_only=True)
  for key in data:
   if key=='codes':continue
   if isinstance(data[key],torch.Tensor):assert torch.equal(data[key],check[key])
   else:assert data[key]==check[key]
  meta=dict(sm,cache_sha256={'validation':sha256(path/'validation.pt')},scope='Research-observed96 IDM-seed replication; previously used research evaluation, not independent final test',conditioning_mean=f'Completed reference head seed{seed} posterior mu from truefuture Q(x,y)-Q(x,x); no refit; not executable prior',head_checkpoint=exports[seed]['head_checkpoint'],head_checkpoint_sha256=exports[seed]['head_checkpoint_sha256'],head_state_sha256=exports[seed]['head_state_sha256'],head_manifest_sha256=exports[seed]['source_head_manifest_sha256'],export=exports[seed],observed96_export=dict(source_cache=str(source),source_cache_sha256=input_hash,query_manifest_sha256=sha256(qout/'manifest.json'),query_sha256=qm['query_sha256'],weights_sha256=weighthash,encoder_state_sha256=encoderhash,calibration_sha256=calhash,head42_reproduction_max_error=maxerr,same96rows=True,all_noncode_fields_exactly_preserved=True,no_refit=True,precision=qm['precision']))
  (path/'cache_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');report[seed]=dict(path=str(path),cache_sha256=meta['cache_sha256']['validation'],head_checkpoint_sha256=exports[seed]['head_checkpoint_sha256'])
 assert sha256(source/'validation.pt')==input_hash and sha256(weightfile)==weighthash and sha256(calpath)==calhash
 (qout/'exports.json').write_text(json.dumps(report,indent=2)+'\n');print('COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
