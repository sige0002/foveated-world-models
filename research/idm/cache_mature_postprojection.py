"""Immutable6501 encoder extraction on exact existing research feature pairs."""
import json,sys,time
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
from latent_action_model.core.utils.lam_encoder import LAMEncoder
from research.idm.cache_reference_queries import sha256
from research.idm.postprojection_representations import fit_calibration,representation,VARIANTS
@torch.inference_mode()
def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
 source=ROOT/'runs/reference-idm-with-prior6501';weights_cache=ROOT/'runs/frozen-idm-step6501';out=ROOT/'research/idm/mature-postprojection-cache'
 if out.exists():raise FileExistsError(out)
 manifest=json.loads((source/'cache_manifest.json').read_text());source_hashes={k:sha256(source/f'{k}.pt') for k in ('train','validation')}
 assert source_hashes==manifest['cache_sha256']
 weights_hash=sha256(weights_cache/'reference_weights.pt');cfg=yaml.safe_load((weights_cache/'resolved_config.yaml').read_text())['model']
 enc=LAMEncoder(context_dim=cfg['dim'],input_dim=768,num_heads=cfg['num_heads'],num_queries=cfg['num_queries'],grid_hw=tuple(v//cfg['patch_size'] for v in cfg['image_hw']),ffn_expansion_factor=cfg['ffn_expansion_factor'],dropout=cfg['dropout'],code_dim=cfg['code_dim'],num_layers=cfg['enc_layers'],num_frames=cfg['num_frames'],modal_mask=cfg['enc_modal_mask'],add_state=False,max_state_dim=cfg['max_state_dim'],num_embodiments=cfg['num_embodiments']).eval().requires_grad_(False)
 saved=torch.load(weights_cache/'reference_weights.pt',weights_only=True,mmap=True);assert saved['global_step']==6501
 prefix='lam.encoder.';enc.load_state_dict({k[len(prefix):]:v for k,v in saved['state_dict'].items() if k.startswith(prefix)},strict=True);del saved
 captured={};original=enc.out_proj.forward
 def forward(q):captured['query']=q.clone();return original(q)
 enc.out_proj.forward=forward;extracted={};start=time.perf_counter()
 for split in ('train','validation'):
  data=torch.load(source/f'{split}.pt',weights_only=True);reference=torch.load(weights_cache/f'{split}.pt',weights_only=True)
  for key in ('current','future'):assert torch.equal(data[key],reference[key])
  identity=lambda r:(r['episode'],r['index'],r['dt'])
  assert list(map(identity,data['rows']))==list(map(identity,reference['rows']))
  queries=[];projections=[]
  for i in range(len(data['current'])):
   p=enc(torch.cat((data['current'][i:i+1],data['future'][i:i+1]),1));queries.append(captured['query']);projections.append(p)
   if (i+1)%16==0:print(split,i+1,flush=True)
  extracted[split]=dict(query=torch.cat(queries),projection=torch.cat(projections),rows=data['rows'])
  assert torch.isfinite(extracted[split]['query']).all() and torch.isfinite(extracted[split]['projection']).all()
 assert not {r['episode'] for r in extracted['train']['rows']} & {r['episode'] for r in extracted['validation']['rows']}
 cal=fit_calibration(extracted['train']['projection'])
 for split,values in extracted.items():
  values['representations']={v:representation(values['projection'],cal,v) for v in VARIANTS}
  for x in values['representations'].values():assert x.shape[-1]==34 and torch.isfinite(x).all()
 assert source_hashes=={k:sha256(source/f'{k}.pt') for k in ('train','validation')}
 assert weights_hash==sha256(weights_cache/'reference_weights.pt')
 out.mkdir()
 for split,values in extracted.items():torch.save(values,out/f'{split}.pt')
 torch.save(cal,out/'train_calibration.pt')
 report=dict(checkpoint=6501,source=str(source),source_cache_sha256=source_hashes,source_manifest_sha256=sha256(source/'cache_manifest.json'),weights_sha256=weights_hash,config_sha256=sha256(weights_cache/'resolved_config.yaml'),script_sha256=sha256(Path(__file__)),representation_script_sha256=sha256(ROOT/'research/idm/postprojection_representations.py'),output_sha256={p.name:sha256(p) for p in out.glob('*.pt')},threads=2,precision='float32 highest',batch_size=1,torch_version=str(torch.__version__),counts={k:len(v['rows']) for k,v in extracted.items()},same_source_and6501_features_and_episode_index_dt=True,episode_overlap=0,calibration_split='train only',variants=list(VARIANTS),output_dim=34,raw_calibration='Per-coordinate train mean/std, population std with floor1e-3',statistics_calibration='Train mean/std of per-example projected coordinate mean and log sqrt(var+1e-5); floor1e-3',seconds=time.perf_counter()-start,limitations='Frozen encoder features only; all34D variants require a fresh identical-capacity head. No learned LayerNorm affine. Finite epsilon retains weak scale dependence in direction. No fitting of predictive heads, decoder evaluation or semantic claims.')
 (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
