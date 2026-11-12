"""Align unused-episode baseline and original decoder scoring to highest/B2.

Does not change current, future, adapted codes, original posterior codes, or
selection. Preserves B1/medium metrics as reproduction diagnostics.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor/lawam'))
import torch,yaml
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
from research.cache_current_prior import predict_prior
from research.train_cached_decoders import evaluate,file_hash

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--prior-cache',type=Path,default=ROOT/'runs/frozen-idm-step6501');a=p.parse_args()
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
 path=a.cache/'validation.pt';meta_path=a.cache/'cache_manifest.json';meta=json.loads(meta_path.read_text())
 if 'decoder_precision_alignment' in meta:raise ValueError('Already aligned')
 original_hash=file_hash(path)
 if original_hash!=meta['cache_sha256']['validation']:raise ValueError('Input cache mismatch')
 data=torch.load(path,weights_only=True);cfg=yaml.safe_load((a.prior_cache/'resolved_config.yaml').read_text())['model']
 model=LAMDecoder_v2(context_dim=cfg['dim'],input_dim=data['current'].shape[-1],code_dim=cfg['code_dim'],num_queries=cfg['num_queries'],
    num_layers=cfg['dec_layers'],num_heads=cfg['num_heads'],dropout=cfg['dropout'],grid_hw=tuple(v//cfg['patch_size'] for v in cfg['image_hw']),
    ffn_expansion_factor=cfg['ffn_expansion_factor'],last_ln=cfg.get('decoder_last_ln',True)).eval().requires_grad_(False)
 weights=torch.load(a.prior_cache/'reference_weights.pt',weights_only=True,mmap=True)
 assert weights['global_step']==6501
 prefix='lam.decoder.';model.load_state_dict({k[len(prefix):]:v for k,v in weights['state_dict'].items() if k.startswith(prefix)},strict=True);del weights
 baseline=predict_prior(model,data['current'],cfg['code_dim'],batch_size=2)
 original=evaluate(model,dict(data,codes=data['original_codes']),batch_size=2)
 new_errors=(baseline-data['future']).square().flatten(1).mean(1)
 old_errors=(data['baseline']-data['future']).square().flatten(1).mean(1)
 old_original=torch.tensor([r['original6501_mse'] for r in data['rows']],dtype=torch.float64)
 new_original=torch.tensor([r['mse'] for r in original['rows']],dtype=torch.float64)
 diagnostics=dict(baseline_per_example_mse_max_difference=float((new_errors-old_errors).abs().max()),
   baseline_prediction_max_abs_difference=float((baseline-data['baseline']).abs().max()),
   original_per_example_mse_max_difference=float((new_original-old_original).abs().max()),
   baseline_medium_b1_mse=float(old_errors.mean()),baseline_highest_b2_mse=float(new_errors.mean()),
   original_medium_b1_mse=float(old_original.mean()),original_highest_b2_mse=float(new_original.mean()))
 if diagnostics['baseline_per_example_mse_max_difference']>1e-4 or diagnostics['original_per_example_mse_max_difference']>1e-4:raise RuntimeError(f'Reproduction threshold failed: {diagnostics}')
 rows=[]
 for i,r in enumerate(data['rows']):rows.append(dict(r,original6501_mse_medium_b1=r['original6501_mse'],baseline_mse_medium_b1=r['baseline_mse'],
     original6501_mse=float(new_original[i]),baseline_mse=float(new_errors[i])))
 result=dict(data,baseline=baseline,rows=rows)
 temp=a.cache/'validation.aligned.tmp';torch.save(result,temp);loaded=torch.load(temp,weights_only=True)
 for key in ('current','future','codes','original_codes'):assert torch.equal(loaded[key],data[key])
 assert [(r['episode'],r['index'],r['dt'],r['endpoint_indices']) for r in rows]==[(r['episode'],r['index'],r['dt'],r['endpoint_indices']) for r in data['rows']]
 temp.replace(path)
 meta['cache_sha256']['validation']=file_hash(path)
 meta['precision']='Features/queries/original posterior: float32 medium B1; adapted head: medium B64/B32; baseline and original decoder reevaluation: highest B2; CPU4'
 meta['baseline_source']='Frozen step6501 decoder(current,zero code), predict_prior batch2, float32 highest'
 meta['original6501_source']='Actual step6501 medium/B1 encoder+VAE posterior mean; original decoder reevaluation highest/B2; medium/B1 metric also retained'
 meta['decoder_precision_alignment']=dict(diagnostics=diagnostics,pre_alignment_cache_sha256=original_hash,
    script_sha256=file_hash(Path(__file__)),predict_prior_source_sha256=file_hash(ROOT/'research/cache_current_prior.py'),
    evaluation_source_sha256=file_hash(ROOT/'research/train_cached_decoders.py'),prior_weights_sha256=file_hash(a.prior_cache/'reference_weights.pt'),
    preserved_tensors=['current','future','codes','original_codes'],selection_metadata_preserved=True)
 temp_meta=a.cache/'cache_manifest.aligned.tmp';temp_meta.write_text(json.dumps(meta,indent=2)+'\n');temp_meta.replace(meta_path)
 print(json.dumps(diagnostics,indent=2))
if __name__=='__main__':main()
