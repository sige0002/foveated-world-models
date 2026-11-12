"""Warm CPU latency: correction alone and current-only prior plus correction."""
import argparse,json,sys,time,statistics
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.slim_ffn_decoder import make_slim_decoder
from research.train_cached_decoders import file_hash
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    models={};hashes={}
    for expansion in [4,2]:
        run=a.source/f'runs/ffn{expansion}-seed42-512-400-20261113'
        report=json.loads((run/'results.json').read_text());path=run/'decoder.pt'
        assert file_hash(path)==report['checkpoint_sha256']
        saved=torch.load(path,weights_only=True)
        assert saved['steps']==400 and saved['seed']==42 and saved['expansion']==expansion
        m=make_slim_decoder(expansion,42);m.load_state_dict(saved['state_dict'],strict=True)
        models[expansion]=m.eval().requires_grad_(False);hashes[str(path)]=file_hash(path)
    prior_root=a.source/'runs/frozen-model-step10000'
    config=yaml.safe_load((prior_root/'resolved_config.yaml').read_text())['model']
    weights=torch.load(prior_root/'reference_weights.pt',weights_only=True,mmap=True)
    assert weights['global_step']==10000
    prior=LAMDecoder_v2(context_dim=config['dim'],input_dim=768,code_dim=config['code_dim'],num_queries=config['num_queries'],
                        num_layers=config['dec_layers'],num_heads=config['num_heads'],dropout=0.,grid_hw=(16,16),
                        ffn_expansion_factor=config['ffn_expansion_factor'],last_ln=config.get('decoder_last_ln',True)).eval().requires_grad_(False)
    prefix='lam.decoder.'
    prior.load_state_dict({k[len(prefix):]:v for k,v in weights['state_dict'].items() if k.startswith(prefix)},strict=True)
    del weights
    cache_root=a.source/'runs/reference-idm42-nested512-prior10000'
    manifest=json.loads((cache_root/'cache_manifest.json').read_text())
    assert file_hash(prior_root/'reference_weights.pt')==manifest['prior_weights_sha256']
    cache=torch.load(cache_root/'train.pt',weights_only=True,mmap=True)
    assert file_hash(cache_root/'train.pt')==manifest['cache_sha256']['train']
    timings=[];baseline_differences={}
    for batch in [1,8]:
        h,z,b=[cache[k][:batch] for k in ('current','codes','baseline')]
        zero=torch.zeros_like(z)
        def call(scope,expansion):
            base=prior(h,zero) if scope=='prior_plus_correction' else b
            return models[expansion](h,z,baseline=base)
        baseline_differences[batch]=float((prior(h,zero)-b).abs().max())
        for scope in ['correction_only','prior_plus_correction']:
            for expansion in [4,2]:
                for _ in range(3):assert torch.isfinite(call(scope,expansion)).all()
            for trial in range(10):
                for expansion in ([4,2] if trial%2==0 else [2,4]):
                    start=time.perf_counter();cpu=time.process_time()
                    call(scope,expansion)
                    timings.append(dict(batch=batch,scope=scope,trial=trial,expansion=expansion,
                                        seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu))
            print(batch,scope,'done',flush=True)
    summary={}
    for batch in [1,8]:
        for scope in ['correction_only','prior_plus_correction']:
            values={e:[r['seconds'] for r in timings if r['batch']==batch and r['scope']==scope and r['expansion']==e] for e in [4,2]}
            summary[f'B{batch}/{scope}']={str(e):dict(median=statistics.median(v),min=min(v),max=max(v)) for e,v in values.items()}
            summary[f'B{batch}/{scope}']['candidate_over_baseline']=statistics.median(values[2])/statistics.median(values[4])
    report=dict(threads=2,precision='highest CPU float32',torch_version=torch.__version__,checkpoint_sha256=hashes,
                prior_weights_sha256=manifest['prior_weights_sha256'],timings=timings,summary=summary,
                prior_vs_cached_max_abs=baseline_differences,
                scope='Warm measurements, trained seed42 endpoints fixed before results. Same input and CPU; no GPU. Prior_plus_correction includes current-only decoder but excludes DINO encoder and IDM. Recomputed prior precision differences are recorded; these timing outputs are not substituted for fixed-cache MSE evaluation.')
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
