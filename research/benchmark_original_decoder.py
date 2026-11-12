"""Direct matched CPU timing of original decoder, zero-code prior, and correction pipeline."""
import argparse,json,sys,time,statistics,resource,platform
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
    for expansion in [4]:
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
    native_path=prior_root/'train.pt'
    native_manifest=json.loads((prior_root/'cache_manifest.json').read_text())
    native_hash=file_hash(native_path)
    assert native_hash==native_manifest['cache_sha256']['train']
    native=torch.load(native_path,weights_only=True,mmap=True)
    assert torch.equal(native['current'][:8],cache['current'][:8])
    cases=['original_native_code','original_zero_code','prior_plus_correction']
    timings=[];baseline_differences={}
    for batch in [1,8]:
        h,z,b=[cache[k][:batch] for k in ('current','codes','baseline')]
        native_z=native['codes'][:batch];zero=torch.zeros_like(z)
        def call(case):
            if case=='original_native_code':return prior(h,native_z)
            base=prior(h,zero)
            if case=='original_zero_code':return base
            return models[4](h,z,baseline=base)
        baseline_differences[batch]=float((prior(h,zero)-b).abs().max())
        for case in cases:
            for _ in range(3):assert torch.isfinite(call(case)).all()
        for trial in range(12):
            order=cases[trial%3:]+cases[:trial%3]
            if trial%2:order=order[::-1]
            for case in order:
                start=time.perf_counter();cpu=time.process_time()
                result=call(case)
                timings.append(dict(batch=batch,case=case,trial=trial,
                    seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu))
            if trial%3==2:print(batch,'trial',trial+1,'done',flush=True)
    summary={}
    for batch in [1,8]:
        summary[str(batch)]={}
        for case in cases:
            v=[r['seconds'] for r in timings if r['batch']==batch and r['case']==case]
            summary[str(batch)][case]=dict(median=statistics.median(v),min=min(v),max=max(v),repetitions=len(v))
        med=summary[str(batch)]
        med['corrected_over_original']=med['prior_plus_correction']['median']/med['original_native_code']['median']
        med['corrected_over_zero_prior']=med['prior_plus_correction']['median']/med['original_zero_code']['median']
    report=dict(threads=2,precision='highest CPU float32',torch_version=torch.__version__,
        platform=platform.platform(),processor=platform.processor(),checkpoint_sha256=hashes,
        prior_weights_sha256=manifest['prior_weights_sha256'],native_cache_sha256=native_hash,
        timings=timings,summary=summary,prior_vs_cached_max_abs=baseline_differences,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,script_sha256=file_hash(Path(__file__)),
        scope='Warm CPU inference, same current features and frozen final10000 decoder; original uses native cached posterior mean, correction uses reference42 mean. Excludes DINO, IDM, data loading, cold startup. No GPU. Native and reference code computation excluded symmetrically. 12 interleaved repetitions per case/batch.')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
