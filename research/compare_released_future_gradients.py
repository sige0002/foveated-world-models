"""Compare frozen local/released gradients in the SAME future-feature coordinates.

The future target is detached and separately cloned. Only the encoder's future
input requires gradients. Thus dL/du_future excludes the trivial target term.
No optimizer is created; every parameter is frozen and hashed before/after.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.train_robotwin_stage1 import write_json


def available_memory_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    return None


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.samples <= 8:
        raise ValueError('This diagnostic is bounded to one through eight B1 pairs')
    if args.output.exists():
        raise FileExistsError(args.output)
    req = yaml.safe_load((args.run/'requested_config.yaml').read_text())
    cfg = yaml.safe_load((args.run/'resolved_config.yaml').read_text())
    source = Path(req['official_source']).parent/'LaWAM'
    os.environ.update(cfg['environment'])
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    sys.path[:0] = [str(source), *req['dependency_roots']]
    commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip()
    assert commit == req['official_commit']
    assert not subprocess.check_output(['git','-C',str(source),'diff','--binary'])
    import torch
    from torch.nn import functional as F
    from latent_action_model.core.utils.lam_encoder import LAMEncoder
    from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
    from latent_action_model.core.vq import VAEQuantizer
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(7142)
    assert torch.cuda.is_available()
    before_available = available_memory_bytes()
    if before_available is not None and before_available < 12*1024**3:
        raise RuntimeError('Less than 12GiB available; defer rather than pressure the training jobs')
    diag = args.run/'evaluations/z-mechanism-20261114'
    cache_path = diag/'features.pt'
    cache = torch.load(cache_path, weights_only=True, map_location='cpu')
    x, y = cache['x'][:args.samples].float(), cache['y'][:args.samples].float()
    assert x.shape == y.shape and x.shape[1] == 1
    # Match the first draw of the companion latent-gradient diagnostic.
    gaussian = torch.randn(4,32,1,32,generator=torch.Generator().manual_seed(81073))[0,:args.samples]
    probes = (torch.randint(0,2,(4,args.samples,1,32),generator=torch.Generator().manual_seed(91372))*2-1).float()
    release = args.run.parents[1]/'models/lawam-lam-bd993da'
    source_hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        source/'latent_action_model/core/utils/lam_encoder.py',
        source/'latent_action_model/core/utils/lam_decoder.py',
        source/'latent_action_model/core/utils/modules.py',
        source/'latent_action_model/core/vq.py']}

    def fingerprint(modules):
        hasher = hashlib.sha256()
        count = 0
        for name,module in modules.items():
            for key,value in module.state_dict().items():
                hasher.update((name+'.'+key).encode()+b'\0')
                hasher.update(str(tuple(value.shape)).encode()+b'\0')
                hasher.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
                count += 1
        return hasher.hexdigest(), count

    def loss_rows(prediction, target):
        return {'smooth_l1': F.smooth_l1_loss(prediction,target,beta=.1,reduction='none').flatten(1).mean(1),
                'mse': (prediction-target).square().flatten(1).mean(1)}

    reports = {}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for kind in ['official','local']:
        checkpoint = release/'checkpoints/pytorch_model.pt' if kind=='official' else args.run/'checkpoints/last.ckpt'
        mc = yaml.safe_load((release/'dino_large_vae.yaml').read_text())['model'] if kind=='official' else cfg['model']
        common = dict(context_dim=mc['dim'], input_dim=x.shape[-1], num_heads=mc['num_heads'], dropout=mc['dropout'],
                      code_dim=mc['code_dim'], num_queries=mc['num_queries'], ffn_expansion_factor=mc['ffn_expansion_factor'],
                      grid_hw=tuple(int(v/mc['patch_size']) for v in mc['image_hw']))
        modules = {
            'encoder': LAMEncoder(**common,num_layers=mc['enc_layers'],num_frames=mc['num_frames'],
                modal_mask=mc['enc_modal_mask'],add_state=mc['enc_add_state'],max_state_dim=mc['max_state_dim'],num_embodiments=mc['num_embodiments']),
            'vq': VAEQuantizer(code_dim=mc['code_dim'],**mc['vq_kwargs']),
            'decoder': LAMDecoder_v2(**common,num_layers=mc['dec_layers']),
        }
        assert not mc['enc_add_state']
        saved = torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)
        if kind=='local':
            assert saved['global_step']==10000
        for name,module in modules.items():
            prefix='lam.'+name+'.'
            state={key[len(prefix):]:value for key,value in saved['state_dict'].items() if key.startswith(prefix)}
            module.load_state_dict(state,strict=True)
            module.eval().requires_grad_(False)
        del state,saved
        before_hash,tensor_count=fingerprint(modules)
        assert tensor_count==439
        for module in modules.values():
            module.cuda()
        encoder,vq,decoder=(modules[name] for name in ['encoder','vq','decoder'])
        high_path=diag/f'{kind}-highest-precision.pt'
        high=torch.load(high_path,weights_only=True,map_location='cpu')
        means=high['real']['mu64'].double().flatten(1)
        centered=means-means.mean(0)
        covariance=centered.T@centered/len(means)
        rows=[]
        for index in range(args.samples):
            current=x[index:index+1].cuda().detach()
            future=y[index:index+1].cuda().clone().detach().requires_grad_(True)
            target=y[index:index+1].cuda().clone().detach()
            assert not target.requires_grad and target.data_ptr()!=future.data_ptr()
            nodes=encoder(torch.cat([current,future],dim=1))
            mu,lv=vq._encode(nodes)
            sigma=(.5*lv).exp()
            torch.testing.assert_close(mu.detach().cpu(),high['real']['mu64'][index:index+1].float(),rtol=2e-4,atol=2e-5)
            with torch.no_grad():
                current_nodes=encoder(torch.cat([current,current],dim=1))
                current_mu,current_lv=vq._encode(current_nodes)
                current_sigma=(.5*current_lv).exp()
            delta_mu=(current_mu-mu).detach()
            input_displacement=(current-future).detach()
            jacobian_probes=[]
            for probe in probes[:,index:index+1]:
                jacobian_vjp=torch.autograd.grad((mu*probe.cuda()).sum(),future,retain_graph=True)[0]
                jacobian_probes.append(float(jacobian_vjp.double().square().sum()))
            jacobian_frobenius_estimate=math.sqrt(sum(jacobian_probes)/len(jacobian_probes))
            del jacobian_vjp
            for path in ['mu_only','sampled']:
                noise=gaussian[index:index+1].cuda()
                z=mu if path=='mu_only' else mu+sigma*noise
                alternative_z=current_mu if path=='mu_only' else current_mu+current_sigma*noise
                prediction=decoder(current,z)
                losses=loss_rows(prediction,target)
                with torch.no_grad():
                    alternative_prediction=decoder(current,alternative_z)
                    alternative_losses=loss_rows(alternative_prediction,target)
                    output_difference_mse=float((alternative_prediction-prediction.detach()).double().square().mean())
                for loss_name,loss in losses.items():
                    future_gradient,z_gradient,node_gradient=torch.autograd.grad(loss.sum(),[future,z,nodes],retain_graph=True)
                    gz=z_gradient.detach().double().cpu().flatten()
                    gf=future_gradient.detach().double()
                    quadratic=float(gz@(covariance@gz))
                    row={
                        'row_index':index,**cache['rows'][index],'path':path,'loss':loss_name,
                        'image_loss':float(loss.detach()),
                        'future_input_gradient_norm':float(gf.norm()),
                        'future_input_gradient_rms':float(gf.square().mean().sqrt()),
                        'future_input_directional_derivative_to_current':float((future_gradient*input_displacement).sum()),
                        'z_gradient_norm':float(z_gradient.detach().double().norm()),
                        'node_gradient_norm':float(node_gradient.detach().double().norm()),
                        'mu_covariance_scaled_z_gradient':math.sqrt(max(0.,quadratic)),
                        'posterior_noise_scaled_z_gradient':float((z_gradient.detach().double()*sigma.detach().double()).norm()),
                        'z_gradient_dot_mu_current_minus_real':float((z_gradient*delta_mu).sum()),
                        'z_gradient_dot_actual_control_displacement':float((z_gradient*(alternative_z-z).detach()).sum()),
                        'mu_current_minus_real_norm':float(delta_mu.double().norm()),
                        'actual_control_z_displacement_norm':float((alternative_z-z).detach().double().norm()),
                        'current_only_control_image_loss':float(alternative_losses[loss_name]),
                        'current_only_control_loss_minus_real':float(alternative_losses[loss_name]-loss.detach()),
                        'current_only_control_output_difference_mse':output_difference_mse,
                        'mu_future_jacobian_frobenius_hutchinson_estimate':jacobian_frobenius_estimate,
                        'mu_future_jacobian_squared_frobenius_probe_estimates':jacobian_probes,
                    }
                    assert all(math.isfinite(value) for value in row.values() if isinstance(value,float))
                    rows.append(row)
                del prediction,losses,alternative_prediction,alternative_losses,future_gradient,z_gradient,node_gradient
            del current,future,target,nodes,mu,lv,sigma,current_mu,current_lv,current_sigma,current_nodes,z,alternative_z
            print(json.dumps({'phase':'whole_path_pair','kind':kind,'pairs_completed':index+1,
                'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'available_system_memory_bytes':available_memory_bytes()}),flush=True)
        assert all(parameter.grad is None for module in modules.values() for parameter in module.parameters())
        after_hash,_=fingerprint(modules)
        assert before_hash==after_hash
        summaries=[]
        scalar_fields=[key for key,value in rows[0].items() if isinstance(value,float)]
        for path in ['mu_only','sampled']:
            for loss in ['smooth_l1','mse']:
                subset=[row for row in rows if row['path']==path and row['loss']==loss]
                summaries.append({'path':path,'loss':loss,'pairs':len(subset),
                    **{key:sum(row[key] for row in subset)/len(subset) for key in scalar_fields}})
        reports[kind]={'checkpoint':str(checkpoint),'strictly_loaded_state_tensors':tensor_count,
            'state_sha256_before':before_hash,'state_sha256_after':after_hash,'unchanged':True,
            'posterior_covariance_source':str(high_path),'covariance_samples':len(means),
            'summary':summaries,'rows':rows}
        write_json(args.output.with_name(args.output.stem+'-partial.json'),{'complete':False,'reports':reports})
        print(json.dumps({'phase':'whole_path_model_complete','kind':kind,'summary':summaries}),flush=True)
        del encoder,vq,decoder,modules,module
        torch.cuda.empty_cache()
    ratios=[]
    for path in ['mu_only','sampled']:
        for loss in ['smooth_l1','mse']:
            get=lambda kind:next(row for row in reports[kind]['summary'] if row['path']==path and row['loss']==loss)
            official,local=get('official'),get('local')
            metrics=['future_input_gradient_norm','z_gradient_norm','node_gradient_norm','mu_covariance_scaled_z_gradient',
                     'posterior_noise_scaled_z_gradient','mu_future_jacobian_frobenius_hutchinson_estimate']
            ratios.append({'path':path,'loss':loss,'official_over_local':{
                key:official[key]/local[key] if local[key] else None for key in metrics}})
    report={'complete':True,'reports':reports,'official_over_local_ratios':ratios,'source':str(source),
        'source_commit':commit,'source_hashes':source_hashes,'samples':args.samples,'batch_size':1,
        'precision':'highest FP32 encoder, VAE and decoder; no autocast; shared originally cached DINO inputs',
        'feature_cache':str(cache_path),'epsilon_sha256':hashlib.sha256(gaussian.numpy().tobytes()).hexdigest(),
        'hutchinson_probes_sha256':hashlib.sha256(probes.numpy().tobytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'elapsed_seconds':time.monotonic()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'available_system_memory_before_bytes':before_available,'available_system_memory_after_bytes':available_memory_bytes(),
        'scope':'Frozen relevant encoder/VAE/decoder components, actual release and local10k weights. No optimizer or parameter updates. Same DINO future input coordinate system; targets separately detached and cloned. B1 per-example feature-mean loss gradients, not parameter-update norms. One shared Gaussian sample per pair and four shared Hutchinson probes; estimates are noisy. C_mu uses128validation codes and is diagnostic, not a population covariance. Current-only replacement can be off distribution and is not a physical zero-action label. g omitted due incompatible official14/local16 state contracts.'}
    write_json(args.output,report)
    print(json.dumps({'complete':True,'output':str(args.output),'ratios':ratios}),flush=True)


if __name__=='__main__':
    main()
