"""Local mean-code repair factorial. No updates without explicit --train.

H: head only, D: decoder only, HD: both, K: decoder plus fixed train-mean code.
Source files and shared trainers are immutable. All runs start from head42/D42.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.train_cached_decoders import make_decoder, file_hash
from research.idm.train_reference_heads import representation

ARMS = ('H', 'D', 'HD', 'K')
BETA = 1e-3
STEPS = 400
BATCH = 8


def row_identity(rows):
    return [(r['episode'], r['index'], r['dt'], r.get('endpoint_indices')) for r in rows]


def state_hash(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def finite_tensor(value, label):
    if value.device.type != 'cpu' or value.requires_grad or not torch.isfinite(value).all():
        raise ValueError('Nonfixed/nonfinite/non-CPU input: ' + label)


def load_inputs():
    root = ROOT / 'runs/reference-idm42-nested512-prior10000'
    observed = ROOT / 'runs/reference-idm42-observed96-prior10000'
    source_hashes = {}
    def register(path):
        path = Path(path); digest = file_hash(path); source_hashes[str(path)] = digest
        return digest
    def read_json(path):
        register(path); return json.loads(Path(path).read_text())
    manifests = {s: read_json((observed if s == 'observed96' else root) / 'cache_manifest.json')
                 for s in ('train', 'validation', 'observed96')}
    data = {}
    for split, count in [('train', 512), ('validation', 64), ('observed96', 96)]:
        path = (observed / 'validation.pt') if split == 'observed96' else root / f'{split}.pt'
        if register(path) != manifests[split]['cache_sha256'][path.stem]:
            raise ValueError('Source cache hash mismatch')
        for key in ('prior_checkpoint_sha256', 'prior_weights_sha256'):
            if manifests[split][key] != manifests['train'][key]:
                raise ValueError('Current-only baseline provenance differs')
        if manifests[split]['prior_checkpoint_step'] != 10000:
            raise ValueError('Expected final10000 prior')
        data[split] = torch.load(path, weights_only=True, mmap=True)
        if len(data[split]['rows']) != count:
            raise ValueError('Unexpected pair count')
        for key in ('current', 'future', 'baseline', 'codes'):
            finite_tensor(data[split][key], split + '/' + key)
    episode_sets = [{r['episode'] for r in d['rows']} for d in data.values()]
    if any(episode_sets[i] & episode_sets[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Episode overlap')
    selection_path = Path(manifests['train']['selection_path'])
    selection = read_json(selection_path)
    if source_hashes[str(selection_path)] != manifests['train']['selection_sha256']:
        raise ValueError('Selection changed')
    if data['train']['rows'] != selection['retained'] + selection['added']:
        raise ValueError('Nested train row selection differs')

    qpaths = dict(old_train=ROOT/'research/idm/reference-query-full/train.pt',
                  added=root/'additional_queries.pt',
                  validation=ROOT/'research/idm/reference-query-full/validation.pt',
                  observed96=ROOT/'research/idm/observed96-reference-queries2900/validation.pt')
    qmanifest = read_json(ROOT/'research/idm/reference-query-full/manifest.json')
    q96manifest = read_json(qpaths['observed96'].parent/'manifest.json')
    audit = read_json(ROOT/'research/idm/nested512-independent-verification.json')
    historical_probe = read_json(ROOT/'research/idm/query-accessibility512/protocol.json')
    qs = {}
    for name, path in qpaths.items():
        digest = register(path)
        if digest != historical_probe['source_sha256'][str(path)]:
            raise ValueError('Query source differs from verified query probe')
        qs[name] = torch.load(path, weights_only=True)
    if source_hashes[str(qpaths['old_train'])] != manifests['train']['extraction_source_sha256'][str(qpaths['old_train'])]:
        raise ValueError('Old query source hash mismatch')
    if source_hashes[str(qpaths['added'])] != audit['additional_queries_sha256']:
        raise ValueError('Added query hash mismatch')
    if source_hashes[str(qpaths['observed96'])] != q96manifest['query_sha256']:
        raise ValueError('Observed query hash mismatch')
    queries = dict(train={k: torch.cat([qs['old_train'][k], qs['added'][k]]) for k in ('query', 'reference')},
                   validation=qs['validation'], observed96=qs['observed96'])
    queries['train']['rows'] = qs['old_train']['rows'] + qs['added']['rows']
    export = manifests['train']['export']
    head_path = Path(export['head_checkpoint'])
    cal_path = ROOT/'research/idm/reference-query-full/train_calibration.pt'
    if register(head_path) != export['head_checkpoint_sha256'] or register(cal_path) != export['calibration_sha256']:
        raise ValueError('Head/calibration source mismatch')
    if q96manifest['head42_checkpoint_sha256'] != export['head_checkpoint_sha256'] or q96manifest['calibration_sha256'] != export['calibration_sha256']:
        raise ValueError('Observed query head/calibration mismatch')
    if qmanifest['step'] != 2900 or q96manifest['encoder_state_sha256'] != export['encoder_state_sha256']:
        raise ValueError('Query encoder provenance mismatch')
    cal = torch.load(cal_path, weights_only=True)
    x = {}
    for split, d in data.items():
        if row_identity(d['rows']) != row_identity(queries[split]['rows']):
            raise ValueError('Query/cache row identity mismatch: ' + split)
        x[split] = representation(queries[split], cal, 'reference').detach()
        finite_tensor(x[split], 'query/' + split)
        if x[split].shape != (len(d['rows']), 1, 1024):
            raise ValueError('Expected one 1024D query per example')
    head_checkpoint = torch.load(head_path, weights_only=True)
    if (head_checkpoint['seed'], head_checkpoint['steps'], head_checkpoint['variant']) != (42, 2000, 'reference'):
        raise ValueError('Wrong completed head')
    head = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, 64))
    head.load_state_dict({k[5:]: v for k, v in head_checkpoint['state_dict'].items() if k.startswith('head.')}, strict=True)
    decoder_path = ROOT/'runs/reference-idm42-nested512-comparison400-20261113/prior_residual-seed42.pt'
    register(decoder_path)
    saved = torch.load(decoder_path, weights_only=True)
    if (saved['seed'], saved['steps'], saved['variant'], saved['dim'], saved['layers']) != (42, 400, 'prior_residual', 128, 2):
        raise ValueError('Wrong common decoder start')
    comparison_path = decoder_path.parent/'comparison.json'
    comparison = read_json(comparison_path)
    if comparison['cache_manifest']['cache_sha256'] != manifests['train']['cache_sha256']:
        raise ValueError('Decoder trained on different cache')
    decoder = make_decoder('prior_residual', 128, 2, 768, 32, 16)
    decoder.load_state_dict(saved['state_dict'], strict=True)
    return data, x, head, decoder, source_hashes


def mean_head_from_gaussian(head):
    with torch.random.fork_rng(devices=[]):
        mean_head = nn.Sequential(copy.deepcopy(head[0]), copy.deepcopy(head[1]), nn.Linear(head[-1].in_features, 32))
    with torch.no_grad():
        mean_head[-1].weight.copy_(head[-1].weight[:32])
        mean_head[-1].bias.copy_(head[-1].bias[:32])
    return mean_head


class RepairModel(nn.Module):
    def __init__(self, head, decoder, constant_mu, arm):
        super().__init__()
        if arm not in (*ARMS, 'F'):
            raise ValueError('Unknown repair arm')
        self.arm = arm
        self.head = mean_head_from_gaussian(head)
        self.decoder = copy.deepcopy(decoder)
        self.register_buffer('constant_mu', constant_mu.detach().clone())
        self.head.requires_grad_(arm in ('H', 'HD'))
        self.decoder.requires_grad_(arm in ('D', 'HD', 'K'))

    def mean(self, query):
        if self.arm == 'K':
            return self.constant_mu.expand(len(query), -1, -1)
        return self.head(query)

    def forward(self, query, current, baseline):
        # Do not suppress autograd through a frozen D in arm H.
        mu = self.mean(query)
        return self.decoder(current, mu, baseline=baseline), mu


def losses(model, query, current, baseline, target, reference_mu, reference_lv):
    pred, mu = model(query, current.detach(), baseline.detach())
    mse = F.mse_loss(pred, target.detach())
    if model.arm == 'K':
        drift = mse.new_zeros(())
    else:
        drift = .5*((mu-reference_mu.detach()).square()/reference_lv.detach().exp()).sum(-1).mean()
    # This standard-normal KL is diagnostic only. Variance is fixed at q0.
    standard_kl = .5*(mu.square()+reference_lv.detach().exp()-1-reference_lv.detach()).sum(-1).mean()
    return mse + BETA*drift, mse, drift, standard_kl


@torch.no_grad()
def anchors(head, features):
    head.eval(); mean_head = mean_head_from_gaussian(head).eval(); result = {}
    for split, x in features.items():
        values = torch.cat([head(v) for v in x.split(BATCH)])
        full_mu, lv = values.chunk(2, -1)
        mu = torch.cat([mean_head(v) for v in x.split(BATCH)])
        result[split] = dict(mu=mu.clone(), logvar=lv.clamp(-10, 10).clone(), full64_mu=full_mu.clone())
    return result


def gradient_diagnostics(model, mse, drift):
    result = {}
    for name in ('head', 'decoder'):
        parameters = [p for p in getattr(model, name).parameters() if p.requires_grad]
        for key, loss in [('mse', mse), ('weighted_drift', BETA*drift)]:
            if not parameters or not loss.requires_grad:
                result[name+'_'+key+'_gradient_norm'] = 0.; continue
            gs = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            result[name+'_'+key+'_gradient_norm'] = math.sqrt(sum(float(g.detach().double().square().sum()) for g in gs if g is not None))
    return result


@torch.inference_mode()
def evaluate(model, data, features, reference):
    model.eval(); rows = []
    for start in range(0, len(features), BATCH):
        end = min(start+BATCH, len(features)); sl = slice(start, end)
        h, b, y = [data[k][sl] for k in ('current', 'baseline', 'future')]
        pred, mu = model(features[sl], h, b)
        lv = reference['logvar'][sl]
        drift = .5*((mu-reference['mu'][sl]).square()/lv.exp()).sum(-1).flatten(1).mean(1)
        standard = .5*(mu.square()+lv.exp()-1-lv).sum(-1).flatten(1).mean(1)
        error = (pred-y).square().flatten(1).mean(1)
        for i in range(end-start):
            rows.append(dict(data['rows'][start+i], mse=float(error[i]), drift_kl_to_reference=float(drift[i]), diagnostic_standard_normal_kl=float(standard[i])))
    return dict(mse=sum(r['mse'] for r in rows)/len(rows),
                mean_drift_kl=sum(r['drift_kl_to_reference'] for r in rows)/len(rows),
                max_drift_kl=max(r['drift_kl_to_reference'] for r in rows),
                rows=rows, scope='K drift is descriptive, not its training penalty; deterministic posterior-mean reconstruction, not a sampled-posterior information estimate')


def frozen_audit(model, initial):
    final = dict(head=state_hash(model.head), decoder=state_hash(model.decoder),
                 constant_mu=tensor_hash(model.constant_mu))
    if model.arm in ('D', 'K', 'F') and final['head'] != initial['head']:
        raise RuntimeError('Frozen head changed')
    if model.arm in ('H', 'F') and final['decoder'] != initial['decoder']:
        raise RuntimeError('Frozen decoder changed')
    if final['constant_mu'] != initial['constant_mu']:
        raise RuntimeError('Fixed constant code changed')
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, choices=[42,43,44], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--train', action='store_true', help='Explicitly authorize fixed400 optimizer updates; absent means preflight/evaluation only')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('Positive CPU thread count required')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads); torch.set_float32_matmul_precision('highest')
    data, x, head, decoder, sources = load_inputs()
    reference = anchors(head, x)
    constant = reference['train']['mu'].mean(0, keepdim=True)
    model = RepairModel(head, decoder, constant, args.arm)
    initial = dict(head=state_hash(model.head), decoder=state_hash(model.decoder), constant_mu=tensor_hash(constant))
    anchor_hashes = {s: {k: tensor_hash(v) for k,v in d.items()} for s,d in reference.items()}
    source_scripts = [Path(__file__), ROOT/'research/idm/train_reference_heads.py', ROOT/'research/train_cached_decoders.py', ROOT/'research/prior_correction.py', ROOT/'research/wide_head_flow.py', ROOT/'vendor/lawam/latent_action_model/core/utils/lam_decoder.py', ROOT/'vendor/lawam/latent_action_model/core/utils/latent_flow.py']
    protocol = dict(arm=args.arm, minibatch_seed=args.seed, common_pretrained_head_seed=42, common_pretrained_decoder_seed=42,
                    steps=STEPS, batch_size=BATCH, lr=3e-4, beta_drift=BETA, standard_normal_KL='diagnostic only',
                    precision='highest/B8 CPU; original cached medium mu is not substituted',
                    mu_only=True, fixed_variance='Reference head logvar clamped[-10,10] and fixed per example',
                    optimizer='AdamW matrix weight_decay .01, bias/norm0; clip global active norm1; mean-only Linear32 output, reference variance fixed outside model',
                    sources=sources, source_scripts={str(p):file_hash(p) for p in source_scripts}, initial=initial,
                    anchor_hashes=anchor_hashes, scope='Observed512/64/96, local mean reconstruction repair; no production aux reproduction, no camera/action identity or execution-time prior claim',
                    drift_gate='Mean TRAIN drift >1 nat marks scope failure; no checkpoint selection or automatic hyperparameter changes',
                    execute_training=args.train)
    args.output.mkdir(parents=True)
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    # Both F and K initial risks are always reported before any optimizer update.
    preflight = dict(source_count=len(sources), rows={s:len(d['rows']) for s,d in data.items()}, initial={}, cached_mu_replay={})
    for split in data:
        preflight['cached_mu_replay'][split] = dict(max_abs=float((reference[split]['mu']-data[split]['codes']).abs().max()), mse=float((reference[split]['mu']-data[split]['codes']).square().mean()), mean32_vs_full64_max_abs=float((reference[split]['mu']-reference[split]['full64_mu']).abs().max()))
    for arm in ('F','K'):
        initial_model = RepairModel(head, decoder, constant, arm)
        preflight['initial'][arm] = {s:evaluate(initial_model,data[s],x[s],reference[s]) for s in data}
        print('initial',arm,{s:v['mse'] for s,v in preflight['initial'][arm].items()},flush=True)
    (args.output/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
    if not args.train:
        frozen_audit(model, initial)
        print('Preflight complete; no optimizer created or updates performed.',flush=True)
        return
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([dict(params=[p for p in parameters if p.ndim>=2],weight_decay=.01),
                                   dict(params=[p for p in parameters if p.ndim<2],weight_decay=0.)],lr=3e-4)
    gen = torch.Generator().manual_seed(args.seed+1000)
    batch_hash = hashlib.sha256(); history=[]; started=time.perf_counter()
    for step in range(1, STEPS+1):
        model.train(); idx=torch.randint(len(x['train']),(BATCH,),generator=gen); batch_hash.update(idx.numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        loss,mse,drift,standard=losses(model,x['train'][idx],data['train']['current'][idx],data['train']['baseline'][idx],data['train']['future'][idx],reference['train']['mu'][idx],reference['train']['logvar'][idx])
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        record=dict(step=step,mse=float(mse.detach()),drift=float(drift.detach()),diagnostic_standard_kl=float(standard.detach()))
        if step in (1,10,100,400): record['gradients']=gradient_diagnostics(model,mse,drift)
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        if not torch.isfinite(norm): raise RuntimeError('Nonfinite clipped gradient norm')
        optimizer.step(); record['gradient_norm_before_clipping']=float(norm)
        if step in (1,10,100,200,300,400):
            frozen_audit(model,initial);history.append(record);print(args.arm,args.seed,record,flush=True)
    evaluation={s:evaluate(model,data[s],x[s],reference[s]) for s in data}
    final=frozen_audit(model,initial)
    for s,d in reference.items():
        for k,v in d.items():
            if tensor_hash(v)!=anchor_hashes[s][k]: raise RuntimeError('Reference posterior mutated')
    for path,digest in sources.items():
        if file_hash(Path(path))!=digest: raise RuntimeError('Source changed during run: '+path)
    checkpoint=args.output/'repair.pt'
    torch.save(dict(state_dict=model.state_dict(),arm=args.arm,minibatch_seed=args.seed,steps=STEPS,
                    initial=initial,final=final,source_hashes=sources,reference_anchor_hashes=anchor_hashes),checkpoint)
    scope_failed=args.arm in ('H','HD') and evaluation['train']['mean_drift_kl']>1.
    report=dict(protocol_sha256=file_hash(args.output/'protocol.json'),preflight_sha256=file_hash(args.output/'preflight.json'),
                checkpoint_sha256=file_hash(checkpoint),batch_sha256=batch_hash.hexdigest(),history=history,
                evaluation=evaluation,initial=initial,final=final,frozen_audit_passed=True,
                trainable_parameters=sum(p.numel() for p in parameters),drift_scope_failed=scope_failed,
                seconds=time.perf_counter()-started)
    (args.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
