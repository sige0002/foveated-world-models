"""Read-only loading and Gaussian metrics for completed local-repair checkpoints."""
import json
from pathlib import Path
import torch
from research.idm.train_local_repair import (
    ROOT, RepairModel, anchors, load_inputs, state_hash, tensor_hash,
)
from research.train_cached_decoders import file_hash

SEEDS = (42,43,44)
ARMS = ('H','D','HD')
INDICES = [0,9,18,27,36,45,54,63]


def validate_run_map(mapping):
    expected={f'{arm}{seed}' for arm in ARMS for seed in SEEDS}
    if set(mapping)!=expected:
        raise ValueError('Require all and only completed H/D/HD x seeds42/43/44; no result-dependent subset')
    return {k:Path(v) for k,v in mapping.items()}


def check_completed(directory, arm, seed, sources):
    needed=[directory/name for name in ('repair.pt','results.json','protocol.json','preflight.json')]
    if not all(p.is_file() for p in needed):
        raise FileNotFoundError('Completed400 endpoint required: '+str(directory))
    ckpath,resultpath,protocolpath,preflightpath=needed
    result=json.loads(resultpath.read_text());protocol=json.loads(protocolpath.read_text())
    if result['checkpoint_sha256']!=file_hash(ckpath) or result['protocol_sha256']!=file_hash(protocolpath) or result['preflight_sha256']!=file_hash(preflightpath):
        raise ValueError('Completed artifact hashes differ')
    if (protocol['arm'],protocol['minibatch_seed'],protocol['steps'],protocol['execute_training'])!=(arm,seed,400,True):
        raise ValueError('Unexpected training endpoint')
    if protocol['sources']!=sources or not result['frozen_audit_passed']:
        raise ValueError('Source/frozen-state audit mismatch')
    checkpoint=torch.load(ckpath,weights_only=True,mmap=True)
    if (checkpoint['arm'],checkpoint['minibatch_seed'],checkpoint['steps'])!=(arm,seed,400):
        raise ValueError('Checkpoint identity mismatch')
    if checkpoint['source_hashes']!=sources or checkpoint['final']!=result['final']:
        raise ValueError('Checkpoint source/final metadata mismatch')
    return checkpoint,result,dict(checkpoint_sha256=file_hash(ckpath),results_sha256=file_hash(resultpath),protocol_sha256=file_hash(protocolpath),drift_scope_failed=result['drift_scope_failed'])


def load_completed_model(directory,arm,seed,head,decoder,constant,sources,reference):
    ck,result,metadata=check_completed(directory,arm,seed,sources)
    model=RepairModel(head,decoder,constant,arm)
    initial=dict(head=state_hash(model.head),decoder=state_hash(model.decoder),constant_mu=tensor_hash(model.constant_mu))
    if ck['initial']!=initial:raise ValueError('Common start differs')
    expected_anchors={s:{k:tensor_hash(v) for k,v in d.items()} for s,d in reference.items()}
    if ck['reference_anchor_hashes']!=expected_anchors:raise ValueError('Fixed reference variance/mean differs')
    model.load_state_dict(ck['state_dict'],strict=True);model.eval().requires_grad_(False)
    actual=dict(head=state_hash(model.head),decoder=state_hash(model.decoder),constant_mu=tensor_hash(model.constant_mu))
    if actual!=result['final']:raise ValueError('Loaded final state hash mismatch')
    return model,metadata


@torch.no_grad()
def mean_codes(model,features,batch=8):
    return {s:torch.cat([model.mean(v) for v in x.split(batch)]).detach() for s,x in features.items()}


def gaussian_pair_metrics(mu0,mu1,lv0,lv1):
    """Per-example true/swap Gaussian distances; variances are from frozen q0."""
    a,b,c,d=[v.double().flatten(1) for v in (mu0,mu1,lv0,lv1)]
    delta=(a-b).square();v0=c.exp();v1=d.exp()
    kl01=.5*(d-c+(v0+delta)/v1-1).sum(-1)
    kl10=.5*(c-d+(v1+delta)/v0-1).sum(-1)
    return dict(symmetric_kl=(kl01+kl10)/2,
                mean_difference_squared=delta.sum(-1),
                mean_difference_mahalanobis_squared=.5*(delta/v0+delta/v1).sum(-1),
                logvar_difference_squared=(c-d).square().sum(-1))
