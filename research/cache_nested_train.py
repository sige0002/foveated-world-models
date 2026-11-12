"""CPU-only nested train512 cache with exact retained128 and frozen IDM/prior."""
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['LAM_ENABLE_MANUAL_WANDB'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'vendor/lawam')]
import torch
from torch import nn
import yaml
from research.cache_stage1_features import sha256
from research.cache_current_prior import predict_prior
from research.idm.cache_unused_episodes import state_hash
from research.idm.train_reference_heads import representation
from vendor.dataset.robomind import RoboMINDDataset, robomind_collate
from latent_action_model.core.lam_lightinng import VJEPA_LAM
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    selection_path = ROOT/'research/results/nested-train512-selection-20261113.json'
    selection = json.loads(selection_path.read_text())
    parent = Path(selection['parent'])
    output = ROOT/'runs/reference-idm42-nested512-prior10000'
    if output.exists():
        raise FileExistsError('Output already exists; inspect existing job/chunks before any restart')
    old = ROOT/'runs/frozen-idm-step2900'
    prior = ROOT/'runs/frozen-model-step10000'
    queries_path = ROOT/'research/idm/reference-query-full/train.pt'
    calibration_path = ROOT/'research/idm/reference-query-full/train_calibration.pt'
    meta = json.loads((parent/'cache_manifest.json').read_text())
    export = meta['export']
    head_path = Path(export['head_checkpoint'])
    sources = {str(p):sha256(p) for p in [selection_path, Path(selection['manifest']),
                old/'reference_weights.pt', prior/'reference_weights.pt', head_path,
                calibration_path, queries_path, parent/'cache_manifest.json']}
    assert sources[str(Path(selection['manifest']))] == selection['manifest_sha256']
    assert sources[str(head_path)] == export['head_checkpoint_sha256']
    assert sources[str(calibration_path)] == export['calibration_sha256']
    for split in ('train','validation'):
        p = parent/f'{split}.pt'
        sources[str(p)] = sha256(p)
        assert sources[str(p)] == selection['parent_cache_sha256'][split] == meta['cache_sha256'][split]
    retained = torch.load(parent/'train.pt', weights_only=True)
    assert retained['rows'] == selection['retained']
    qold = torch.load(queries_path, weights_only=True)
    assert qold['rows'] == retained['rows']
    cal = torch.load(calibration_path, weights_only=True)
    cfg = yaml.safe_load((old/'resolved_config.yaml').read_text())['model']
    pcfg = yaml.safe_load((prior/'resolved_config.yaml').read_text())['model']
    assert cfg == pcfg and not cfg['enc_add_state']
    dataset = RoboMINDDataset(selection['manifest'], **selection['options'])
    for row in selection['retained'] + selection['added']:
        i,a,b,_ = dataset.pairs[row['index']]
        arrays = dataset._arrays(i)
        assert str(dataset.rows[i]['episode_id']) == row['episode']
        assert abs(float(arrays['timestamps'][b]-arrays['timestamps'][a])-row['dt']) < 1e-9
        assert arrays['original_valid'][[a,b]].all()
        if 'endpoint_indices' in row:
            assert [a,b] == row['endpoint_indices']
    weights = torch.load(old/'reference_weights.pt', weights_only=True, mmap=True)
    pw = torch.load(prior/'reference_weights.pt', weights_only=True, mmap=True)
    assert weights['global_step'] == 2900 and pw['global_step'] == 10000
    assert state_hash(weights['state_dict'], 'lam.vision_encoder.') == state_hash(pw['state_dict'], 'lam.vision_encoder.')
    assert state_hash(weights['state_dict'], 'lam.encoder.') == export['encoder_state_sha256']
    model = VJEPA_LAM(**cfg).cpu().eval().requires_grad_(False)
    model.load_state_dict(weights['state_dict'], strict=True)
    grid = math.isqrt(retained['current'].shape[-2])
    decoder = LAMDecoder_v2(context_dim=cfg['dim'], input_dim=768, code_dim=32,
              num_queries=cfg['num_queries'], num_layers=cfg['dec_layers'], num_heads=cfg['num_heads'],
              dropout=0., grid_hw=(grid,grid), ffn_expansion_factor=cfg['ffn_expansion_factor'],
              last_ln=cfg.get('decoder_last_ln',True)).eval().requires_grad_(False)
    decoder.load_state_dict({k[len('lam.decoder.'):]:v for k,v in pw['state_dict'].items()
                            if k.startswith('lam.decoder.')}, strict=True)
    del weights,pw
    hs = torch.load(head_path, weights_only=True)
    assert state_hash(hs['state_dict'], 'head.') == export['head_state_sha256']
    head = nn.Sequential(nn.Linear(1024,256),nn.GELU(),nn.Linear(256,64)).eval().requires_grad_(False)
    head.load_state_dict({k[5:]:v for k,v in hs['state_dict'].items() if k.startswith('head.')})
    del hs
    capture = {}
    original = model.lam.encoder.out_proj.forward
    def capture_query(q):
        capture['query'] = q.clone()
        return original(q)
    model.lam.encoder.out_proj.forward = capture_query

    def extract(row):
        torch.set_float32_matmul_precision('medium')
        sample = dataset[row['index']]
        assert str(sample['trajectory_id']) == row['episode'] and abs(sample['dt']-row['dt']) < 1e-9
        batch = model.transfer_batch_to_device(robomind_collate([sample], max_state_dim=32), torch.device('cpu'), 0)
        out = model.lam._run(batch['videos'],batch['states'],batch['dec_videos'],
                 state_mask=batch['state_mask'],embodiment_ids=batch['embodiment_ids'],
                 vq_training=False,predict_future_frame=False)
        h,y = out[1].clone(),out[2].clone()
        q = capture['query'].clone()
        model.lam.encoder(torch.cat((h,h),1))
        return dict(current=h, future=y, query=q, reference=capture['query'].clone())

    # Verify the exact deployed extraction path before decoding new episodes.
    for j in (0,1):
        item = extract(retained['rows'][j])
        for key in ('current','future'):
            assert torch.equal(item[key],retained[key][j:j+1]), key
        for key in ('query','reference'):
            assert torch.equal(item[key],qold[key][j:j+1]), key
    torch.set_float32_matmul_precision('medium')
    assert torch.equal(head(representation(qold,cal,'reference')).chunk(2,-1)[0],retained['codes'])
    torch.set_float32_matmul_precision('highest')
    assert torch.equal(predict_prior(decoder,retained['current'][:2],32),retained['baseline'][:2])
    output.mkdir()
    (output/'extraction_protocol.json').write_text(json.dumps(dict(source_sha256=sources,
        selection=selection, script_sha256=sha256(Path(__file__)), smoke_exact=True,
        precision='Vision/query medium B1, head medium B128, prior highest B2, CPU4'),indent=2)+'\n')
    started = time.perf_counter()
    chunks = []
    for start in range(0,len(selection['added']),16):
        rows = selection['added'][start:start+16]
        extracted = [extract(row) for row in rows]
        chunk = {k:torch.cat([d[k] for d in extracted]) for k in extracted[0]}
        torch.set_float32_matmul_precision('highest')
        chunk['baseline'] = predict_prior(decoder,chunk['current'],32)
        assert all(torch.isfinite(v).all() for v in chunk.values())
        chunk['rows'] = rows
        p = output/f'chunk-{start:03d}.pt'
        torch.save(chunk,p)
        chunks.append(dict(path=str(p),sha256=sha256(p)))
        print('Added',start+len(rows),'/384',flush=True)
    loaded = [torch.load(v['path'],weights_only=True) for v in chunks]
    newq = {k:torch.cat([v[k] for v in loaded]) for k in ('query','reference')}
    torch.set_float32_matmul_precision('medium')
    x = representation(newq,cal,'reference')
    newcodes = torch.cat([head(xi).chunk(2,-1)[0] for xi in x.split(128)])
    expanded = {k:torch.cat([retained[k]]+[v[k] for v in loaded]) for k in ('current','future','baseline')}
    expanded['codes'] = torch.cat((retained['codes'],newcodes))
    expanded['rows'] = retained['rows'] + selection['added']
    for k in ('current','future','baseline','codes'):
        assert torch.equal(expanded[k][:128],retained[k]) and torch.isfinite(expanded[k]).all()
    torch.save(expanded,output/'train.pt')
    shutil.copyfile(parent/'validation.pt',output/'validation.pt')
    torch.save(dict(**newq,rows=selection['added']),output/'additional_queries.pt')
    for p,digest in sources.items():
        assert sha256(p) == digest, p
    result = dict(meta,cache_sha256={s:sha256(output/f'{s}.pt') for s in ('train','validation')},
                  nested_parent=str(parent), nested_parent_cache_sha256=meta['cache_sha256'],
                  selection_sha256=sha256(selection_path), selection_path=str(selection_path),
                  extraction_source_sha256=sources, chunks=chunks,
                  nested_retained128_exact=True, count={'train':512,'validation':64},
                  scope=selection['scope'], extraction_seconds=time.perf_counter()-started,
                  frozen_head_training_pairs=128, extraction_script_sha256=sha256(Path(__file__)))
    result.pop('prior_metrics',None)
    result['prior_metrics']={'train':{'mse':float((expanded['baseline']-expanded['future']).square().mean())},
                            'validation':meta['prior_metrics']['validation']}
    (output/'cache_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Completed',output,flush=True)


if __name__ == '__main__':
    main()
