"""Run the unchanged native-model diagnostic on an idle GPU; cache for summary."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
import torch
from research.evaluate_z_utility_matched import NativeModel,evaluate,SOURCE,CHECKPOINT,TEST,SELECTION
from research.train_early_condition import load_inputs,file_hash
from research.train_difference_idm import write


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--device',default='cuda:1');a=p.parse_args()
    for arm in ('external-real','direct-constant','external-constant'):
        for seed in (42,43,44):
            r=json.loads((a.runs/f'{arm}-{seed}'/'results.json').read_text())
            assert r['complete'] and not r['smoke_only'] and r['history'][-1]['step']==400
    target=a.runs/'original-native.json'
    if target.exists():raise FileExistsError(target)
    torch.cuda.set_device(a.device);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
    data,base,hashes,_=load_inputs(SOURCE);del base
    meta=json.loads(TEST.with_suffix('.json').read_text())
    assert file_hash(TEST)==meta['cache_sha256'] and file_hash(SELECTION)==meta['selection_sha256']
    plan=json.loads(SELECTION.read_text());test=torch.load(TEST,weights_only=True,map_location='cpu')
    assert test['rows']==plan['selected'];data['reused_test170']=test
    groups={}
    for path in plan['manifest_paths'].values():
        for line in Path(path).read_text().splitlines():
            row=json.loads(line);groups[str(row['episode_id'])]=row['split_group']
    digest=file_hash(CHECKPOINT);assert digest==meta['checkpoint_sha256']
    model=NativeModel().to(a.device);checks={}
    for split in ['train','validation','observed96']:
        if 'native_codes' in data[split]:
            x,y=[data[split][k][:8].to(a.device) for k in ['current','future']]
            error=float((model.code(x,y).cpu()-data[split]['native_codes'][:8]).abs().max())
            assert error<1e-3,error;checks[split]=error
    result={'checkpoint_sha256':digest,'script_sha256':file_hash(ROOT/'research/evaluate_z_utility_matched.py'),
            'worker_script_sha256':file_hash(Path(__file__)),'device':a.device,
            'native_cache_max_abs':checks,'evaluation':evaluate(model,data,a.device,groups),
            'scope':'Local final10000 native posterior mean; not official distributed weights. null means C,C, not constant; mean and zero are constants.'}
    temporary=target.with_suffix('.ready');write(temporary,result)
    # Atomic, no-clobber publication; the coordinator can reuse only a complete file.
    import os
    try:os.link(temporary,target)
    except FileExistsError:raise RuntimeError('Coordinator already wrote native results; retained .ready')
    temporary.unlink();print('NATIVE COMPLETE',flush=True)


if __name__=='__main__':main()
