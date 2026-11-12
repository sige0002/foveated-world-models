"""Test code dependence of the original trained decoder on the identical cached pairs."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT), str(ROOT/'vendor/lawam')]
import torch
import yaml
from research.train_cached_decoders import evaluate, file_hash
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    torch.set_num_threads(4)
    manifest=json.loads((args.cache/'cache_manifest.json').read_text())
    assert file_hash(args.cache/'validation.pt')==manifest['cache_sha256']['validation']
    data=torch.load(args.cache/'validation.pt',weights_only=True)
    config=yaml.safe_load((args.cache/'resolved_config.yaml').read_text())['model']
    grid=tuple(v//config['patch_size'] for v in config['image_hw'])
    model=LAMDecoder_v2(context_dim=config['dim'],input_dim=data['current'].shape[-1],
                       code_dim=config['code_dim'],num_queries=config['num_queries'],
                       num_layers=config['dec_layers'],num_heads=config['num_heads'],
                       dropout=config['dropout'],grid_hw=grid,
                       ffn_expansion_factor=config['ffn_expansion_factor'],
                       last_ln=config.get('decoder_last_ln',True)).eval()
    saved=torch.load(args.cache/'reference_weights.pt',weights_only=True,mmap=True)
    prefix='lam.decoder.'
    model.load_state_dict({k[len(prefix):]:v for k,v in saved['state_dict'].items() if k.startswith(prefix)},strict=True)
    step=saved['global_step']
    del saved
    report=evaluate(model,data,batch_size=2)
    report['checkpoint_step']=step
    # Compare with the original extraction to detect checkpoint/feature mismatch.
    max_error=max(abs(r['mse']-r['reference_mse']) for r in report['rows'])
    if max_error > 1e-4:
        raise RuntimeError(f'Original predictions did not reproduce: {max_error}')
    report['max_reproduction_error']=max_error
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':
    main()
