"""Separate flow integration error from learning error at a fixed checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
import torch
from research.train_cached_decoders import make_decoder, evaluate, file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--integration-steps',type=int,nargs='+',default=[1,2,4,8,16])
    args=parser.parse_args()
    if min(args.integration_steps)<1 or args.output.exists():
        parser.error('positive integration steps and a new output path required')
    torch.set_num_threads(4)
    manifest=json.loads((args.cache/'cache_manifest.json').read_text())
    assert file_hash(args.cache/'validation.pt')==manifest['cache_sha256']['validation']
    data=torch.load(args.cache/'validation.pt',weights_only=True)
    saved=torch.load(args.checkpoint,weights_only=True)
    if saved['variant'] not in ('flow', 'endpoint_flow', 'endpoint_anchored', 'noisy_residual', 'wide_flow'):
        parser.error('flow checkpoint required')
    h=data['current']
    model=make_decoder(saved['variant'],saved['dim'],saved['layers'],h.shape[-1],data['codes'].shape[-1],int(h.shape[-2]**.5))
    model.load_state_dict(saved['state_dict'],strict=True)
    results=[]
    for steps in args.integration_steps:
        model.steps=steps
        start=time.perf_counter()
        result=evaluate(model,data)
        result.update(integration_steps=steps,seconds=time.perf_counter()-start)
        results.append(result)
        print(steps,result['metrics'],flush=True)
    args.output.write_text(json.dumps(dict(checkpoint=str(args.checkpoint),results=results),indent=2)+'\n')


if __name__=='__main__':
    main()
