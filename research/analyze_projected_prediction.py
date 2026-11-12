"""Decompose fixed predictions by train-only subspace; no evaluation fitting."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from research.train_cached_decoders import make_decoder, file_hash


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--basis', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    torch.set_num_threads(4)
    manifest = json.loads((args.cache/'cache_manifest.json').read_text())
    assert file_hash(args.cache/'validation.pt') == manifest['cache_sha256']['validation']
    data = torch.load(args.cache/'validation.pt', weights_only=True)
    basis = torch.load(args.basis, weights_only=True)['basis']
    project = lambda value: (value @ basis) @ basis.T
    mse = lambda value: value.square().flatten(1).mean(1)
    report = dict(scope='Previously observed evaluation; posterior z; no target fitting. Postprojection is diagnostic, not a new trained-model result.',
                  basis_sha256=file_hash(args.basis), cache_sha256=file_hash(args.cache/'validation.pt'), results=[])
    for path in args.checkpoints:
        saved = torch.load(path, weights_only=True)
        model = make_decoder(saved['variant'], saved['dim'], saved['layers'],
                             data['current'].shape[-1],data['codes'].shape[-1],math.isqrt(data['current'].shape[-2]))
        model.load_state_dict(saved['state_dict'], strict=True)
        model.eval()
        gen = torch.Generator().manual_seed(2026)
        rows = []
        for start in range(0,len(data['current']),8):
            h,y,z,b = [data[k][start:start+8] for k in ('current','future','codes','baseline')]
            draws = []
            for _ in range(4 if model.stochastic else 1):
                options = dict(noise=torch.randn(h.shape,generator=gen)) if model.stochastic else {}
                draws.append(model(h,z,baseline=b,**options))
            predictions = torch.stack(draws)
            error = predictions-y
            avg = predictions.mean(0)
            avg_error = avg-y
            retained = project(avg_error)
            removed = avg_error-retained
            postprojected = b+project(avg-b)
            metrics = {'sample_mse':error.square().flatten(2).mean(2).mean(0),
                       'mean_mse':mse(avg_error),'retained_error_mse':mse(retained),
                       'discarded_error_mse':mse(removed),'postprojected_mean_mse':mse(postprojected-y),
                       'discarded_prediction_correction_energy':mse((avg-b)-project(avg-b))}
            torch.testing.assert_close(metrics['mean_mse'],metrics['retained_error_mse']+metrics['discarded_error_mse'],atol=1e-7,rtol=1e-5)
            for j in range(len(h)):
                rows.append(dict(episode=data['rows'][start+j]['episode'],**{k:float(v[j]) for k,v in metrics.items()}))
        result = dict(checkpoint=str(path),checkpoint_sha256=file_hash(path),variant=saved['variant'],
                      means={k:sum(r[k] for r in rows)/len(rows) for k in metrics},rows=rows)
        report['results'].append(result)
        print(saved['variant'],result['means'],flush=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
