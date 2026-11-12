"""Train-only uncentered PCA of prior errors; evaluation projections are oracles."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--observed-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    paths = {'train': args.cache/'train.pt', 'dev64': args.cache/'validation.pt',
             'observed96': args.observed_cache/'validation.pt'}
    for cache in (args.cache, args.observed_cache):
        manifest = json.loads((cache/'cache_manifest.json').read_text())
        for split in (('train', 'validation') if cache == args.cache else ('validation',)):
            assert digest(cache/f'{split}.pt') == manifest['cache_sha256'][split]
    protocol = dict(primary_rank=128, diagnostic_ranks=[32,64,128,256,512,768],
                    basis='Uncentered train512 second moment of future minus fixed current-only prior; no whitening, no mean offset.',
                    scope='Oracle projection floor, not predicted error. Fit/save basis before evaluation. Rank128 prespecified, not selected from eval.',
                    source_sha256={k:digest(p) for k,p in paths.items()})
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    train = torch.load(paths['train'], weights_only=True)
    dim = train['future'].shape[-1]
    moment = torch.zeros(dim, dim, dtype=torch.float64)
    count = 0
    for start in range(0, len(train['future']), 8):
        delta = (train['future'][start:start+8]-train['baseline'][start:start+8]).reshape(-1,dim).double()
        moment += delta.T @ delta
        count += len(delta)
    values, basis = torch.linalg.eigh(moment/count)
    values, basis = values.flip(0), basis.flip(1)
    torch.save({'basis':basis[:,:128].float(), 'eigenvalues':values,
                'source_train_sha256':protocol['source_sha256']['train']}, args.output/'basis128.pt')
    result = {'basis_sha256':digest(args.output/'basis128.pt'), 'splits':{}}
    del train
    for split,path in paths.items():
        data = torch.load(path, weights_only=True)
        rows = []
        for start in range(0,len(data['future']),8):
            delta = (data['future'][start:start+8]-data['baseline'][start:start+8]).double()
            coeff = delta @ basis
            for j in range(len(delta)):
                total = delta[j].square().mean()
                energy = coeff[j].square().reshape(-1,dim).mean(0).cumsum(0)/dim
                rows.append({'episode':data['rows'][start+j]['episode'],
                             'prior_mse':float(total),
                             'oracle_mse':{str(k):float((total-energy[k-1]).clamp_min(0)) for k in protocol['diagnostic_ranks']}})
        result['splits'][split] = {'prior_mse':sum(r['prior_mse'] for r in rows)/len(rows),
                                  'oracle_mse':{str(k):sum(r['oracle_mse'][str(k)] for r in rows)/len(rows) for k in protocol['diagnostic_ranks']},
                                  'rows':rows}
        print(split, result['splits'][split]['oracle_mse'], flush=True)
    (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
