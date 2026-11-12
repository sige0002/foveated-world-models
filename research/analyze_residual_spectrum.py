"""Training-only residual PCA: representation diagnostic, NOT a future predictor."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    torch.set_num_threads(4)
    train=torch.load(args.cache/'train.pt',weights_only=True)
    val=torch.load(args.cache/'validation.pt',weights_only=True)
    d=(train['future']-train['current']).flatten(0,-2).double()
    dv=(val['future']-val['current']).flatten(0,-2).double()
    mean=d.mean(0)
    centered=d-mean
    cov=centered.T@centered/len(d)
    eigenvalues,basis=torch.linalg.eigh(cov)
    eigenvalues=eigenvalues.flip(0).clamp_min(0)
    basis=basis.flip(1)
    rows=[]
    for rank in (16,32,64,128,256,512,d.shape[-1]):
        if rank>d.shape[-1]:
            continue
        vectors=basis[:,:rank]
        # This uses the true future to encode residuals: an oracle reconstruction bound.
        residual=(dv-mean)-((dv-mean)@vectors)@vectors.T
        rows.append(dict(rank=rank, train_centered_variance_fraction=float(eigenvalues[:rank].sum()/eigenvalues.sum()),
                         val_oracle_projection_mse=float(residual.square().mean())))
    report=dict(dimension=d.shape[-1], train_pairs=len(train['current']), val_pairs=len(val['current']),
                participation_rank=float(eigenvalues.sum().square()/eigenvalues.square().sum()),
                train_copy_mse=float(d.square().mean()), val_copy_mse=float(dv.square().mean()),
                val_train_mean_residual_mse=float((dv-mean).square().mean()), rows=rows,
                interpretation='PCA fitted only on train patch residuals. Oracle projection uses future features; values are NOT prediction accuracy. No dimensionality was selected using validation.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
