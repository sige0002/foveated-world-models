"""Verify the exact output-subspace noise obstruction of the narrow velocity head."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
import torch
from research.train_cached_decoders import make_decoder


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('output exists')
    torch.set_num_threads(4)
    data=torch.load(args.cache/'validation.pt',weights_only=True)
    saved=torch.load(args.checkpoint,weights_only=True)
    h=data['current'][:1].expand(16,-1,-1,-1).clone()
    z=data['codes'][:1].expand(16,-1,-1).clone()
    model=make_decoder('flow',saved['dim'],saved['layers'],h.shape[-1],z.shape[-1],int(h.shape[-2]**.5)).eval()
    model.load_state_dict(saved['state_dict'],strict=True)
    weight=model.output.weight.double()
    u,s,_=torch.linalg.svd(weight,full_matrices=False)
    rank=int((s > s.max()*1e-10).sum())
    basis=u[:,:rank]
    def perp(x):
        x=x.double()
        return x-(x@basis)@basis.T
    gen=torch.Generator().manual_seed(2026)
    noise=torch.randn(h.shape,generator=gen)
    expected=model.noise_scale*perp(noise)
    bias=model.output.bias.double()
    rows=[]
    for steps in (1,4,16):
        predicted=model(h,z,noise=noise,steps=steps)
        measured=perp(predicted.double()-h.double()-bias)
        error=float((measured-expected).abs().max())
        assert error < 2e-5, error
        rows.append(dict(steps=steps,max_projected_identity_error=error,
                         observed_orthogonal_noise_power=float(measured.square().mean())))
    report=dict(feature_dim=h.shape[-1], output_rank=rank, noise_scale=model.noise_scale,
                expected_unremovable_noise_mse=model.noise_scale**2*(h.shape[-1]-rank)/h.shape[-1],
                proof='v=W f+b implies P_perp(x_final-h-b)=sigma P_perp(epsilon) for Euler integration over unit time. This is a noise contribution, not the entire MSE.',
                checkpoints_scope=str(args.checkpoint), measurements=rows)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
