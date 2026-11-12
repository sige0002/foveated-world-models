"""Fixed16-draw validation of the flow mean-risk estimate; no model fitting."""
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from research.train_cached_decoders import make_decoder,file_hash


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    cache=ROOT/'runs/reference-idm42-observed96-prior10000'
    checkpoint=ROOT/'runs/reference-idm42-final-prior-flow42-20261113/prior_flow-seed42.pt'
    output=ROOT/'research/results/final-prior-flow-mc16-20261113.json'
    protocol=output.with_name(output.stem+'-protocol.json')
    if output.exists() or protocol.exists():
        raise FileExistsError('Outputs must be new')
    manifest=json.loads((cache/'cache_manifest.json').read_text())
    cache_hash=file_hash(cache/'validation.pt')
    assert cache_hash==manifest['cache_sha256']['validation']
    data=torch.load(cache/'validation.pt',weights_only=True)
    saved=torch.load(checkpoint,weights_only=True)
    assert saved['variant']=='prior_flow' and saved['steps']==400 and saved['seed']==42
    model=make_decoder('prior_flow',saved['dim'],saved['layers'],data['current'].shape[-1],
                       data['codes'].shape[-1],math.isqrt(data['current'].shape[-2])).eval()
    model.load_state_dict(saved['state_dict'],strict=True)
    protocol.write_text(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),
         cache_sha256=cache_hash,script_sha256=file_hash(Path(__file__)),noise_seed=20261113,
         samples=16,solver_steps=4,batch_size=8,threads=4,
         primary='Mean of16 iid draws, fixed model/solver; secondary prefix1/4 from same16. No best-of-N, no model or code optimization.',
         scope='Previously observed96; independently seeded Monte Carlo evaluation, not new data.'),indent=2)+'\n')
    gen=torch.Generator().manual_seed(20261113)
    rows=[];started=time.perf_counter()
    for start in range(0,len(data['current']),8):
        h=data['current'][start:start+8];y=data['future'][start:start+8]
        z=data['codes'][start:start+8];base=data['baseline'][start:start+8]
        predictions=torch.stack([model(h,z,baseline=base,noise=torch.randn(h.shape,generator=gen)) for _ in range(16)])
        assert torch.isfinite(predictions).all()
        errors={str(n):(predictions[:n].mean(0)-y).square().flatten(1).mean(1) for n in (1,4,16)}
        sample=(predictions-y).square().flatten(2).mean(2).mean(0)
        for i in range(len(h)):
            rows.append(dict(episode=data['rows'][start+i]['episode'],index=data['rows'][start+i]['index'],
                 mean_prediction_mse={n:float(v[i]) for n,v in errors.items()},mean_sample_mse=float(sample[i])))
        print('Evaluated',start+len(h),'/96',flush=True)
    result=dict(protocol_sha256=file_hash(protocol),rows=rows,seconds=time.perf_counter()-started,
                mean_prediction_mse={str(n):sum(r['mean_prediction_mse'][str(n)] for r in rows)/len(rows) for n in (1,4,16)},
                mean_sample_mse=sum(r['mean_sample_mse'] for r in rows)/len(rows))
    assert file_hash(cache/'validation.pt')==cache_hash
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['mean_prediction_mse'],flush=True)


if __name__=='__main__':
    main()
