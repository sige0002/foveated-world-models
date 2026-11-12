"""All completed local-repair heads/decoders: fixed same-current future-swap diagnostic."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.local_repair_evaluation_common import (
    SEEDS,ARMS,INDICES,validate_run_map,load_completed_model,load_inputs,
    anchors,RepairModel,gaussian_pair_metrics,file_hash,
)
from research.idm.train_reference_heads import representation


@torch.inference_mode()
def diagnose(model,data,xtrue,xswap,lvtrue,lvswap,rows):
    true=model.mean(xtrue);swapped=model.mean(xswap)
    metrics=gaussian_pair_metrics(true,swapped,lvtrue,lvswap)
    h,b,y=[data[k][INDICES] for k in ('current','baseline','future')]
    pred=model.decoder(h,true,baseline=b);changed=model.decoder(h,swapped,baseline=b)
    metrics.update(decoder_output_difference_mse=(pred-changed).square().flatten(1).mean(1),
                   true_future_mse=(pred-y).square().flatten(1).mean(1),
                   swapped_code_true_target_mse=(changed-y).square().flatten(1).mean(1))
    return dict(means={k:float(v.mean()) for k,v in metrics.items()},rows=[dict(rows[i],**{k:float(v[i]) for k,v in metrics.items()}) for i in range(len(rows))])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runs-json',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    runs=validate_run_map(json.loads(a.runs_json.read_text()))
    data,x,head,decoder,sources=load_inputs();reference=anchors(head,x);constant=reference['train']['mu'].mean(0,keepdim=True)
    qp=ROOT/'research/idm/reference-query-full/validation.pt';q=torch.load(qp,weights_only=True)
    calibration_path=ROOT/'research/idm/reference-query-full/train_calibration.pt';cal=torch.load(calibration_path,weights_only=True)
    swapped=representation(dict(query=q['future_swapped'],reference=q['reference']),cal,'reference')[INDICES]
    true=x['validation'][INDICES]
    # The variance function remains the original full64 head for each input,
    # not the updated mean head, and not a single variance shared across swaps.
    lvtrue=head(true).chunk(2,-1)[1].clamp(-10,10)
    lvswap=head(swapped).chunk(2,-1)[1].clamp(-10,10)
    rows=[dict(data['validation']['rows'][i],diagnostic_index=i,swapped_source_cache_row=int(q['indices'][(i+1)%len(q['indices'])])) for i in INDICES]
    metadata={}
    for arm in ARMS:
        for seed in SEEDS:
            key=f'{arm}{seed}';m,metadata[key]=load_completed_model(runs[key],arm,seed,head,decoder,constant,sources,reference);del m
    protocol=dict(indices=INDICES,query_sha256=file_hash(qp),calibration_sha256=file_hash(calibration_path),runs_json_sha256=file_hash(a.runs_json),completed_runs=metadata,
                  source_hashes=sources,variance='Frozen original full64 head logvar on true and swapped query separately; clamp[-10,10], highest/B8',
                  scope='Same-current response and actual decoder sensitivity, not realizable action counterfactuals, camera invariance, or information-theoretic utility. All results, including drift-scope failures, retained.',
                  fixed_baseline_aliases={f'F{s}':'F' for s in SEEDS},script_sha256=file_hash(Path(__file__)))
    a.output.mkdir(parents=True);(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    baseline=RepairModel(head,decoder,constant,'F').eval().requires_grad_(False)
    results={'F':diagnose(baseline,data['validation'],true,swapped,lvtrue,lvswap,rows)}
    for arm in ARMS:
        for seed in SEEDS:
            key=f'{arm}{seed}';model,_=load_completed_model(runs[key],arm,seed,head,decoder,constant,sources,reference)
            if arm=='D':
                if not torch.equal(model.mean(true),baseline.mean(true)) or not torch.equal(model.mean(swapped),baseline.mean(swapped)):raise ValueError('D changed fixed mean codes')
            results[key]=diagnose(model,data['validation'],true,swapped,lvtrue,lvswap,rows)
    for path,digest in sources.items():
        if file_hash(Path(path))!=digest:raise ValueError('Source changed')
    (a.output/'results.json').write_text(json.dumps(dict(protocol_sha256=file_hash(a.output/'protocol.json'),results=results),indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
