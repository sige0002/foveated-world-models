"""Prespecified train-only interaction probes for every completed repair head."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.local_repair_evaluation_common import (
    SEEDS,validate_run_map,load_completed_model,load_inputs,anchors,RepairModel,mean_codes,file_hash,
)
from research.idm.probe_interaction512 import combine
from research.idm.probe_spatial_current import geometry,solve_many,CONFIGS
from research.idm.probe_nested512_displacement import ALPHAS,gripper_metrics
from research.idm.probe_displacement_ridge import scaler,FLOOR


def select_train_scores(cv):
    """Accept training CV errors only; evaluation errors are not an input."""
    selections={}
    for variant,scores in cv.items():
        candidates=[dict(config=c,alpha=a,cv_normalized_mse=float(scores[c][j].mean()),per_example=scores[c][j].tolist()) for c in CONFIGS for j,a in enumerate(ALPHAS)]
        chosen=min(candidates,key=lambda v:v['cv_normalized_mse'])
        selections[variant]=dict(selected=chosen,candidates=candidates,alpha_at_boundary=chosen['alpha'] in (ALPHAS[0],ALPHAS[-1]))
    return selections


def load_labels(data):
    prior=ROOT/'research/idm/nested512-displacement-probe'
    pp=json.loads((prior/'protocol.json').read_text());pr=json.loads((prior/'results.json').read_text())
    if pp['alphas']!=ALPHAS or pp['current_configs']!=CONFIGS:raise ValueError('Probe search changed')
    labelpaths={s:(prior/'train_labels.pt' if s=='train' else ROOT/f'research/idm/observed-displacement-labels/{s}.pt') for s in data}
    images={};states={};targets={};hashes={str(prior/'protocol.json'):file_hash(prior/'protocol.json'),str(prior/'results.json'):file_hash(prior/'results.json')}
    for s,d in data.items():
        path=labelpaths[s];digest=file_hash(path)
        expected=pr['labels_sha256'] if s=='train' else pp['source_sha256'][str(path)]
        if digest!=expected:raise ValueError('Label artifact changed')
        hashes[str(path)]=digest;l=torch.load(path,weights_only=True)
        if len(l['rows'])!=len(d['rows']) or not l['delta8_mask'].all():raise ValueError('Label count/mask mismatch')
        for r,lr in zip(d['rows'],l['rows']):
            if any(r[k]!=lr[k] for k in ('episode','index','dt')):raise ValueError('Label row mismatch')
        images[s]=d['current'].flatten(1).float();states[s]=l['current_state20'].float();targets[s]=l['delta8'].double()
    fitpath=prior/'fit_artifacts.pt'
    if file_hash(fitpath)!=pr['fit_artifacts_sha256']:raise ValueError('Prior calibration artifact changed')
    hashes[str(fitpath)]=file_hash(fitpath);common=torch.load(fitpath,weights_only=True)['common_target_scale128']
    return images,states,targets,common,pp,hashes


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs-json',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    runs=validate_run_map(json.loads(a.runs_json.read_text()))
    data,x,head,decoder,sources=load_inputs();reference=anchors(head,x);constant=reference['train']['mu'].mean(0,keepdim=True)
    baseline=RepairModel(head,decoder,constant,'F').eval().requires_grad_(False)
    baseline_codes=mean_codes(baseline,x)
    codes={'zero':{s:torch.zeros_like(z).reshape(len(z),32).double() for s,z in baseline_codes.items()},
           'F':{s:z.reshape(len(z),32).double() for s,z in baseline_codes.items()}}
    metadata={};d_checks={}
    for arm in ('H','HD','D'):
        for seed in SEEDS:
            key=f'{arm}{seed}';m,metadata[key]=load_completed_model(runs[key],arm,seed,head,decoder,constant,sources,reference)
            mu=mean_codes(m,x)
            if arm=='D':
                d_checks[key]={s:torch.equal(mu[s],baseline_codes[s]) for s in data}
                if not all(d_checks[key].values()):raise ValueError('D code differs from F')
            else:codes[key]={s:z.reshape(len(z),32).double() for s,z in mu.items()}
            del m
    images,states,targets,common_scale,previous,label_hashes=load_labels(data)
    folds=[torch.tensor(v) for v in previous['folds']]
    if sorted(torch.cat(folds).tolist())!=list(range(512)):raise ValueError('Fold partition differs')
    protocol=dict(variants=list(codes),aliases={**{f'F{s}':'F' for s in SEEDS},**{f'D{s}':'F' for s in SEEDS}},D_codes_exact=d_checks,
                  alphas=ALPHAS,current_configs=CONFIGS,folds=previous['folds'],bin_thresholds=previous['bin_thresholds'],
                  mean_code_precision='highest/B8 from mean-only RepairModel, fixed original variance unused in mu probe',
                  selection='Train-only5fold choice for every head; all selections saved before eval. Report every group, including drift failures. Never choose repair checkpoint/seed by observed labels.',
                  kernels='Existing centered/unit-diagonal current+linear code+centered/unit-diagonal Schur product; zero uses exact current-only kernel',
                  target_scaling='Train-fold fitting with original floors; fixed old128 evaluation target scale',
                  sources=sources,label_sources=label_hashes,completed_runs=metadata,runs_json_sha256=file_hash(a.runs_json),
                  script_sha256=file_hash(Path(__file__)),kernel_script_sha256=file_hash(ROOT/'research/idm/probe_interaction512.py'),
                  limitations='Finite readout accessibility, not conditional MI/action identification. Existing observed64/96; not independent finaltest. F may differ numerically from historical medium-code probe; zero should reproduce.')
    a.output.mkdir(parents=True);(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    torch.save(dict(codes=codes,rows={s:d['rows'] for s,d in data.items()}),a.output/'evaluated_codes.pt')
    cv={v:{c:torch.empty(len(ALPHAS),512,dtype=torch.float64) for c in CONFIGS} for v in codes};allidx=torch.arange(512)
    for fi,held in enumerate(folds):
        keep=allidx[~torch.isin(allidx,held)];g,q,n,_=geometry(images['train'][keep],states['train'][keep],images['train'][held],states['train'][held])
        ym,ys=scaler(targets['train'][keep],FLOOR);y=(targets['train'][keep]-ym)/ys
        for config,mult in CONFIGS.items():
            for variant,z in codes.items():
                k,cross,_=combine(g,q,n,z['train'][keep],z['train'][held],mult,'zero' if variant=='zero' else 'interaction')
                pred=solve_many(k,cross,y,ALPHAS)
                for j,prediction in enumerate(pred):cv[variant][config][j,held]=(prediction-(targets['train'][held]-ym)/ys).square().mean(-1)
        print('train CV fold',fi+1,'complete',flush=True)
    selection=select_train_scores(cv)
    oldselection=json.loads((ROOT/'research/idm/nested512-displacement-probe/train_selection.json').read_text())
    for old,new in zip(oldselection['zero']['candidates'],selection['zero']['candidates']):
        if old['config']!=new['config'] or old['alpha']!=new['alpha'] or abs(old['cv_normalized_mse']-new['cv_normalized_mse'])>1e-12:raise ValueError('Zero control CV not reproduced')
    (a.output/'train_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    # Only now compute observed-split predictions with fixed train selections.
    splits=list(data);g,q,n,current_scalers=geometry(images['train'],states['train'],torch.cat([images[s] for s in splits]),torch.cat([states[s] for s in splits]))
    ym,ys=scaler(targets['train'],FLOOR);y=(targets['train']-ym)/ys;results={};models={};bins=previous['bin_thresholds']
    for variant,z in codes.items():
        selected=selection[variant]['selected'];mult=CONFIGS[selected['config']]
        k,cross,norm=combine(g,q,n,z['train'],torch.cat([z[s] for s in splits]),mult,'zero' if variant=='zero' else 'interaction')
        prediction=solve_many(k,cross,y,[selected['alpha']])[0]*ys+ym;start=0;metrics={}
        for s in splits:
            count=len(data[s]['rows']);pred=prediction[start:start+count];truth=targets[s];sq=(pred-truth).square();scaled=sq/common_scale.square()
            metrics[s]=dict(normalized_mse_old128_scale=float(scaled.mean()),normalized_mse_train512_scale=float((sq/ys.square()).mean()),physical_coordinate_mse=sq.mean(0).tolist(),right_gripper=gripper_metrics(truth,pred,bins['static'],bins['small_large']),rows=[dict(r,prediction=pred[i].tolist(),target=truth[i].tolist(),physical_coordinate_squared_error=sq[i].tolist(),normalized_mse_old128_scale=float(scaled[i].mean())) for i,r in enumerate(data[s]['rows'])]);start+=count
        results[variant]=dict(selected_config=selected['config'],selected_alpha=selected['alpha'],evaluation=metrics)
        models[variant]=dict(kernel=k,normalization=norm,alpha=selected['alpha'])
        print(variant,{s:v['normalized_mse_old128_scale'] for s,v in metrics.items()},flush=True)
    previous_results=json.loads((ROOT/'research/idm/nested512-displacement-probe/results.json').read_text())
    for s in splits:
        if abs(results['zero']['evaluation'][s]['normalized_mse_old128_scale']-previous_results['results']['zero']['selected_evaluation'][s]['normalized_mse_old128_scale'])>1e-12:raise ValueError('Zero eval not reproduced')
    for path,digest in {**sources,**label_hashes}.items():
        if file_hash(Path(path))!=digest:raise ValueError('Immutable source changed')
    torch.save(dict(models=models,current_scalers=current_scalers,target_mean512=ym,target_scale512=ys,common_target_scale128=common_scale),a.output/'fit_artifacts.pt')
    (a.output/'results.json').write_text(json.dumps(dict(protocol_sha256=file_hash(a.output/'protocol.json'),selection_sha256=file_hash(a.output/'train_selection.json'),codes_sha256=file_hash(a.output/'evaluated_codes.pt'),fit_sha256=file_hash(a.output/'fit_artifacts.pt'),results=results,aliases=protocol['aliases'],zero_previous_cv_and_eval_reproduced=True,D_codes_exact=d_checks),indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
