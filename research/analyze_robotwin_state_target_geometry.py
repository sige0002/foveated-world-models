"""CPU-only target/residual spectrum and train-PCA held-out association audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/media/sadasue/SSD1/foveated-world-models/runs/robotwin-loss-ablation-b4-20261114/evaluations/matched-state-probes-fp32-20261114')
GROUPS = {'XYZ': [0, 1, 2, 8, 9, 10], 'quaternion': [3, 4, 5, 6, 11, 12, 13, 14], 'gripper': [7, 15]}
NAMES = [f'{side}_{axis}' for side in ('left', 'right') for axis in ('x','y','z','qw','qx','qy','qz','gripper')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(value):
    return value.numpy().astype(np.float64)


def geometry(x):
    center=x.mean(0); c=x-center; covariance=c.T@c/len(x)
    eig=np.maximum(np.linalg.eigvalsh(covariance),0)[::-1]; prop=eig/eig.sum()
    second=np.mean(x*x,axis=0); variance=np.var(x,axis=0)
    return {'samples':len(x),'mean':dict(zip(NAMES,center.tolist())),
        'variance':dict(zip(NAMES,variance.tolist())), 'second_moment':dict(zip(NAMES,second.tolist())),
        'mean_square_16d':float(second.mean()), 'centered_variance_trace':float(eig.sum()),
        'pc1_fraction':float(prop[0]),'pc1_plus_pc2_fraction':float(prop[:2].sum()),
        'participation_ratio':float(1/np.sum(prop*prop)), 'dimensions_95pct':int(np.searchsorted(prop.cumsum(),.95)+1),
        'covariance_eigenvalues':eig.tolist(),
        'groups':{k:{'mean_squared_per_coordinate':float(second[ix].mean()),
                     'second_moment_fraction':float(second[ix].sum()/second.sum()),
                     'centered_variance_fraction':float(variance[ix].sum()/variance.sum()),
                     'contribution_to_16d_MSE':float(second[ix].sum()/16)} for k,ix in GROUPS.items()}}


def corr(x,y):
    x=x-x.mean(); y=y-y.mean(0)
    return (x[:,None]*y).mean(0)/np.sqrt(np.mean(x*x)*np.mean(y*y,axis=0))


def main():
    torch.set_num_threads(1)
    report=json.loads((SOURCE/'results.json').read_text()); protocol=report['protocol']
    old_path=Path(protocol['old_codes_path']);assert digest(old_path)==protocol['old_codes_sha256']
    old=torch.load(old_path,weights_only=True,map_location='cpu')
    yt,yv=[array(old[s]['delta']) for s in ('train','val')]
    st,sv=[array(old[s]['state']) for s in ('train','val')]
    assert yt.shape==(2048,16) and yv.shape==(512,16)
    assert not {r['episode'] for r in old['train']['rows']} & {r['episode'] for r in old['val']['rows']}
    eps,inv=np.unique([r['episode'] for r in old['val']['rows']],return_inverse=True);assert len(eps)==415
    picks=np.random.default_rng(142).integers(len(eps),size=(4000,len(eps)));den=np.bincount(inv)[picks].sum(1)
    def ci(values):
        draws=np.bincount(inv,weights=values)[picks].sum(1)/den
        return {'estimate':float(values.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist()}
    models={}; reference_residual=None
    for name in ('official','old10000','baseline3000','combined3000'):
        codes=torch.load(SOURCE/f'{name}-posterior-means.pt',map_location='cpu',weights_only=True)
        read=torch.load(SOURCE/f'{name}-readouts.pt',map_location='cpu',weights_only=True)
        zt,zv=[array(codes[s]['z']) for s in ('train','val')]
        fit={k:array(v) for k,v in read['fits']['state'].items()}
        pred_t=(st-fit['xm'])/fit['xs']@fit['w']+fit['ym'];pred_v=array(read['predictions']['state'])
        np.testing.assert_allclose((sv-fit['xm'])/fit['xs']@fit['w']+fit['ym'],pred_v,atol=1e-12,rtol=0)
        residual_t,residual_v=yt-pred_t,yv-pred_v
        if reference_residual is None:reference_residual=residual_v
        np.testing.assert_array_equal(residual_v,reference_residual)
        # PCA uses train means/covariance only. Sign is arbitrary; fix its largest loading positive.
        zmean=zt.mean(0); eig,vec=np.linalg.eigh(np.cov(zt,rowvar=False,bias=True));pc=vec[:,-1]
        if pc[np.argmax(np.abs(pc))]<0:pc=-pc
        score_t=(zt-zmean)@pc;score_v=(zv-zmean)@pc
        # Fit score dependence on current state using train only, then measure held-out residual association.
        design_t=np.c_[np.ones(len(st)),(st-fit['xm'])/fit['xs']]
        design_v=np.c_[np.ones(len(sv)),(sv-fit['xm'])/fit['xs']]
        state_to_pc=np.linalg.lstsq(design_t,score_t,rcond=None)[0]
        rt=score_t-design_t@state_to_pc;rv=score_v-design_v@state_to_pc
        # One scalar posterior score correction to the frozen fitted state-only head; no tuning on validation.
        target_mean=residual_t.mean(0);rt_mean=rt.mean()
        slope=(rt-rt_mean)@(residual_t-target_mean)/np.square(rt-rt_mean).sum()
        corrected=pred_v+target_mean+(rv-rt_mean)[:,None]*slope
        predictions={k:array(v) for k,v in read['predictions'].items()}
        gains={}
        for group,ix in GROUPS.items():
            state_error=np.square(pred_v-yv)[:,ix].mean(1)
            gains[group]={
                'state_plus_all_mu_over_state':ci(state_error-np.square(predictions['state_z']-yv)[:,ix].mean(1)),
                'state_plus_all_mu_over_state_plus_current_mu':ci(np.square(predictions['state_current_only_z']-yv)[:,ix].mean(1)-np.square(predictions['state_z']-yv)[:,ix].mean(1)),
                'train_fit_PC1_residual_correction_over_state':ci(state_error-np.square(corrected-yv)[:,ix].mean(1))}
        models[name]={'train_mu_pc1_fraction':float(eig[-1]/eig.sum()),
            'validation_PC1_correlation_with_target':dict(zip(NAMES,corr(score_v,yv).tolist())),
            'validation_PC1_correlation_with_current_state':dict(zip(NAMES,corr(score_v,sv).tolist())),
            'validation_PC1_correlation_with_state_only_residual':dict(zip(NAMES,corr(score_v,residual_v).tolist())),
            'validation_state_residualized_PC1_correlation_with_state_only_residual':dict(zip(NAMES,corr(rv,residual_v).tolist())),
            'train_fit_current_state_explained_validation_PC1_fraction':float(1-np.var(rv)/np.var(score_v)),
            'per_group_external_readout_gain':gains,
            'posterior_codes_sha256':digest(SOURCE/f'{name}-posterior-means.pt'),
            'readouts_sha256':digest(SOURCE/f'{name}-readouts.pt')}
    result={'targets':{'train2048':geometry(yt),'val512':geometry(yv),'state_only_heldout_residual512':geometry(reference_residual)},
        'models':models,'source_result_sha256':digest(SOURCE/'results.json'),'old_codes_sha256':digest(old_path),
        'script_sha256':digest(__file__),'cuda_initialized':torch.cuda.is_initialized(),
        'methods':['Posterior PCA fits train2048 only; scores and correlations evaluated on disjoint512/415 episodes.',
            'Existing full-mu and state-only ridge fits/predictions are reused unchanged, with existing train-episode CV.',
            'Optional PC1 correction fits current-state-to-PC1 OLS and residual-to-one-score OLS on train only; fixed external diagnostic, no backbone or g update.',
            'Intervals use4000 paired target-episode bootstrap draws conditional on fitted representations/readouts.'],
        'limits':['Unconditional teacher energy does not establish actual g gradient dominance or causal mechanism.',
            'State-only residual refers to an external ridge baseline, not trained native g residual.',
            'PC1 correlations can reflect current visual context; conditioning only on measured current state does not remove image/exposure confounding.',
            'Full-mu ridge can amplify tiny axes; its gains cannot automatically be assigned to PC1 or native visual-decoder usage.',
            'PC1 is determined separately per representation; sign/alignment does not establish shared semantics.',
            'Quaternion componentwise squared delta is not geodesic rotation error.']}
    assert not result['cuda_initialized']
    out=ROOT/'research/results/robotwin-z-underuse-20261114/state-target-and-pc1-association.json'
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'targets':{k:{'pc1':v['pc1_fraction'],'pc12':v['pc1_plus_pc2_fraction'],'PR':v['participation_ratio'],'groups':v['groups']} for k,v in result['targets'].items()},'models':{k:{'gripper':v['per_group_external_readout_gain']['gripper'],'grip_residual_corr':{n:v['validation_state_residualized_PC1_correlation_with_state_only_residual'][n] for n in ('left_gripper','right_gripper')}} for k,v in models.items()}},indent=2))
    print(out)


if __name__=='__main__':main()
