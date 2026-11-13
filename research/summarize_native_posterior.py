"""Summarize fixed-checkpoint diagnostics and verify numerical helper contracts."""
import ast
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Callable, Literal

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from research.investigate_native_posterior import gaussian_kl,posterior_stats,file_hash
from research.check_posterior_preprocessing import official_resize
from research.summarize_idm_ablation import paired_comparison


def verify():
    torch.manual_seed(20261113);torch.set_num_threads(2)
    tensors=[torch.randn(7,1,32,dtype=torch.float64) for _ in range(4)]
    m,l,om,ol=tensors
    normal=torch.distributions.Normal;ind=torch.distributions.Independent
    expected=torch.distributions.kl_divergence(ind(normal(m,(l*.5).exp()),1),ind(normal(om,(ol*.5).exp()),1))
    error=float((gaussian_kl(*tensors)-expected).abs().max());assert error<1e-10
    same=posterior_stats(torch.ones(8,1,32),torch.zeros(8,1,32),5e-5)
    assert abs(same['empirical_sample_identity_information_upper_bound_nats'])<1e-12
    assert same['noise_fraction_of_sampled_z_variance']==1
    # Execute the exact upstream class bodies, omitting only dependency-heavy
    # module imports and replacing Pydantic field declarations with no-op stubs.
    source=Path('/tmp/lawam-official-compare-20261113/starVLA/dataloader/gr00t_lerobot/transform/video.py')
    tree=ast.parse(source.read_text())
    selected=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('VideoMaxAspectCrop','VideoResize')]
    from torchvision.transforms import v2
    scope=dict(torch=torch,np=np,T=v2,Callable=Callable,Literal=Literal,VideoTransform=object,
        Field=lambda *args,**kwargs:None,field_validator=lambda *args,**kwargs:lambda f:f)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),scope)
    mock=SimpleNamespace(height=256,width=256,antialias=True,backend='torchvision',interpolation='linear',
        _get_interpolation=lambda *args:v2.InterpolationMode.BILINEAR)
    resize=scope['VideoResize'].get_transform(mock,'eval')
    sizes=[(720,1280),(480,640),(256,256),(640,480),(241,427)]
    for h,w in sizes:
        frames=torch.randint(0,256,(2,3,h,w),dtype=torch.uint8)
        ref=resize(scope['VideoMaxAspectCrop'].crop_to_max_aspect(frames,max_aspect=4/3))
        assert torch.equal(official_resize(frames),ref),(h,w)
    return dict(gaussian_kl_vs_torch_distribution_max_error=error,
        identical_posteriors_information_bound_zero=True,
        preprocessing_bitwise_equal_to_extracted_official_methods_sizes=sizes,
        official_video_source_sha256=file_hash(source),
        note='Full StarVLA transform module could not import without accelerate; tested exact AST-extracted crop/resize class methods, not the complete LeRobot loader.')


def main():
    out=ROOT/'research/results/native-posterior-investigation-20261113'
    loaded={label:json.loads((out/f'{label}.json').read_text()) for label in ('local','official')}
    pre=json.loads((out/'preprocessing.json').read_text());summary={}
    for label,d in loaded.items():
        assert d['script_sha256']==file_hash(ROOT/'research/investigate_native_posterior.py')
        split=d['splits']['reused_test170'];ev=split['evaluation'];m=ev['metrics'];rows=ev['rows']
        assert len({r['source_row']['episode'] for r in rows})==len(rows)
        groups=[r['source_row']['split_group'] for r in rows]
        comparisons={}
        for key in ('train_mean_mse','swapped_mse','same_current_other_future_mse'):
            comparisons[key+'_minus_real']=paired_comparison(
                np.array([[r[key] for r in rows]]),np.array([[r['mse'] for r in rows]]),groups)
        gradient=d['gradient_probe']['rows']
        summary[label]=dict(normal_mse=m['mse'],mean_constant_mse=m['train_mean_mse'],
            mean_constant_relative_increase=m['train_mean_mse']/m['mse']-1,
            mean_constant_output_rms_over_actual_feature_change=math.sqrt(m['train_mean_output_delta_mse']/m['copy_mse']),
            train_posterior=d['splits']['train']['posterior'],test_posterior=split['posterior'],
            intervention_comparisons=comparisons,
            mean_kl_over_reconstruction_nodes_gradient_norm=float(np.mean([r['kl_over_reconstruction_gradient_norm'] for r in gradient])),
            gradient_probe_scope=d['gradient_probe']['scope'])
    pre_summary={}
    for label,d in pre['results'].items():
        pre_summary[label]={}
        for mode,v in d['preprocessing'].items():
            m=v['evaluation']['metrics']
            pre_summary[label][mode]=dict(normal_mse=m['mse'],mean_constant_mse=m['train_mean_mse'],
                mean_constant_relative_increase=m['train_mean_mse']/m['mse']-1,
                mean_constant_output_delta_mse=m['train_mean_output_delta_mse'])
    result=dict(main=summary,preprocessing_subset32=pre_summary,verification=verify(),
        source_sha256={name:file_hash(out/name) for name in ('local.json','official.json','preprocessing.json')},
        scope='Paired bootstrap conditions on fixed checkpoints and reused test170; 7 video groups, no unseen-task claim. Preprocessing subset uses a separate train64 mean.')
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    names=['query','nodes','pre_norm','mu','decoder_projection','adaLN_modulation_layer_0']
    labels=['Query','Nodes','VAE LN','mu','Action proj.','AdaLN 0']
    for label,color in [('local','#cb6635'),('official','#2377a4')]:
        stages=loaded[label]['splits']['train']['stages']
        axes[0].plot(labels,[stages[n]['variance_fraction'] for n in names],marker='o',label=label,color=color)
    axes[0].set_yscale('log');axes[0].set_ylabel('Between-example variance / total energy')
    axes[0].set_title('Frozen representations on train512');axes[0].legend();axes[0].grid(alpha=.2)
    x=np.arange(2);width=.32
    axes[1].bar(x-width/2,[summary[k]['normal_mse'] for k in ('local','official')],width,label='Original mu',color='#2377a4')
    axes[1].bar(x+width/2,[summary[k]['mean_constant_mse'] for k in ('local','official')],width,label='Train-mean constant',color='#cb6635')
    axes[1].set_xticks(x,['Local final10000','Official released']);axes[1].set_ylabel('Feature MSE (lower is better)')
    axes[1].set_title('Same padded inputs, reused test170');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(main={k:{n:v for n,v in d.items() if n not in ('train_posterior','test_posterior','intervention_comparisons')} for k,d in summary.items()},preprocessing=pre_summary,verification=result['verification']),indent=2))


if __name__=='__main__':main()
