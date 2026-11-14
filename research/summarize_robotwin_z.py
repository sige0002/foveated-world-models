"""Collect the matched frozen RoboTwin diagnostics and plot their results."""
import argparse
import json
from pathlib import Path
import shutil
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plot-deps', type=Path, help='Optional standalone matplotlib dependency directory')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    ev = a.run / 'evaluations'
    names = {'local': 'step10000-z-utility', 'official': 'official-release-visual-matched-20261114'}
    summary = {'run': str(a.run), 'models': {}}
    for kind, name in names.items():
        r = json.loads((ev / name / 'results.json').read_text())
        m = json.loads((ev / 'z-mechanism-20261114' / kind / 'results.json').read_text())
        assert r['complete'] and m['complete']
        summary['models'][kind] = {'protocol': r['protocol'], 'groups': r['groups'],
            'control_minus_real': r['control_minus_real'], 'z_geometry': r['z_geometry'],
            'mechanism': {k: m[k] for k in ['samples', 'posterior', 'layers', 'decoded', 'oracle', 'scope']}}
    summary['oracle_controls'] = json.loads((ev / 'z-mechanism-20261114/oracle-controls.json').read_text())
    summary['precision_check'] = json.loads((ev / 'z-mechanism-20261114/precision-check.json').read_text())
    summary['precision_notes'] = ('The first diagnostic FP32 mode disables autocast but inherits medium matmul precision from VJEPA_LAM. '
                                  'The separate precision_check explicitly uses highest FP32 encoder and FP64 posterior head.')
    summary['loss_routes'] = json.loads((ev / 'loss-route-audit-20261114/results.json').read_text())
    summary['limitations'] = [
        'Same 512 validation pairs and donors; episode-cluster CIs condition on the fixed checkpoint and sampled cases.',
        'Official training membership and actual training recipe are unknown. State heads have incompatible dimensions and are not compared.',
        'Mechanistic diagnostics use first 128 of the 512 pairs; oracle uses first 16 and sees evaluated targets.',
        'No dataset-scaling or controlled from-scratch training experiment was run here.',
        'Layer energy ratios are coordinate-dependent geometry, not measurements of mutual information.',
        'The measured forward mechanism does not establish which historical optimizer/data/loss setting caused it.'
    ]
    (a.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

    import yaml
    requested = yaml.safe_load((a.run / 'requested_config.yaml').read_text())
    sys.path[:0] = requested['dependency_roots']
    if a.plot_deps:
        sys.path.insert(0, str(a.plot_deps))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    colors = {'local': '#0072B2', 'official': '#D55E00'}
    modes = ['real', 'zero', 'train_mean', 'other_episode', 'current_only', 'persistence']
    xpos = np.arange(len(modes))
    for i, kind in enumerate(names):
        values = summary['models'][kind]
        axes[0].bar(xpos + (i - .5) * .36,
                    [values['groups']['all']['metrics'][m]['visual_mse'] for m in modes],
                    .36, color=colors[kind], label='Local 10k' if kind == 'local' else 'Official release')
        layer_keys = ['layer_0', 'layer_3', 'layer_7', 'layer_11', 'layer_15', 'layer_19', 'layer_23', 'nodes', 'pre_norm', 'mu']
        axes[1].plot(range(len(layer_keys)),
                     [values['mechanism']['layers'][k]['real']['variance_fraction'] for k in layer_keys],
                     'o-', color=colors[kind], label='Local 10k' if kind == 'local' else 'Official release')
    axes[0].set_xticks(xpos, ['Real z', 'Zero', 'Train mean', 'Other z', 'Current-only z', 'Copy current'], rotation=28, ha='right')
    axes[0].set_ylabel('Future feature MSE (lower is better)')
    axes[0].set_title('Same 512 image pairs / frozen models')
    axes[0].set_ylim(0, .57)
    axes[0].legend(frameon=False)
    axes[1].set_xticks(range(len(layer_keys)), ['L1', 'L4', 'L8', 'L12', 'L16', 'L20', 'L24', 'Project', 'Norm', 'Mean z'], rotation=32, ha='right')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Between-pair variance / total energy')
    axes[1].set_title('Representation geometry / same 128 pairs')
    axes[1].grid(axis='y', alpha=.2)
    fig.savefig(a.output / 'comparison.png', dpi=180)
    fig.savefig(a.output / 'comparison.svg')
    plt.close(fig)
    for name in ['diagnose_robotwin_z.py', 'probe_robotwin_oracle_controls.py', 'check_robotwin_loss_routes.py',
                 'check_robotwin_head_precision.py', 'summarize_robotwin_z.py']:
        src = Path(__file__).with_name(name)
        shutil.copy2(src, ev / 'z-mechanism-20261114' / name)
    print(json.dumps({'output': str(a.output), 'complete': True}))


if __name__ == '__main__':
    main()
