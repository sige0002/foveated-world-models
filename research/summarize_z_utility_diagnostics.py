"""Summarize measured response energy, native controls and fixed data strata."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
import numpy as np
from research.summarize_idm_ablation import paired_comparison
from research.train_early_condition import file_hash


def response(metrics):
    return {name:{'mse':metrics[name+'_mse'],
                   'mse_change_percent':100*(metrics[name+'_mse']/metrics['mse']-1),
                   'output_delta_mse':metrics[name+'_output_delta_mse'],
                   'output_delta_rms_over_true_change_rms':math.sqrt(metrics[name+'_output_delta_mse']/metrics['copy_mse'])}
            for name in ('null','mean','zero','swapped','neighbor')}


def strata(evaluations,split,key):
    rr=[e[split]['rows'] for e in evaluations]
    groups=sorted({key(r) for r in rr[0]});out={}
    for group in groups:
        selected=[[r for r in rows if key(r)==group] for rows in rr]
        count=len(selected[0]);assert all(len(rows)==count for rows in selected)
        m={name:float(np.mean([[r[name] for r in rows] for rows in selected]))
           for name in ('mse','null_mse','mean_mse','copy_mse','mean_output_delta_mse')}
        m['mean_code_mse_change_percent']=100*(m['mean_mse']/m['mse']-1)
        out[group]={'count':count,'metrics':m}
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'research/results/z-utility-matched-20261113');a=p.parse_args()
    report=json.loads((a.output/'results.json').read_text());assert report['complete']
    output={'source_sha256':file_hash(a.output/'results.json'),'script_sha256':file_hash(Path(__file__)),
            'scope':'Descriptive diagnostics on reused data. Output changes are feature-space distances, not RGB or causal action effects.',
            'native':{},'learned':{}}
    native=report['original_native']['evaluation']
    for split,e in native.items():
        rows=e['rows'];real=np.array([[r['mse'] for r in rows]]);groups=[r['group'] for r in rows]
        output['native'][split]={'metrics':e['metrics'],'response':response(e['metrics']),'geometry':e['code_geometry'],
            'variance_fraction':e['code_geometry']['variance_sum']/e['code_geometry']['mean_squared_norm'],
            'controls':{name:paired_comparison(np.array([[r[name+'_mse'] for r in rows]]),real,groups)
                        for name in ('mean','zero','swapped')},'dt':e['dt']}
    for arm in ('direct-real','external-real','direct-constant','external-constant'):
        ev=[json.loads((a.output/f'{arm}-{seed}.json').read_text())['evaluation'] for seed in (42,43,44)]
        output['learned'][arm]={}
        for split,v in report['splits'].items():
            m=v['models'][arm]['metrics']
            output['learned'][arm][split]={'response':response(m),
                'by_change':strata(ev,split,lambda r:str(r['motion_bin'])),
                'by_dt':strata(ev,split,lambda r:'full_1.2s' if r['source_row']['dt']>=1.15 else 'shortened'),
                'by_video_group':strata(ev,split,lambda r:r['group'])}
    (a.output/'diagnostic-summary.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print('Diagnostic summary saved')


if __name__=='__main__':main()
