"""Run full corrected Stage-1 training, then fixed final-checkpoint diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train_stage1 import load_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path,help='Resume an interrupted run including optimizer state')
    a=p.parse_args();out=a.output.resolve()
    if out.exists() and not a.resume:p.error('Output must be new unless --resume is supplied')
    if a.resume and not a.resume.is_file():p.error('Resume checkpoint is missing')
    out.mkdir(parents=True,exist_ok=True)
    cfg=load_config(a.config)
    assert cfg['data']['format']=='lerobot_v3' and cfg['trainer']['max_steps']==10000
    tracked=['train_stage1.py','research/evaluate_corrected_stage1.py',
             'vendor/dataset/lerobot_robomind.py','research/run_corrected_stage1.py']
    protocol={'started_utc':datetime.now(timezone.utc).isoformat(),
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in tracked},
              'resolved_config':cfg,'fresh_training':not bool(a.resume),
              'evaluation':'Final10000 on all corrected validation/test pairs; native posterior mean, zero and cyclic permutation; MSE, SmoothL1, MAE, cosine, high/low feature-change regions. No old-cache comparison.'}
    launch_name=('resume-launch-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json') if a.resume else 'launch.json'
    (out/launch_name).write_text(json.dumps(protocol,indent=2)+'\n')
    commands=[('training',[sys.executable,'-u',str(ROOT/'train_stage1.py'),'--config',str(a.config.resolve()),'--output',str(out)]),
              ('evaluation',[sys.executable,'-u',str(ROOT/'research/evaluate_corrected_stage1.py'),
                             '--run',str(out),'--output',str(out/'revalidation.json')])]
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    if a.resume:commands[0][1].extend(['--resume',str(a.resume.resolve())])
    for stage,cmd in commands:
        (out/'pipeline_status.json').write_text(json.dumps({'stage':stage,'status':'running','command':cmd})+'\n')
        with (out/(stage+'.log')).open('a' if a.resume else 'w') as log:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            (out/'pipeline_status.json').write_text(json.dumps({'stage':stage,'status':'failed','returncode':result.returncode})+'\n')
            raise SystemExit(result.returncode)
    (out/'pipeline_status.json').write_text(json.dumps({'status':'complete','training_steps':10000})+'\n')


if __name__=='__main__':main()
