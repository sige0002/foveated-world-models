"""Durable two-GPU queue; finish all fixed training before diagnostic scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'research/results/z-utility-matched-20261113');a=p.parse_args()
    a.runs.mkdir(parents=True,exist_ok=True);jobs=queue.Queue()
    for seed in (42,43,44):
        for arm in ('external-real','direct-constant','external-constant'):jobs.put((arm,seed))
    lock=threading.Lock();failed=threading.Event();status={'started_unix':time.time(),'jobs':{},'state':'training'}
    def record(name,obj):
        with lock:
            status['jobs'][name]=obj
            tmp=a.runs/'queue-status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(a.runs/'queue-status.json')
    def worker(device):
        while not failed.is_set():
            try:arm,seed=jobs.get_nowait()
            except queue.Empty:return
            name=f'{arm}-{seed}';out=a.runs/name
            if (out/'results.json').exists():
                r=json.loads((out/'results.json').read_text())
                if r['complete'] and not r['smoke_only']:record(name,{'state':'already_complete'});continue
            cmd=[sys.executable,str(ROOT/'research/z_utility_matched.py'),'--arm',arm,'--seed',str(seed),'--device',device,'--output',str(out)]
            if (out/'latest.pt').exists():cmd.append('--resume')
            with (a.runs/f'{name}.log').open('a') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                record(name,{'state':'running','pid':proc.pid,'device':device,'started_unix':time.time(),'command':cmd})
                code=proc.wait()
            record(name,{'state':'complete' if code==0 else 'failed','returncode':code,'finished_unix':time.time()})
            if code:failed.set()
    with ThreadPoolExecutor(max_workers=2) as executor:
        for f in [executor.submit(worker,d) for d in ('cuda:0','cuda:1')]:f.result()
    if failed.is_set():status['state']='failed';record('summary',{'state':'not_started'});return 1
    status['state']='evaluating';record('summary',{'state':'running'})
    # Disjoint output ownership; no two workers evaluate the same arm.
    def evaluate_worker(device,arms):
        with (a.runs/f'evaluation-{device.replace(":","-")}.log').open('a') as log:
            return subprocess.run([sys.executable,str(ROOT/'research/evaluate_z_utility_matched.py'),
                '--runs',str(a.runs),'--output',str(a.output),'--device',device,'--worker-arms',*arms],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(evaluate_worker,'cuda:0',['direct-real','external-constant']),
                 executor.submit(evaluate_worker,'cuda:1',['external-real','direct-constant'])]
        codes=[f.result() for f in futures]
    if any(codes):
        status['state']='failed';record('summary',{'state':'failed','worker_returncodes':codes});return 1
    with (a.runs/'summary.log').open('a') as log:
        code=subprocess.run([sys.executable,str(ROOT/'research/evaluate_z_utility_matched.py'),'--runs',str(a.runs),'--output',str(a.output)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
    status['state']='complete' if code==0 else 'failed';record('summary',{'state':status['state'],'returncode':code,'finished_unix':time.time()})
    return code


if __name__=='__main__':sys.exit(main())
