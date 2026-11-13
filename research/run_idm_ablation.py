"""Durable two-GPU queue for six fixed runs and terminal-only test evaluation."""
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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1'])
    p.add_argument('--output',type=Path,default=ROOT/'research/results/difference-idm-ablation-20261113')
    p.add_argument('--test-cache',type=Path,default=ROOT/'runs/difference-idm-ablation-inputs-20261113/test.pt')
    a=p.parse_args()
    if len(set(a.devices))!=len(a.devices):p.error('Devices must be distinct')
    if not a.test_cache.exists():p.error('Prepare the held-out feature cache first')
    a.runs.mkdir(parents=True,exist_ok=True)
    jobs=queue.Queue()
    for seed in (42,43,44):
        for arm in ('difference','raw'):jobs.put((arm,seed))
    lock=threading.Lock();failed=threading.Event()
    status={'started_unix':time.time(),'jobs':{},'state':'training','summary_returncode':None}
    def record(name,value):
        with lock:
            status['jobs'][name]=value
            tmp=a.runs/'queue-status.tmp'
            tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(a.runs/'queue-status.json')
    def worker(device):
        while not failed.is_set():
            try:arm,seed=jobs.get_nowait()
            except queue.Empty:return
            name=f'{arm}-{seed}';out=a.runs/name
            if (out/'results.json').exists():
                r=json.loads((out/'results.json').read_text())
                if r.get('complete') and not r.get('smoke_only'):
                    record(name,{'state':'already_complete','device':device});continue
            cmd=[sys.executable,str(ROOT/'research/train_idm_ablation.py'),'--representation',arm,
                 '--seed',str(seed),'--device',device,'--output',str(out)]
            if (out/'latest.pt').exists():cmd.append('--resume')
            log=a.runs/f'{name}.log';started=time.time()
            with log.open('a') as stream:
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                record(name,{'state':'running','device':device,'pid':proc.pid,'started_unix':started,'command':cmd})
                code=proc.wait()
            record(name,{'state':'complete' if code==0 else 'failed','device':device,'returncode':code,
                         'started_unix':started,'finished_unix':time.time()})
            if code:failed.set()
    with ThreadPoolExecutor(max_workers=len(a.devices)) as executor:
        futures=[executor.submit(worker,device) for device in a.devices]
        for f in futures:f.result()
    if failed.is_set():
        status['state']='failed';record('summary',{'state':'not_started_due_to_training_failure'});return 1
    status['state']='terminal_evaluation';record('summary',{'state':'running'})
    cmd=[sys.executable,str(ROOT/'research/summarize_idm_ablation.py'),'--runs',str(a.runs),
         '--device',a.devices[0],'--output',str(a.output),'--test-cache',str(a.test_cache)]
    with (a.runs/'summary.log').open('a') as stream:
        code=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
    status['summary_returncode']=code;status['state']='complete' if code==0 else 'failed'
    record('summary',{'state':status['state'],'finished_unix':time.time(),'returncode':code})
    return code


if __name__=='__main__':sys.exit(main())
