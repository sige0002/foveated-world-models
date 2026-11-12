"""Wait for verified extraction, then run smoke and full matched comparisons."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
cache=ROOT/'runs/frozen-idm-step2900';queries=ROOT/'research/idm/reference-query-full'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4')
start=time.monotonic()
while not (queries/'manifest.json').exists():
    if time.monotonic()-start>3600:raise TimeoutError('Query extraction did not complete in one hour')
    time.sleep(10)
for name,steps,seeds in [('reference-smoke400',400,[42]),('reference-comparison2000',2000,[42,43,44])]:
    output=ROOT/'research/idm'/name
    command=[sys.executable,str(ROOT/'research/idm/train_reference_heads.py'),'--cache',str(cache),'--queries',str(queries),
             '--output',str(output),'--steps',str(steps),'--seeds',*map(str,seeds)]
    print('Starting',name,flush=True)
    subprocess.run(command,cwd=ROOT,env=env,check=True)
    report=json.loads((output/'comparison.json').read_text())
    assert len(report['results'])==4*len(seeds)
    for seed in seeds:
        records=[r for r in report['results'] if r['seed']==seed]
        for key in ['initial_weight_sha256','minibatch_sha256','noise_sha256','parameters']:
            assert len({r[key] for r in records})==1,(seed,key)
        constant=next(r for r in records if r['variant']=='constant')
        assert constant['validation']['metrics']['swap_prediction_change']<1e-12
    print('Validated',name,flush=True)
