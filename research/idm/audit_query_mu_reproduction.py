"""Head-only numerical audit of immutable observed96 cached Q and mu."""
import json,sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.train_reference_heads import representation
from research.idm.probe_displacement_ridge import sha256
@torch.inference_mode()
def main():
 paths=dict(manifest=ROOT/'research/idm/reference-codes-seed42/cache_manifest.json',query=ROOT/'research/idm/observed96-reference-queries2900/validation.pt',calibration=ROOT/'research/idm/reference-query-full/train_calibration.pt',cache=ROOT/'runs/research-unused-96-prior6501/validation.pt')
 m=json.loads(paths['manifest'].read_text())['export'];paths['head']=Path(m['head_checkpoint']);assert sha256(paths['head'])==m['head_checkpoint_sha256'];assert sha256(paths['calibration'])==m['calibration_sha256']
 qm=json.loads((paths['query'].parent/'manifest.json').read_text());assert sha256(paths['query'])==qm['query_sha256'];assert sha256(paths['cache'])==qm['source_cache_sha256']
 ck=torch.load(paths['head'],weights_only=True);h=nn.Sequential(nn.Linear(1024,256),nn.GELU(),nn.Linear(256,64)).eval();h.load_state_dict({k[5:]:v for k,v in ck['state_dict'].items() if k.startswith('head.')})
 q=torch.load(paths['query'],weights_only=True);c=torch.load(paths['calibration'],weights_only=True);d=torch.load(paths['cache'],weights_only=True);assert q['rows']==d['rows'];z=d['codes'];r={}
 for threads in [2,4]:
  torch.set_num_threads(threads)
  for precision in ['medium','highest']:
   torch.set_float32_matmul_precision(precision);x=representation(q,c,'reference')
   for b in [1,32,64,96]:
    pred=torch.cat([h(v).chunk(2,-1)[0] for v in x.split(b)]);r[f'T{threads}_{precision}_B{b}']=dict(maxerror=float((pred-z).abs().max()),mse=float((pred-z).square().mean()),exact=torch.equal(pred,z))
 torch.set_num_threads(2)
 report=dict(source_sha256={str(p):sha256(p) for p in paths.values()},script_sha256=sha256(Path(__file__)),results=r,scope='Head-only numerical reproduction; no cache mutation or evaluation-score selection')
 (ROOT/'research/idm/query-accessibility512-head-reproduction-cpu-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
