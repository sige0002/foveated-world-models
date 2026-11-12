"""Read-only integrity checks for the prespecified 96-episode research cache."""
import hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 torch.set_num_threads(4)
 base=ROOT/'runs/research-unused-96-prior6501'
 m=json.loads((base/'cache_manifest.json').read_text())
 assert 'decoder_precision_alignment' in m
 assert sha(base/'validation.pt')==m['cache_sha256']['validation']
 d=torch.load(base/'validation.pt',weights_only=True)
 selection=json.loads(Path(m['selection_path']).read_text())
 assert sha(m['selection_path'])==m['selection_sha256']
 assert len(d['rows'])==96 and len({r['episode'] for r in d['rows']})==96
 for a,b in zip(d['rows'],selection['selected']):
  for key in ('episode','index','dt','endpoint_indices'):assert a[key]==b[key]
 assert not {r['episode'] for r in d['rows']} & set(map(str,selection['excluded_episode_ids']))
 for key in ('current','future','codes','baseline','original_codes'):
  assert len(d[key])==96 and torch.isfinite(d[key]).all()
 assert sha(m['head_checkpoint'])==m['head_checkpoint_sha256']
 assert sha(ROOT/'research/idm/reference-query-full/train_calibration.pt')==m['calibration_sha256']
 for path,digest in m['source_cache_sha256'].items():assert sha(path)==digest
 output=dict(cache_sha256=m['cache_sha256']['validation'],count=96,all_checks_passed=True,
  tensor_shapes={k:list(d[k].shape) for k in ('current','future','codes','baseline','original_codes')},
  precision=m['precision'],diagnostics=m['decoder_precision_alignment']['diagnostics'],verification_script_sha256=sha(__file__))
 print(json.dumps(output,indent=2))
if __name__=='__main__':main()
