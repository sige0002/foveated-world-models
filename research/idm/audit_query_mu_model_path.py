"""One final original-Model replay, no fitting or precision search."""
import sys,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.train_reference_heads import Model,representation
from research.idm.probe_displacement_ridge import sha256
@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.set_float32_matmul_precision('medium')
 paths=dict(query=ROOT/'research/idm/observed96-reference-queries2900/validation.pt',calibration=ROOT/'research/idm/reference-query-full/train_calibration.pt',cache=ROOT/'runs/research-unused-96-prior6501/validation.pt')
 e=json.loads((ROOT/'research/idm/reference-codes-seed42/cache_manifest.json').read_text())['export'];paths['head']=Path(e['head_checkpoint']);assert sha256(paths['head'])==e['head_checkpoint_sha256'] and sha256(paths['calibration'])==e['calibration_sha256'];saved=torch.load(paths['head'],weights_only=True);model=Model().eval().requires_grad_(False);model.load_state_dict(saved['state_dict'],strict=True)
 q=torch.load(paths['query'],weights_only=True);c=torch.load(paths['calibration'],weights_only=True);d=torch.load(paths['cache'],weights_only=True);assert q['rows']==d['rows'];x=representation(q,c,'reference');mu=torch.cat([model.posterior(x[i:i+64])[0] for i in range(0,96,64)])
 report=dict(model_class='research.idm.train_reference_heads.Model',precision=torch.get_float32_matmul_precision(),threads=4,inference_mode=torch.is_inference_mode_enabled(),batches=[64,32],max_error=float((mu-d['codes']).abs().max()),mse=float((mu-d['codes']).square().mean()),exact=torch.equal(mu,d['codes']),source_sha256={str(p):sha256(p) for p in paths.values()},script_sha256=sha256(Path(__file__)),scope='Single final original-Model replay; cache unchanged; no fitting or model selection')
 (ROOT/'research/idm/query-accessibility512-model-replay.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 torch.set_num_threads(2)
if __name__=='__main__':main()
