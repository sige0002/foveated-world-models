import argparse,json
from pathlib import Path
import torch
from research.compressed_prior import load_inputs,fit_linear,evaluate,file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    data,correction,hashes=load_inputs(a.source)
    model,fit=fit_linear(data['train']['current'],data['train']['baseline'])
    a.output.mkdir(parents=True)
    checkpoint=a.output/'prior.pt';torch.save(model.state_dict(),checkpoint)
    report=dict(source_sha256=hashes,fit=fit,script_sha256=file_hash(Path(__file__)),
                checkpoint_sha256=file_hash(checkpoint),
                evaluation={s:evaluate(model,correction,d) for s,d in data.items()},
                scope='Train-only linear teacher-prior fit. Future/code not student inputs. Frozen correction; posterior reconstruction for final metric. Reused64/96, no independent test.')
    (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print({s:v['metrics'] for s,v in report['evaluation'].items()},flush=True)


if __name__=='__main__':main()
