"""Count dense linear/attention MACs; this is not a latency benchmark."""
import json
from pathlib import Path
import sys
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.train_cached_decoders import make_decoder


@torch.inference_mode()
def main():
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    path=ROOT/'runs/reference-idm42-nested512-comparison400-20261113/prior_residual-seed42.pt'
    saved=torch.load(path,weights_only=True)
    model=make_decoder('prior_residual',saved['dim'],saved['layers'],768,32,16).eval()
    model.load_state_dict(saved['state_dict'],strict=True)
    data=torch.load(ROOT/'runs/reference-idm42-nested512-prior10000/train.pt',weights_only=True,mmap=True)
    records=[];handles=[]
    def linear(name):
        def hook(module,args,output):
            rows=args[0].numel()//module.in_features
            records.append(dict(name=name,kind='linear',macs=rows*module.in_features*module.out_features))
        return hook
    def attention(name):
        def hook(module,args,output):
            q,k,v=args[:3];b,n,d=q.shape;m=k.shape[1]
            assert d==module.embed_dim and module._qkv_same_embed_dim
            projection=b*(n+2*m+n)*d*d
            interaction=2*b*n*m*d
            records.append(dict(name=name,kind='attention_projection',macs=projection))
            records.append(dict(name=name,kind='attention_interaction',macs=interaction))
        return hook
    for name,module in model.named_modules():
        if isinstance(module,nn.MultiheadAttention):handles.append(module.register_forward_hook(attention(name)))
        elif isinstance(module,nn.Linear):handles.append(module.register_forward_hook(linear(name)))
    prediction=model(data['current'][:1],data['codes'][:1],baseline=data['baseline'][:1])
    for h in handles:h.remove()
    assert all(not r['name'].endswith('attn.out_proj') for r in records), 'Attention out projection counted twice'
    total=sum(r['macs'] for r in records)
    groups={}
    for r in records:
        group='head_block' if r['name'].startswith('head_block.') else 'narrow_blocks' if r['name'].startswith('blocks.') else 'projections'
        groups[group]=groups.get(group,0)+r['macs']
    report=dict(batch=1,tokens=256,features=768,trainable_parameters=sum(p.numel() for p in model.parameters()),
                total_dense_macs=total,groups=groups,records=records,
                quadratic_attention_fraction=sum(r['macs'] for r in records if r['kind']=='attention_interaction')/total,
                finite_prediction=bool(torch.isfinite(prediction).all()),
                scope='Analytical dense MACs verified from executed module shapes. Excludes biases, normalization, softmax, nonlinearity, memory traffic, frozen prior and vision encoder. Not measured wall-clock speed; do not infer GPU latency from MACs.')
    p=ROOT/'research/results/decoder-dense-macs-20261113.json'
    with p.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print({k:v for k,v in report.items() if k!='records'})


if __name__=='__main__':main()
