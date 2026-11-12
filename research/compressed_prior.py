"""Current-only prior students and fixed-correction evaluation utilities."""
import json,sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.train_cached_decoders import make_decoder,file_hash


class CurrentPrior(nn.Module):
    def __init__(self,seed=42):
        super().__init__();torch.manual_seed(seed)
        self.core=make_decoder('prior_residual',128,2,768,32,16)
    def forward(self,current):
        zero=current.new_zeros((len(current),1,32))
        return self.core(current,zero,baseline=current)


class LinearPrior(nn.Module):
    def __init__(self,mean,delta_mean,weight):
        super().__init__()
        for name,value in [('mean',mean),('delta_mean',delta_mean),('weight',weight)]:
            self.register_buffer(name,value.float())
    def forward(self,current):
        return current+(current-self.mean)@self.weight+self.delta_mean


@torch.no_grad()
def fit_linear(current,teacher,ridge=.01):
    # Fixed train-only fit; the future-feature target and IDM code are not arguments.
    d=current.shape[-1];n=current.numel()//d
    sx=torch.zeros(d,dtype=torch.float64);sy=sx.clone()
    xx=torch.zeros(d,d,dtype=torch.float64);xy=xx.clone()
    for x,y in zip(current.split(32),teacher.split(32)):
        x=x.reshape(-1,d).double();y=y.reshape(-1,d).double()-x
        sx+=x.sum(0);sy+=y.sum(0);xx+=x.T@x;xy+=x.T@y
    mx,my=sx/n,sy/n
    covariance=xx/n-mx[:,None]*mx[None,:]
    cross=xy/n-mx[:,None]*my[None,:]
    regularized=covariance+ridge*torch.eye(d,dtype=torch.float64)
    weight=torch.linalg.solve(regularized,cross)
    residual=float((regularized@weight-cross).abs().max())
    return LinearPrior(mx,my,weight),dict(patches=n,ridge=ridge,normal_equation_max_abs=residual)


def load_inputs(source):
    base=source/'runs/reference-idm42-nested512-prior10000'
    paths={'train':base/'train.pt','dev64':base/'validation.pt','observed96':source/'runs/reference-idm42-observed96-prior10000/validation.pt'}
    data={};hashes={};manifests={}
    for split,path in paths.items():
        m=json.loads((path.parent/'cache_manifest.json').read_text());manifests[split]=m
        hashes[str(path)]=file_hash(path);assert hashes[str(path)]==m['cache_sha256'][path.stem]
        data[split]=torch.load(path,weights_only=True,mmap=True)
    episodes=[{r['episode'] for r in d['rows']} for d in data.values()]
    assert all(episodes[i].isdisjoint(episodes[j]) for i in range(3) for j in range(i+1,3))
    assert [len(d['rows']) for d in data.values()]==[512,64,96]
    for m in manifests.values():
        for key in ('prior_checkpoint_sha256','prior_weights_sha256'):assert m[key]==manifests['train'][key]
    path=source/'runs/reference-idm42-nested512-comparison400-20261113/prior_residual-seed42.pt'
    hashes[str(path)]=file_hash(path);saved=torch.load(path,weights_only=True)
    assert (saved['steps'],saved['seed'],saved['variant'])==(400,42,'prior_residual')
    correction=make_decoder('prior_residual',saved['dim'],saved['layers'],768,32,16)
    correction.load_state_dict(saved['state_dict'],strict=True);correction.eval().requires_grad_(False)
    return data,correction,hashes


def objective(student,correction,kind,current,teacher,future,code):
    predicted=student(current)
    if kind=='prior_teacher':return (predicted-teacher).square().mean()
    if kind=='future':return (predicted-future).square().mean()
    if kind!='downstream_teacher':raise ValueError(kind)
    with torch.no_grad():target=correction(current,code,baseline=teacher)
    # Keep autograd to the student baseline through the frozen correction.
    output=correction(current,code,baseline=predicted)
    return (output-target).square().mean()+.1*(predicted-teacher).square().mean()


@torch.inference_mode()
def evaluate(student,correction,data):
    student.eval();rows=[]
    for start in range(0,len(data['rows']),8):
        h,b,y,z=[data[k][start:start+8] for k in ('current','baseline','future','codes')]
        g=student(h);pred=correction(h,z,baseline=g);reference=correction(h,z,baseline=b)
        values={k:(a-t).square().flatten(1).mean(1) for k,a,t in [
            ('prior_teacher_mse',g,b),('prior_future_mse',g,y),('final_future_mse',pred,y),
            ('final_teacher_mse',pred,reference),('reference_final_mse',reference,y),('reference_prior_mse',b,y)]}
        rows += [dict(data['rows'][start+i],**{k:float(v[i]) for k,v in values.items()}) for i in range(len(h))]
    return dict(metrics={k:sum(r[k] for r in rows)/len(rows) for k in values},rows=rows)
