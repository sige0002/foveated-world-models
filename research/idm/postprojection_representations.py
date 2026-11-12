"""Train-calibrated 34D controls for a fresh head; no learned LN affine."""
import torch
VARIANTS=('direction','direction_statistics','calibrated_raw')
def decompose(p,epsilon=1e-5):
 if epsilon<=0:raise ValueError('epsilon must be positive')
 m=p.mean(-1,keepdim=True);s=((p-m).square().mean(-1,keepdim=True)+epsilon).sqrt()
 return (p-m)/s,m,s.log()
def fit_calibration(train_p,epsilon=1e-5,floor=1e-3):
 if train_p.ndim!=3 or train_p.shape[-1]!=32 or len(train_p)<2:raise ValueError('Expected N x queries x 32 training projections')
 if floor<=0 or not torch.isfinite(train_p).all():raise ValueError('Invalid training projections/floor')
 _,m,logs=decompose(train_p,epsilon);stats=torch.cat((m,logs),-1)
 def moments(x):return x.mean(0,keepdim=True),x.std(0,unbiased=False,keepdim=True).clamp_min(floor)
 pm,ps=moments(train_p);sm,ss=moments(stats)
 return dict(epsilon=epsilon,floor=floor,raw_mean=pm,raw_scale=ps,statistics_mean=sm,statistics_scale=ss,fit_count=len(train_p),fit_split='train')
def representation(p,calibration,variant):
 if variant not in VARIANTS or p.ndim!=3 or p.shape[-1]!=32:raise ValueError('Unknown variant or invalid shape')
 u,m,logs=decompose(p,calibration['epsilon']);zeros=p.new_zeros((*p.shape[:-1],2))
 if variant=='direction':return torch.cat((u,zeros),-1)
 if variant=='calibrated_raw':return torch.cat(((p-calibration['raw_mean'])/calibration['raw_scale'],zeros),-1)
 stats=(torch.cat((m,logs),-1)-calibration['statistics_mean'])/calibration['statistics_scale']
 return torch.cat((u,stats),-1)
