"""Check posterior conclusions under local padding vs official FR3 crop/resize.

64 calibration train pairs and 32 reused test pairs chosen uniformly by index,
without output-based selection. Uses the same views/timestamps in both pipelines.
"""
import json
import os
from pathlib import Path
import sys

os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LAM_ENABLE_MANUAL_WANDB='0')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
import av
import numpy as np
import torch
from torchvision.transforms import v2
from latent_action_model.core.vjepa_encoder import DINOv3Encoder
from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug
from vendor.dataset.robomind import RoboMINDDataset,read_frame
from research.investigate_native_posterior import load_data,make_model,encode,evaluate,posterior_stats,CHECKPOINT,file_hash


def raw_frame(container,stream,target):
    container.seek(int(round(target/float(stream.time_base))),stream=stream,backward=True)
    for frame in container.decode(stream):
        if frame.pts is None:continue
        t=float(frame.pts*stream.time_base)
        if abs(t-target)<1e-5:
            return torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2,0,1)
        if t>target+1e-5:break
    raise RuntimeError(f'Missing frame {target}')


def official_resize(frames):
    # Exact VideoMaxAspectCrop.crop_to_max_aspect formula and VideoResize defaults
    # at official c12168d. No stochastic transforms in this evaluation.
    h,w=frames.shape[-2:]
    if w/h>4/3:
        width=max(1,int(np.floor(h*4/3)));left=(w-width)//2
        frames=frames[...,left:left+width].contiguous()
    return v2.Resize((256,256),interpolation=v2.InterpolationMode.BILINEAR,antialias=True)(frames)


@torch.no_grad()
def main():
    out=ROOT/'research/results/native-posterior-investigation-20261113/preprocessing.json'
    if out.exists():raise FileExistsError(out)
    device='cuda:1';torch.cuda.set_device(device);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
    original,hashes=load_data(); selected={}
    for split,count in [('train',64),('reused_test170',32)]:
        d=original[split];idx=np.linspace(0,len(d['rows'])-1,count,dtype=int).tolist()
        selected[split]=dict(indices=idx,rows=[d['rows'][i] for i in idx],
            current=d['current'][idx].clone(),future=d['future'][idx].clone())
    del original
    plan=json.loads((ROOT/'research/results/difference-idm-ablation-20261113/test-selection.json').read_text())
    datasets={s:RoboMINDDataset(plan['manifest_paths'][key],**plan['options'])
              for s,key in [('train','train'),('reused_test170','test')]}
    for key,path in plan['manifest_paths'].items():assert file_hash(path)==plan['manifest_sha256'][key]
    model=DINOv3Encoder('/data2/yuki_sadasue/models/dinov3-vitb16-pretrain-lvd1689m',
        enable_norm=True,norm_layer_type='ln').eval().to(device)
    saved=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')['state_dict']
    prefix='lam.vision_encoder.'
    model.load_state_dict({k[len(prefix):]:v for k,v in saved.items() if k.startswith(prefix)},strict=True)
    del saved
    def features(frames):
        v1,v2=gpu_two_view_video_aug(frames.unsqueeze(0).to(device),output_size=(256,256),training=False,dual_view_aug=True)
        f=model.encode(torch.cat((v1,v2),1),n=-2)
        return f[:,2:3].cpu(),f[:,-1:].cpu()
    cropped={};checks={}
    for split,d in selected.items():
        ds=datasets[split];current=[];future=[];check=0.
        for j,row in enumerate(d['rows']):
            i,a,b,tail=ds.pairs[row['index']];source=ds.rows[i];arrays=ds._arrays(i)
            assert str(source['episode_id'])==str(row['episode'])
            assert abs(float(arrays['timestamps'][b]-arrays['timestamps'][a])-row['dt'])<1e-6
            with av.open(source['video_file']) as container:
                stream=container.streams.video[0];stream.codec_context.thread_count=1
                frames=torch.stack([raw_frame(container,stream,float(arrays['video_pts'][k])) for k in (a,b)])
                if j==0:
                    padded=torch.stack([read_frame(container,stream,float(arrays['video_pts'][k]),(256,256)) for k in (a,b)])
                    c,f=features(padded)
                    check=max(float((c-d['current'][j:j+1]).abs().max()),float((f-d['future'][j:j+1]).abs().max()))
                    assert check<1e-5,check
            c,f=features(official_resize(frames));current.append(c);future.append(f)
            if (j+1)%16==0:print('crop features',split,j+1,flush=True)
        cropped[split]=dict(current=torch.cat(current),future=torch.cat(future),rows=d['rows'])
        checks[split]=check
    del model;torch.cuda.empty_cache()
    results={}
    for label in ('local','official'):
        modules,meta=make_model(label,device);results[label]=dict(model=meta,preprocessing={})
        for name,data in [('local_padding',selected),('official_crop_resize',cropped)]:
            train=encode(modules,data['train']['current'],data['train']['future'],device)
            test=encode(modules,data['reused_test170']['current'],data['reused_test170']['future'],device)
            ev=evaluate(modules,data['reused_test170'],test,train,device,samples=2)
            results[label]['preprocessing'][name]=dict(evaluation=ev,
                posterior=posterior_stats(test['mu'],test['logvar'],modules['vq'].beta))
            print(label,name,ev['metrics'],flush=True)
        del modules;torch.cuda.empty_cache()
    report=dict(results=results,cache_sha256=hashes,padded_cache_max_abs=checks,
        selection={s:dict(indices=d['indices'],rows=d['rows']) for s,d in selected.items()},
        script_sha256=file_hash(Path(__file__)),core_script_sha256=file_hash(ROOT/'research/investigate_native_posterior.py'),
        scope='Uniform 64 train/32 reused test subset; train-only calibration; same overhead views and 1.2s timing. Official FR3 crop/resize only, not its data mixture/view distribution. No matched alternative-action ground truth.')
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('COMPLETE preprocessing',flush=True)


if __name__=='__main__':main()
