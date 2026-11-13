"""Audit selected endpoint pairs against source state trajectories and decoded video."""
import hashlib
import json
from pathlib import Path
import sys

import av
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vendor.dataset.robomind import RoboMINDDataset, read_frame


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def motion(ds, index):
    i, a, b, tail = ds.pairs[index]
    entry = ds.rows[i]
    arrays = ds._arrays(i)
    state = arrays['states'][a:b+1].astype(np.float64)
    endpoint, path = [], []
    for j in (0, 10):
        xyz = state[:, j:j+3]
        endpoint.append(float(np.linalg.norm(xyz[-1] - xyz[0])))
        path.append(float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()))
    return {'index': index, 'episode': str(entry['episode_id']), 'endpoint_indices': [a, b],
            'dt': float(arrays['timestamps'][b]-arrays['timestamps'][a]), 'tail': bool(tail),
            'endpoint_translation_m': endpoint, 'path_translation_m': path,
            'max_arm_endpoint_m': max(endpoint), 'max_arm_path_m': max(path),
            'gripper_endpoint_abs_source_units': np.abs(state[-1, [9, 19]]-state[0, [9, 19]]).tolist(),
            'rotation6d_endpoint_max_abs': float(np.abs(state[-1, [3,4,5,6,7,8,13,14,15,16,17,18]]-state[0, [3,4,5,6,7,8,13,14,15,16,17,18]]).max()),
            'all_interval_state_observed': bool(arrays['original_valid'][a:b+1,1:].all()),
            'endpoints_observed': bool(arrays['original_valid'][[a,b]].all()),
            'manifest_row': i}


def summarize(rows):
    out = {'count': len(rows)}
    for key in ('dt', 'max_arm_endpoint_m', 'max_arm_path_m'):
        x = np.array([r[key] for r in rows])
        out[key+'_quantiles_0_25_50_75_100'] = np.quantile(x,[0,.25,.5,.75,1]).tolist()
    for limit in (.001,.01,.05):
        out[f'both_arms_endpoint_lt_{limit}_m'] = sum(r['max_arm_endpoint_m'] < limit for r in rows)
        out[f'both_arms_path_lt_{limit}_m'] = sum(r['max_arm_path_m'] < limit for r in rows)
    out['all_interval_state_observed'] = sum(r['all_interval_state_observed'] for r in rows)
    out['all_endpoint_states_and_frames_observed'] = sum(r['endpoints_observed'] for r in rows)
    return out


def main():
    torch.set_num_threads(2)
    out = ROOT/'research/results/pair-motion-audit-20261113'
    if out.exists(): raise FileExistsError(out)
    out.mkdir()
    selection = json.loads((ROOT/'research/results/nested-train512-selection-20261113.json').read_text())
    paths = {'train': selection['manifest'], 'validation': selection['validation_manifest']}
    datasets = {s: RoboMINDDataset(p, **selection['options']) for s,p in paths.items()}
    report = {'source_sha256': {p:sha(p) for p in paths.values()}, 'script_sha256':sha(Path(__file__)),
              'scope':'Read-only source state/video audit. Translation thresholds are descriptive, not calibrated action labels. Path length can include measurement noise; intervals may contain bounded fills. No training or new test-set use.',
              'pools': {}, 'selected': {}, 'visual_examples': []}
    for split,ds in datasets.items():
        all_rows = [motion(ds,i) for i in range(len(ds))]
        report['pools'][split] = summarize(all_rows)
        print('pool',split,report['pools'][split],flush=True)
    audit = json.loads((ROOT/'research/idm/model-step10000-baseline-audit.json').read_text())
    for split,rel,disk in [('train','runs/reference-idm42-nested512-prior10000','train'),
                           ('validation','runs/reference-idm42-nested512-prior10000','validation'),
                           ('observed96','runs/reference-idm42-observed96-prior10000','validation')]:
        cache_path=ROOT/rel/(disk+'.pt');manifest=json.loads((ROOT/rel/'cache_manifest.json').read_text())
        digest=sha(cache_path);assert digest==manifest['cache_sha256'][disk]
        report['source_sha256'][str(cache_path)]=digest
        d=torch.load(cache_path,weights_only=True,mmap=True);ds=datasets['train' if split=='train' else 'validation']
        rows=[]
        for n,r in enumerate(d['rows']):
            m=motion(ds,r['index']);assert m['episode']==str(r['episode']) and abs(m['dt']-r['dt'])<1e-9
            if 'endpoint_indices' in r:assert m['endpoint_indices']==r['endpoint_indices']
            m['feature_copy_mse']=float((d['current'][n]-d['future'][n]).square().mean())
            m['features_exactly_equal']=bool(torch.equal(d['current'][n],d['future'][n]))
            if split!='train':
                ar=audit['results'][split]['rows'][n];assert ar['source_row']['index']==r['index'] and str(ar['source_row']['episode'])==m['episode']
                m.update(original_model_mse=ar['original_model_mse'],zero_model_mse=ar['zero_model_mse'])
            rows.append(m)
        summary=summarize(rows)
        summary['identical_feature_pairs']=sum(r['features_exactly_equal'] for r in rows)
        summary['feature_copy_mse']=float(np.mean([r['feature_copy_mse'] for r in rows]))
        strata={}
        if split!='train':
            for label,keep in [('endpoint_below_1cm',lambda r:r['max_arm_endpoint_m']<.01),('endpoint_at_least_1cm',lambda r:r['max_arm_endpoint_m']>=.01),('endpoint_at_least_5cm',lambda r:r['max_arm_endpoint_m']>=.05)]:
                subset=[r for r in rows if keep(r)]
                if subset:
                    z=float(np.mean([r['zero_model_mse'] for r in subset]));o=float(np.mean([r['original_model_mse'] for r in subset]))
                    strata[label]={'count':len(subset),'zero_mse':z,'original_mse':o,'relative_gain_percent':100*(z-o)/z}
        report['selected'][split]={'summary':summary,'motion_strata':strata,'rows':rows}
        print('selected',split,summary,strata,flush=True)
    # Select examples by state motion only, before reading their image pixels.
    rows=report['selected']['observed96']['rows'];order=np.argsort([r['max_arm_endpoint_m'] for r in rows])
    chosen=[int(order[0]),int(order[len(order)//2]),int(order[-1])]
    sheet=Image.new('RGB',(3*256,3*(256+40)), 'white');draw=ImageDraw.Draw(sheet)
    ds=datasets['validation']
    for j,n in enumerate(chosen):
        r=rows[n];i,a,b,_=ds.pairs[r['index']];entry=ds.rows[i];arr=ds._arrays(i);indices=[a,(a+b)//2,b]
        frames=[];pts=[]
        with av.open(entry['video_file']) as con:
            stream=con.streams.video[0];stream.codec_context.thread_count=1
            for k,t in enumerate(indices):
                target=float(arr['video_pts'][t]);frame=read_frame(con,stream,target,(256,256));frames.append(frame);pts.append(target)
                sheet.paste(Image.fromarray(frame.permute(1,2,0).numpy()),(k*256,j*296+40))
                draw.text((k*256+4,j*296+3),f"ep {r['episode']} / frame {t} / PTS {target:.3f}",fill='black')
                draw.text((k*256+4,j*296+18),f"endpoint motion {r['max_arm_endpoint_m']*100:.2f} cm",fill='black')
        sample=ds[r['index']]
        assert torch.equal(sample['frames'][0],frames[0]) and torch.equal(sample['frames'][1],frames[-1])
        assert pts[0] < pts[-1]
        delta=(frames[-1].float()-frames[0].float()).abs()
        report['visual_examples'].append({'source_row':r,'video_file':entry['video_file'],'task':entry.get('tasks'),
            'pts':pts,'dataset_frame_match_exact':True,'pixel_mean_abs_difference_0_255':float(delta.mean()),
            'fraction_pixels_any_channel_difference_gt10':float((delta.max(0).values>10).float().mean())})
    sheet.save(out/'examples.png')
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
