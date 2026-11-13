"""Create and fully verify a Stage1-only RoboTwin state-quaternion derivative.

Uses the pinned QuaterNet qfix function. Original data/action columns are untouched.
CPU preprocessing, not model training. Output is published only after all checks.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def flat(column):
    arr=column.combine_chunks()
    assert arr.null_count==0
    lengths=np.diff(arr.offsets.to_numpy())
    assert (lengths==16).all()
    return arr.values.to_numpy().reshape(-1,16)


def replace_vectors(table,name,values):
    col=table[name].combine_chunks()
    new=pa.ListArray.from_arrays(col.offsets,pa.array(values.reshape(-1),type=col.type.value_type))
    return table.set_column(table.schema.get_field_index(name),table.schema.field(name),new)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--qfix-source',type=Path,required=True);a=p.parse_args()
    pa.set_cpu_count(2);pa.set_io_thread_count(2)
    a.source=a.source.resolve();a.output=a.output.absolute()
    if a.output.exists(): raise ValueError('Output must be new')
    if a.output.is_relative_to(a.source): raise ValueError('Output cannot be inside source')
    stage=a.output.with_name(a.output.name+'.partial')
    if stage.exists(): raise ValueError('Partial output exists; inspect before retry')
    stage.mkdir(parents=True)
    started=time.monotonic()
    def progress(msg): print(json.dumps({'stage':msg,'seconds':round(time.monotonic()-started,2)}),flush=True)
    tree=ast.parse(a.qfix_source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='qfix')
    scope={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(a.qfix_source),'exec'),scope);qfix=scope['qfix']
    info=json.loads((a.source/'meta/info.json').read_text());n=info['total_frames']
    paths=sorted((a.source/'data').glob('*/*.parquet'))
    source_hashes={str(p.relative_to(a.source)):sha(p) for p in paths}
    source_meta={str(p.relative_to(a.source)):sha(p) for p in (a.source/'meta').rglob('*') if p.is_file()}
    state=np.empty((n,16),np.float32);ep=np.empty(n,np.int64);frame=np.empty(n,np.int64);offset=0
    lengths=[]
    for path in paths:
        t=pq.read_table(path,columns=['observation.state','episode_index','frame_index'],use_threads=False);end=offset+len(t)
        state[offset:end]=flat(t['observation.state']);ep[offset:end]=t['episode_index'].to_numpy();frame[offset:end]=t['frame_index'].to_numpy();lengths.append(len(t));offset=end
    assert offset==n and np.isfinite(state).all()
    cuts=np.r_[0,np.flatnonzero(ep[1:]!=ep[:-1])+1,n];starts=cuts[:-1]
    assert len(starts)==info['total_episodes'] and len(np.unique(ep[starts]))==len(starts)
    assert np.array_equal(frame,np.arange(n)-np.repeat(starts,np.diff(cuts)))
    qi=np.array([3,4,5,6,11,12,13,14]);keep=np.array([0,1,2,7,8,9,10,15])
    fixed=state.copy();episode_stats={};episode_counts=[]
    for start,end in zip(cuts[:-1],cuts[1:]):
        old=state[start:end][:,qi].reshape(-1,2,4)
        new=qfix(old)
        assert np.array_equal(old[0],new[0])
        fixed[start:end,qi]=new.reshape(-1,8)
        flips=np.any(new!=old,axis=-1).sum(0)
        episode_counts.append([int(ep[start]),int(end-start),*map(int,flips)])
        x=new.reshape(-1,8).astype(np.float64)
        episode_stats[int(ep[start])]={'min':x.min(0),'max':x.max(0),'mean':x.mean(0),'std':x.std(0)}
    assert np.array_equal(state[:,keep],fixed[:,keep])
    assert np.isfinite(fixed).all()
    progress('qfix applied within all episodes')
    adjacent=np.flatnonzero(ep[:-1]==ep[1:]);report={'source':str(a.source),'output':str(a.output),'frames':n,'episodes':len(starts),'scope':'Stage1 observation.state quaternion only; actions and other state coordinates unchanged',
      'qfix_upstream':'facebookresearch/QuaterNet@9d8485b732b0a44b99b6cf4b12d3915703507ddc',
      'qfix_source_sha256':sha(a.qfix_source),'script_sha256':sha(__file__),'arms':{},'source_sha256':source_hashes,'source_meta_sha256':source_meta}
    for arm,off in [('left',3),('right',11)]:
        q=state[:,off:off+4].astype(np.float64);f=fixed[:,off:off+4].astype(np.float64)
        unchanged=np.all(q==f,axis=1);flipped=np.all(q==-f,axis=1)
        assert (unchanged|flipped).all()
        before=(q[adjacent]*q[adjacent+1]).sum(1);after=(f[adjacent]*f[adjacent+1]).sum(1)
        assert (after>=0).all() and np.array_equal(np.abs(before),np.abs(after))
        norm=np.linalg.norm(q,axis=1);assert np.array_equal(norm,np.linalg.norm(f,axis=1))
        report['arms'][arm]={'flipped_frames':int((~unchanged).sum()),'adjacent_negative_dot_before':int((before<0).sum()),'adjacent_negative_dot_after':int((after<0).sum()),
          'norm_max_error':float(np.max(np.abs(norm-1))),'rotation_preserved_all_frames':True,'first_frame_preserved_all_episodes':True,
          'adjacent_quat_l2_before':np.quantile(np.linalg.norm(q[adjacent+1]-q[adjacent],axis=1),[.5,.95,.99,1]).tolist(),
          'adjacent_quat_l2_after':np.quantile(np.linalg.norm(f[adjacent+1]-f[adjacent],axis=1),[.5,.95,.99,1]).tolist()}
        del q,f,before,after,norm
    (stage/'qfix').mkdir()
    np.savetxt(stage/'qfix/episode-flips.csv',np.array(episode_counts),fmt='%d',delimiter=',',header='episode,frames,left_flipped,right_flipped',comments='')
    # Share immutable images; copy metadata so new statistics never overwrite originals.
    shutil.copytree(a.source/'meta',stage/'meta')
    (stage/'videos').symlink_to(a.source/'videos',target_is_directory=True)
    offset=0;output_hashes={}
    for path,length in zip(paths,lengths):
        raw=pq.read_table(path,use_threads=False);new=replace_vectors(raw,'observation.state',fixed[offset:offset+length]);dest=stage/path.relative_to(a.source);dest.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(new,dest,compression='zstd')
        reread=pq.read_table(dest,use_threads=False)
        assert reread.schema.equals(raw.schema,check_metadata=True)
        assert np.array_equal(flat(reread['observation.state']),fixed[offset:offset+length])
        for name in raw.column_names:
            if name!='observation.state': assert reread[name].equals(raw[name]),name
        output_hashes[str(path.relative_to(a.source))]=sha(dest);offset+=length
        progress('wrote and reread '+str(path.relative_to(a.source)))
    # Update only changed quaternion coordinates in all provided state statistics.
    global_stats={}
    for metric in ['min','max','mean','std','q01','q99']:
        values=[]
        for k in qi:
            x=fixed[:,k].astype(np.float64)
            values.append(float(np.quantile(x,.01 if metric=='q01' else .99)) if metric.startswith('q') else float(getattr(x,metric)()))
        global_stats[metric]=values
    for name in ['stats.json','stats_gr00t.json']:
        path=stage/'meta'/name;stats=json.loads(path.read_text())
        for metric,vals in global_stats.items():
            if metric in stats['observation.state']:
                for k,v in zip(qi,vals): stats['observation.state'][metric][int(k)]=v
        path.write_text(json.dumps(stats,indent=2)+'\n')
    for path in sorted((stage/'meta/episodes').rglob('*.parquet')):
        t=pq.read_table(path,use_threads=False);ids=t['episode_index'].to_pylist()
        for metric in ['min','max','mean','std']:
            name='stats/observation.state/'+metric
            vals=t[name].to_pylist()
            for i,eid in enumerate(ids):
                for k,v in zip(qi,episode_stats[eid][metric]): vals[i][int(k)]=float(v)
            t=t.set_column(t.schema.get_field_index(name),t.schema.field(name),pa.array(vals,type=t[name].type))
        pq.write_table(t,path,compression='zstd')
        reread=pq.read_table(path,use_threads=False);assert reread.equals(t)
    progress('updated global and all episode quaternion statistics')
    # Independent audit of the exact official horizon after qfix.
    stats=json.loads((stage/'meta/stats_gr00t.json').read_text())['observation.state'];trans=fixed.copy()
    for k in keep:
        lo,hi=np.float32(stats['min'][k]),np.float32(stats['max'][k]);trans[:,k]=2*(fixed[:,k]-lo)/(hi-lo)-1 if hi!=lo else 0
    trans[:,[7,15]]*=-1
    future=np.minimum(np.arange(n)+48,np.repeat(cuts[1:]-1,np.diff(cuts)))
    assert np.array_equal(ep,ep[future]);coord={};any1=np.zeros(n,bool);any2=np.zeros(n,bool)
    names=['left_x','left_y','left_z','left_qw','left_qx','left_qy','left_qz','left_gripper','right_x','right_y','right_z','right_qw','right_qx','right_qy','right_qz','right_gripper']
    for k,name in enumerate(names):
        delta=(trans[future,k]-trans[:,k]).astype(np.float64);absd=np.abs(delta);any1|=absd>1;any2|=absd>2
        coord[name]={'abs_max':float(absd.max()),'abs_gt1_count':int((absd>1).sum()),'abs_gt2_count':int((absd>2).sum()),'tanh_l2_floor':float(np.mean(np.maximum(absd-1,0)**2)),'two_tanh_l2_floor':float(np.mean(np.maximum(absd-2,0)**2))}
    report['horizon']={'frame_stride':48,'no_episode_crossing':True,'coordinates':coord,'any_abs_gt1_count':int(any1.sum()),'any_abs_gt2_count':int(any2.sum()),'tanh_state_l2_floor':float(np.mean([v['tanh_l2_floor'] for v in coord.values()])),'two_tanh_state_l2_floor':float(np.mean([v['two_tanh_l2_floor'] for v in coord.values()]))}
    assert not any2.any()
    # Original hashes are rechecked after conversion, including every metadata file.
    assert all(sha(a.source/p)==h for p,h in source_hashes.items())
    assert all(sha(a.source/p)==h for p,h in source_meta.items())
    report.update({'status':'passed','original_data_and_metadata_unchanged':True,'non_quaternion_columns_unchanged':True,'output_data_sha256':output_hashes,'elapsed_seconds':time.monotonic()-started})
    (stage/'qfix/audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (stage/'qfix/preprocess_robotwin_qfix.py').write_text(Path(__file__).read_text())
    os.rename(stage,a.output)
    progress('PASSED and published '+str(a.output))
    print(json.dumps({k:v for k,v in report.items() if k in ['arms','status','elapsed_seconds']},indent=2),flush=True)

if __name__=='__main__':main()
