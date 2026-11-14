"""Compare raw Euler differences against relative SO(3) rotation on source states."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
import numpy as np
from vendor.dataset.lerobot_robomind import LeRobotSource,LeRobotRoboMINDDataset,minmax


def matrix(rpy,reverse=False):
    r,p,y=rpy
    cr,sr=np.cos(r),np.sin(r);cp,sp=np.cos(p),np.sin(p);cy,sy=np.cos(y),np.sin(y)
    rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return rx@ry@rz if reverse else rz@ry@rx


def quaternion(rpy):
    r,p,y=rpy/2
    cr,sr=np.cos(r),np.sin(r);cp,sp=np.cos(p),np.sin(p);cy,sy=np.cos(y),np.sin(y)
    return np.array([cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy])


def relative(a,b,reverse=False):
    d=matrix(a,reverse).T@matrix(b,reverse)
    return float(np.degrees(np.arccos(np.clip((np.trace(d)-1)/2,-1,1))))


def main():
    source=LeRobotSource('/media/sadasue/SSD1/foveated-world-models/datasets/RoboMIND-lerobot/franka_fr3_dual')
    assert abs(relative(np.radians([179,0,0]),np.radians([-179,0,0]))-2)<1e-9
    assert relative(np.radians([180,0,0]),np.radians([-180,0,0]))<1e-5
    report={'convention':'R=Rz(yaw)Ry(pitch)Rx(roll), matching existing states_from_rpy. Source field names specify RPY but hardware convention is not independently verified. Opposite composition RxRyRz is reported as sensitivity check.',
        'method':'acos((trace(R_current.T @ R_future)-1)/2), checked independently using unit-quaternion dot product.',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'splits':{}}
    maxdiff=0.
    for split in ('train','validation'):
        ds=LeRobotRoboMINDDataset(source,split);rows=[];bounds=[]
        for idx,(ep,i,j,tail) in enumerate(ds.pairs):
            data=ds._arrays(ep);states=data['states'][[i,j]].astype(np.float64)
            normalized=minmax(states,source.minimum,source.maximum)
            for arm,offset in [('left',0),('right',7)]:
                a,b=states[:,offset+3:offset+6];delta=b-a
                angle=relative(a,b)
                dot=abs(float(quaternion(a)@quaternion(b)))
                qa=float(np.degrees(2*np.arccos(np.clip(dot,-1,1))))
                maxdiff=max(maxdiff,abs(angle-qa))
                if not np.any(abs(delta)>np.pi):continue
                row={'pair_index':idx,'episode':int(ds.rows[ep]['episode_index']),'arm':arm,
                     'first_index':i,'last_index':j,'timestamps_seconds':data['timestamps'][[i,j]].tolist(),
                     'dt':float(data['timestamps'][j]-data['timestamps'][i]),'tail':bool(tail),
                     'current_rpy_degrees':np.degrees(a).tolist(),'future_rpy_degrees':np.degrees(b).tolist(),
                     'raw_delta_degrees':np.degrees(delta).tolist(),
                     'wrapped_component_delta_degrees':np.degrees((delta+np.pi)%(2*np.pi)-np.pi).tolist(),
                     'normalized_rpy_delta':(normalized[1,offset+3:offset+6]-normalized[0,offset+3:offset+6]).tolist(),
                     'relative_rotation_degrees':angle,'opposite_order_rotation_degrees':relative(a,b,True)}
                bounds.append(row)
        eligible=[r for r in bounds if not r['tail'] and r['dt']>=1.19]
        examples=[min(eligible,key=lambda r:abs(r['relative_rotation_degrees']-target)) for target in (1,5,20)]
        report['splits'][split]={'pairs':len(ds),'boundary_arm_transitions':len(bounds),
            'boundary_pairs':len({r['pair_index'] for r in bounds}),
            'boundary_pairs_with_crossing_arm_rotation_under_degrees':{str(t):len({r['pair_index'] for r in bounds if r['relative_rotation_degrees']<t}) for t in (1,5,10)},
            'boundary_rotation_degrees_quantiles':dict(zip(['min','p25','median','p75','max'],np.quantile([r['relative_rotation_degrees'] for r in bounds],[0,.25,.5,.75,1]).tolist())),
            'example_selection':'Full-horizon crossing-arm transitions nearest to 1,5,20 degrees of relative rotation',
            'examples':examples,'boundary_rows':bounds}
    report['matrix_vs_quaternion_max_difference_degrees']=maxdiff
    assert maxdiff<1e-4,maxdiff
    p=ROOT/'research/results/rpy-relative-rotation-audit-20261113.json'
    p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({s:{k:v for k,v in d.items() if k!='boundary_rows'} for s,d in report['splits'].items()},indent=2))
    print('matrix/quaternion max difference degrees:',maxdiff)


if __name__=='__main__':main()
