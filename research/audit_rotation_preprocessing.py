"""Reproduce rotation sign/boundary probes using pinned preprocessing functions.

CPU diagnostic only. Does not change data or training labels. --sources points
at rotation-survey-sources on SSD1; see its manifest.json for upstream revisions.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def extracted_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    # Execute exactly the pinned function body, without unrelated framework imports.
    lerobot = extracted_function(a.sources/'huggingface/lerobot/src/lerobot/processor/env_processor.py',
                                 '_quat2axisangle', {'torch': torch})
    peract = extracted_function(a.sources/'peract/peract/helpers/utils.py',
                                'quaternion_to_discrete_euler', {'np': np, 'Rotation': Rotation})
    path = a.sources/'real-stanford/diffusion_policy/diffusion_policy/model/common/rotation_transformer.py'
    spec = importlib.util.spec_from_file_location('pinned_rotation_transformer', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tf = mod.RotationTransformer('quaternion', 'rotation_6d')
    audit = json.loads(a.audit.read_text())
    report = {'runtime_scope': 'DP RotationTransformer and extracted LeRobot/PerAct functions; GR00T/OpenVLA source review only', 'real_events': {}}
    for arm in ['left','right']:
        event = audit['arms'][arm]['sign_flip_examples_most_negative_dot'][0]
        q = np.array([event['quaternion_wxyz_a'],event['quaternion_wxyz_b']],np.float64)
        q /= np.linalg.norm(q,axis=1,keepdims=True)
        xyzw = q[:,[1,2,3,0]]
        aa = lerobot(None, torch.tensor(xyzw)).numpy()
        r6 = tf.forward(q)
        dot = np.dot(q[0],q[1])
        report['real_events'][arm] = dict(episode=event['episode'],frames=[event['frame_a'],event['frame_b']],
            qw_both_positive=bool((q[:,0]>0).all()),so3_degrees=event['so3_angle_degrees'],
            raw_quat_l2=float(np.linalg.norm(q[1]-q[0])),
            temporal_sign_aligned_quat_l2=float(np.linalg.norm(q[1]*np.sign(dot)-q[0])),
            rotation6d_l2=float(np.linalg.norm(r6[1]-r6[0])),
            lerobot_axisangle_vector_l2_radians=float(np.linalg.norm(aa[1]-aa[0])),
            peract_discrete_euler_5deg=peract(xyzw,5).tolist())
    xyzw = Rotation.from_euler('z',60,degrees=True).as_quat()
    aa = lerobot(None,torch.tensor(np.stack([xyzw,-xyzw]))).numpy()
    q = np.stack([xyzw,-xyzw])[:,[3,0,1,2]]
    r6 = tf.forward(q)
    report['same_pose_opposite_sign_60deg_z'] = dict(lerobot_vectors=aa.tolist(),
        lerobot_vector_distance=float(np.linalg.norm(aa[1]-aa[0])),
        rotation6d_distance=float(np.linalg.norm(r6[1]-r6[0])),
        peract_labels=peract(np.stack([xyzw,-xyzw]),5).tolist())
    # Show that canonical qw>=0 does not guarantee temporal continuity at pi.
    q = Rotation.from_euler('z',[179.9,180.1],degrees=True).as_quat()[:,[3,0,1,2]]
    q[q[:,0]<0]*=-1
    report['pi_crossing'] = dict(rotation_degrees=.2,canonical_quat_l2=float(np.linalg.norm(q[1]-q[0])),
        rotation6d_l2=float(np.linalg.norm(tf.forward(q)[1]-tf.forward(q)[0])))
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
