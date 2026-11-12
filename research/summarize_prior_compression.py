"""Compare current-prior approximation with its downstream prediction error."""
import json,hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    launch=json.loads((ROOT/'research/results/prior-compression512-launch-20261113.json').read_text())
    linear_path=ROOT/'runs/linear-prior512-20261113/results.json'
    linear=json.loads(linear_path.read_text())
    assert sha(linear_path.parent/'prior.pt')==linear['checkpoint_sha256']
    models={'linear':linear};pending=[];sources={str(linear_path.relative_to(ROOT)):sha(linear_path)}
    initial=None;batch=None
    for arm in launch['neural_runs']:
        run=ROOT/arm['output'];path=run/'results.json'
        if not path.exists():pending.append(arm['kind']);continue
        r=json.loads(path.read_text());p=json.loads((run/'protocol.json').read_text())
        assert r['protocol_sha256']==sha(run/'protocol.json') and r['checkpoint_sha256']==sha(run/'prior.pt')
        assert p['source_sha256']==linear['source_sha256'] and p['kind']==arm['kind'] and p['steps']==400
        assert p['implementation_sha256']==launch['source_sha256']
        assert r['frozen_correction_unchanged']
        if initial is None:initial=p['initial_student_sha256'];batch=r['batch_sha256']
        assert initial==p['initial_student_sha256'] and batch==r['batch_sha256']
        models[arm['kind']]=r;sources[str(path.relative_to(ROOT))]=sha(path)
    draws=np.random.default_rng(20261113).integers(0,96,(20000,96))
    output={}
    identity=lambda rows:[(r['episode'],r['index'],r['dt'],r.get('endpoint_indices')) for r in rows]
    for name,r in models.items():
        evaluation=r['evaluation'];rows=evaluation['observed96']['rows']
        assert identity(rows)==identity(linear['evaluation']['observed96']['rows'])
        errors=np.array([v['final_future_mse']-v['reference_final_mse'] for v in rows])
        metrics=evaluation['observed96']['metrics']
        output[name]={'metrics':{s:v['metrics'] for s,v in evaluation.items()},
                      'final_difference':float(errors.mean()),
                      'final_relative_change_percent':float(100*errors.mean()/metrics['reference_final_mse']),
                      'conditional_paired_ci95':np.quantile(errors[draws].mean(1),[.025,.975]).tolist(),
                      'downstream_to_prior_distortion_ratio':metrics['final_teacher_mse']/metrics['prior_teacher_mse'],
                      'derived_error_cross_term':metrics['final_future_mse']-metrics['reference_final_mse']-metrics['final_teacher_mse']}
    result={'complete':not pending,'pending':pending,'source_sha256':sources,'models':output,
            'cross_term_definition':'Derived 2 E[(teacher_final-Y) dot (student_final-teacher_final)] per feature, from squared-error identity. Uses true future for diagnosis only.',
            'scope':'All fixed endpoints, common current-only inputs and frozen correction; single neural seed, reused64/96. Conditional bootstrap excludes model-selection/training uncertainty. Ratio is observed nonlinear error propagation, not a global Lipschitz constant or causal proof.'}
    path=ROOT/'research/results/prior-compression512-summary-20261113.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'pending':pending,'models':{k:v['metrics']['observed96'] for k,v in output.items()}},indent=2))


if __name__=='__main__':main()
