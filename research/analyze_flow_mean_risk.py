"""Unbiased finite-MC estimate of fixed flow conditional-mean squared risk."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    paths=[ROOT/'research/results/final-prior-flow-observed96-20261113.json',ROOT/'research/results/reference-idm42-final-prior-observed96-20261113.json']
    flow,direct=[json.loads(p.read_text())['results'][0] for p in paths]
    assert flow['variant']=='prior_flow' and flow['seed']==direct['seed']==42
    assert [(r['episode'],r['index']) for r in flow['rows']]==[(r['episode'],r['index']) for r in direct['rows']]
    sample=np.array([r['mse'] for r in flow['rows']]); mean4=np.array([r['mean_mse'] for r in flow['rows']]); directrisk=np.array([r['mse'] for r in direct['rows']])
    # N independent draws: E[mean sample loss]=B+V; E[mean-prediction loss]=B+V/N.
    n=4; variance=n/(n-1)*(sample-mean4); risk=(n*mean4-sample)/(n-1)
    assert np.all(variance>-1e-12)
    assert np.allclose(risk+variance,sample) and np.allclose(risk+variance/n,mean4)
    difference=risk-directrisk;rng=np.random.default_rng(20261113)
    ci=np.quantile(difference[rng.integers(0,len(risk),(20000,len(risk)))].mean(1),[.025,.975])
    result=dict(sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),samples_per_episode=n,
        sample_risk=float(sample.mean()),mean4_risk=float(mean4.mean()),
        estimated_predictive_variance=float(variance.mean()),
        estimated_infinite_mc_mean_risk=float(risk.mean()),direct_risk=float(directrisk.mean()),
        mean_risk_minus_direct=float(difference.mean()),paired_episode_ci95=ci.tolist(),
        formula='B_hat=(N*loss(mean_N)-mean_N(loss(sample)))/(N-1); V_hat=N/(N-1)*(mean_sample_loss-loss_mean_prediction)',
        assumptions='Four iid generation-noise draws conditional on fixed current/code/model; all dimensions averaged identically. Current/source/posterior code fixed per episode. Same saved flow model and solver.',
        scope='Descriptive reuse of already observed96. B_hat is an unbiased Monte-Carlo estimator of this fixed flow model conditional-mean risk, not an actual infinite-ensemble evaluation or guarantee. Individual estimates are not clipped. Episode bootstrap conditions on fitted models; no training/selection uncertainty or extra noise replications.',
        interpretation='Sampling variance alone does not explain the observed gap to direct prediction if this conditional-mean risk estimate is representative. Does not establish failure of all flow architectures or greater-data training.',
        rows=[dict(episode=r['episode'],index=r['index'],estimated_mean_risk=float(b),estimated_variance=float(v),direct_mse=float(d)) for r,b,v,d in zip(flow['rows'],risk,variance,directrisk)])
    out=ROOT/'research/results/final-prior-flow-mean-risk-estimate-20261113.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','sources')},indent=2))
if __name__=='__main__':main()
