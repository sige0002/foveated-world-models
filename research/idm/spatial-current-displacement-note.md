# Reference-code utility survives a substantially better current-only probe

**The pooled-current baseline was weak, but it does not explain the entire observed reference-code benefit.** A spatial current kernel probe lowered current-only error substantially; reference mu still added measured-displacement prediction utility on both research-observed evaluation sets. This is a fixed-representation prediction result, not action identification or conditional mutual information.

## Locked design and kernel scale

The current input preserves all256×768 DINO patch coordinates, with fixed spatial ordering, plus current20D measured state. Prespecified candidates were a linear spatial kernel and RBF kernels with bandwidth multipliers .25,1,4 times the training median-distance scale. Code was zero32, original6501 mu, or reference42 mu, added through a linear code kernel. Every fold refit all feature/target scaling on its training episodes only. The exact earlier5episode folds were reused. Targets retain the .001m/.01gripper standard-deviation floors.

Each component kernel is training-centered and normalized to mean diagonal1 if nonzero. The current+code sum is then renormalized to training mean diagonal1. A zero code contributes zero rather than changing the total kernel amplitude; every group/fold has total strength1. Thus a mechanically weaker total kernel does not penalize the current-only group. The added code still changes the kernel geometry and relative component weighting, as intended.

Protocols were saved before fitting, and all train-CV choices were saved before evaluation predictions. The initial alpha grid was1e-4 through1e4. The two baselines selected its lower boundary; one separately recorded extension to1e-8 was then justified solely by that train-CV fact. Original results were retained. No further extension was performed.

## Results after the one-time lower-boundary audit

All groups selected bandwidth multiplier1 by train CV. Reference selected alpha.01; zero/original selected1e-8, again the lower boundary. This remaining boundary uncertainty is explicit.

| Code | Old pooled-linear dev64 | Spatial-RBF dev64 | Old pooled-linear observed96 | Spatial-RBF observed96 |
|---|---:|---:|---:|---:|
| zero | .943928 | .599409 | .866747 | .566081 |
| original6501 mu | .908416 | .585521 | .839108 | .531908 |
| reference42 mu | .732079 | **.489638** | .603342 | **.437926** |

These are normalized8D displacement MSEs, not visual world-model errors. The spatial current-only baseline is empirically much better than the pooled one; merely calling a model nonlinear was not used as evidence. Reference adds an18.31% reduction on dev64 and22.64% on observed96 relative to this selected current-only model. Seven of eight physical-coordinate MSEs improve in each set; **right gripper worsens**, from .100462 to .127678 on dev64 and .094619 to .112079 on observed96. Thus endpoint utility improves in aggregate, not uniformly across fine-control targets.

The paired episode bootstrap,10000 resamples/seed20261113, gives reference-minus-zero normalized-MSE differences:

| Set | Mean paired difference | Percentile95% interval |
|---|---:|---:|
| dev64 | −.109771 | [−.184548, −.043331] |
| observed96 | −.128155 | [−.177565, −.081640] |

These intervals are **conditional on the already trained representations, chosen hyperparameters and fitted probes**. They omit training/CV-selection uncertainty, multiplicity and adaptivity from prior use of these evaluation sets. They are exploratory conditional uncertainty summaries, not confirmatory significance claims on an independent test.

## Boundary and numerical audits

The initial selections at1e-4 produced zero .599417/.566081 and original .585533/.531914. Extending to1e-8 changed evaluation errors by at most1.20e-5; reference was unchanged. For the selected RBF current-only model, train-CV MSE changed from .61193553 at1e-4 to .61192934 at1e-8; the corresponding original-code values were .62177055 and .62176020. No claim is made that the lower-grid optimum has been bracketed.

Float32 spatial Gram construction could interact with small regularization. With choices frozen, a separate audit recomputed all current geometry/scalers in float64. The maximum evaluation-MSE change across the three models/two sets was4.27e-6; normalized prediction-change MSE was below7.5e-10. The conclusions were not selected or altered using those evaluations.

Training-centered kernels have a constant-vector nullspace. In the original geometry, the clipped negative eigenvalues were tiny (zero:−6.49e-16; reference:−3.02e-17); original6501 had none. At alpha1e-8 the full regularized condition numbers are about1.93e9/4.98e9 for zero/original, dominated by that nullspace, while positive-subspace condition numbers are about198/967. Reference atalpha.01 has full condition≈1240 and positive-subspace≈229. The near-zero threshold used in the numerical script is |eigenvalue|<1e-10. Clean observed float64 sensitivity reduces this numerical concern but does not remove finite-sample/statistical uncertainty. Training errors are nearly zero for the baseline kernels, so interpolation/overfitting must remain visible.

## What is and is not resolved

This result weakens the specific explanation that all extra utility came from discarding spatial current information. It does not rule out a stronger learned current representation, more training data, or a different nonlinear predictor explaining further gains. The reference posterior consumes true future information; observing its deterministic mu differs from decoding a single Gaussian sample. The probe cannot establish a camera-free/action-identifiable latent or an execution-time future predictor. Both64 and96 were already observed during research; neither is a new independent final test.

The linear spatial candidate and all RBF candidates were evaluated only after their settings were locked. Their outcomes are retained in full rather than choosing the lowest evaluation score: e.g. spatial-linear zero scored .556450 on observed96, slightly better than the train-CV-selected RBF zero .566081. We retain the prescribed CV selection, not an evaluation-set oracle.

## Artifacts and verification

- `spatial-current-displacement-probe/`: initial protocol, every CV score, selections, all per-example evaluation errors and fit artifacts.
- `spatial-current-boundary-audit/`: independent one-time lower-grid extension; old-grid CV scores reproduce and original files remain unchanged.
- `spatial-current-numerical-audit/`: frozen-selection float64 geometry sensitivity and spectra.
- `spatial-current-bootstrap/`: predeclared conditional-bootstrap protocol/results.
- `spatial-current-tests.txt`: four meaningful tests pass (kernel-strength equality, constant-code equivalence, train-only scale/bandwidth, eigen/direct-ridge agreement).
- `spatial-current-verification.json`: hashes, source/rows, train-only selections, all fold kernel strengths and saved-error arithmetic verified.

Scripts are `probe_spatial_current.py`, `probe_spatial_current_boundary.py`, `audit_spatial_probe_precision.py`, `bootstrap_spatial_probe.py` and `verify_spatial_probe.py`. All computation used CPU≤2 threads and existing feature/state artifacts; no GPU, image decoding or world-model training occurred.
