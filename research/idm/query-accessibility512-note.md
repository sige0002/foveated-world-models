# Frozen query versus learned μ: train512 accessibility probe

The full 1024D query difference predicts measured displacement better overall than the learned 32D posterior mean with the specified readout. It **does not repair right-gripper prediction**. Thus this experiment supports investigating what the learned representation/readout emphasizes, but does not support the narrower explanation that right-gripper failure is caused by its 32D bottleneck.

All results use already observed dev64 and observed96. No new images, labels, IDM training, or world-model training were used. The measured target is the endpoint difference of prepared state indices `[0,1,2,9,10,11,12,19]`; xyz are meters in source arm frames, while gripper values use the source dataset's `[0,1]` convention with physical units unverified. They are not action commands or joint angles.

## Fixed experiment

[Protocol](query-accessibility512/protocol.json) was saved before fitting; [all train-CV choices](query-accessibility512/train_selection.json) were saved before evaluation. Train512 folds, targets, current/state features, four spatial-current kernels, and the once-extended alpha grid exactly match the preceding [512 probe](nested512-displacement-note.md). Each current kernel and linear code kernel is train-centered and normalized to mean training diagonal one; their sum is normalized to one as well. Thus 1024 coordinates do not automatically increase total kernel strength, although kernel rank/geometry and effective statistical capacity still differ.

The three code inputs are saved μ32, raw frozen `Q(x,y)-Q(x,x)`1024, and PCA32 of that difference. PCA uses float64 SVD of train-fold coordinate-standardized queries, with no labels or heldout samples in the PCA fit. The usual subsequent train-only score standardization effectively whitens retained PCA coordinates. Full512 PCA is fitted only after selection. PCA32 retains 76.65% of standardized query variance on full512; that fraction is not an estimate of action information. The original head/calibration are fixed and were trained/calibrated on128, not512.

Every group selected spatial RBF multiplier1 and ridge alpha0.01, strictly inside the common grid. CV normalized MSE is μ32 0.202925, query1024 0.171656, PCA32 0.186373. The rerun μ group reproduces all preceding512 CV candidate scores and selected train/dev/96 MSE within `1e-12`.

| Input added to spatial current | dev64 common-scale MSE | observed96 common-scale MSE | dev right-gripper MSE | observed96 right-gripper MSE |
|---|---:|---:|---:|---:|
| No code, previous512 control | 0.420013 | 0.402629 | 0.085939 | 0.076725 |
| Learned μ32 | 0.321819 | 0.316632 | 0.089649 | 0.081496 |
| Query difference1024 | 0.291697 | 0.286871 | 0.089932 | 0.082426 |
| Train-fold PCA32 | 0.315757 | 0.306865 | 0.091530 | 0.082419 |

Common-scale MSE uses exactly the old128 target standard deviations/floors, so values remain comparable across training sizes. Results additionally preserve own512 normalization and all physical-coordinate/per-row errors. Query1024 reduces aggregate MSE relative to μ by 9.36%/9.40%; PCA32 by 1.88%/3.08%. The former advantage is largely in xyz coordinates. Right-gripper MSE worsens slightly in both sets; none of these code representations beats the no-code right-gripper control.

## Large positive right-gripper subgroup

The old128 cutoffs remain static `|delta|<=0.01` and large `|delta|>0.7268722057342529`; no thresholds were chosen from evaluation labels. The fixed large-positive subgroup contains7 dev and9 observed96 examples.

| Representation | dev large-positive MSE | dev predicted mean | observed96 large-positive MSE | observed96 predicted mean |
|---|---:|---:|---:|---:|
| μ32 | 0.606409 | 0.205956 | 0.544411 | 0.238065 |
| Query1024 | 0.619971 | 0.203320 | 0.522753 | 0.251031 |
| PCA32 | 0.610660 | 0.205344 | 0.540727 | 0.240962 |

True means are0.977974/0.959373. Every representation still underpredicts all large-positive magnitudes by more than0.01. The modest observed96 raw-query improvement is not reproduced on dev; whole-coordinate errors worsen. This is not evidence of recovering the missing gripper signal. Small subgroup counts and descriptive reuse of evaluation data constrain interpretation.

## Interpretation and next boundary

The specified linear code readout accesses more overall displacement utility from1024D queries than from learned μ32. PCA32's small aggregate improvement means output dimension alone is not a complete explanation. However, head nonlinearity, training objective, standardized kernel geometry, and finite-sample regularization differ. These are not information-theoretic tests: a better nonlinear readout of μ could close the gap, and future-dependent query information could include camera/appearance signals correlated with displacement.

For the original right-gripper mechanism question, the result is negative: bypassing the μ head did not repair it. Remaining explanations include query-level representation, rare/multimodal gripper transitions, insufficient predictive context, and readout/target weighting. This experiment does not distinguish those explanations or justify automatically enlarging z. Any follow-up should preregister its training-only comparison rather than select a new model using these observed labels.

## Reproduction and numerical caveat

New script: [probe_query_accessibility512.py](probe_query_accessibility512.py). Run from repository root:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/probe_query_accessibility512.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m unittest research.idm.test_query_accessibility512 -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/verify_query_accessibility512.py
```

The extraction/probe script intentionally refuses an existing output directory. [Verification](query-accessibility512-verification.json) passes19 immutable-source hashes, exact rows/folds, train-only calibration artifacts, PCA orthonormality, equal kernel strength, selection minima, recomputed errors, and old μ-probe reproduction. Three synthetic tests pass: PCA reconstruction/orthonormality, heldout application not mutating calibration, and duplicated code coordinates not increasing kernel strength. Source hashes, PCA components/scalers, full per-candidate CV scores, predictions, and fit kernels are preserved in the artifacts.

Query-to-head reproduction is exact for train512 and dev64. The initial head-only `nn.Sequential` replay of observed96 with medium/B64 yielded max absolute difference `1.373291015625e-4`, mean squared difference `5.50339496197e-11`. CPU2 and CPU4 yielded the same difference, ruling out a simple thread-count explanation. B1 and highest-precision checks changed that discrepancy, but were not used to select representations. Source Q/cache/head/calibration hashes and rows matched. See the [head-only audit](query-accessibility512-head-reproduction-cpu-audit.json).

**The final requested replay using the complete original `train_reference_heads.Model`, strict full state loading, medium precision, CPU4, B64/B32, and `inference_mode` exactly reproduces observed96 μ: max error0, MSE0.** The [replay script](audit_query_mu_model_path.py) and [result](query-accessibility512-model-replay.json) record source hashes and execution settings. Thus immutable Q and μ correspondence is reproduced through the original Model path. The implementation-level reason the stripped head replay differs has not been isolated; no further precision search was performed. The earlier protocol and numerical audit preserve the historical discrepancy instead of being rewritten to conceal it.

The probe always uses immutable saved μ, never a recomputed replacement, and exactly reproduces the previous μ study. No observed96 μ/Q/head was rewritten or selected based on evaluation error. All three synthetic tests and the artifact verification pass; the final full-Model replay resolves the exact-source-correspondence concern while leaving the stripped-head numerical-path discrepancy documented.
