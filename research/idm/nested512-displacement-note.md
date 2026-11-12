# Nested512 probe: better overall utility, reduced but unresolved right-gripper failure

The fixed reference42 code and its original128-example calibration were **not retrained**. Only the displacement probe was refit with the fixed nested512 dataset. All settings were saved before fitting; all train-CV choices were saved before evaluating the already research-observed64/96 sets. Computation used CPU2, existing cached features and NPZ states, without GPU, image decoding or world-model training.

## Controlled protocol and comparable metrics

The four current-kernel candidates, extended alpha grid1e-8…1e4, per-fold scaling, target floors and train-centered/unit-diagonal kernel normalization are unchanged from the128 probe. Zero and reference code are the two groups. The old128 episode-fold assignments are retained; added384 indices are seeded-shuffled and distributed across the same5 folds. Labels/current20D states for retained128 match exactly. All512 labels are observed, and train512 episodes are disjoint from the160 evaluation episodes.

Target scalers are refit within each512 training fold as required for training-only CV. **Cross-size comparisons below use the fixed old128 target scale**, not the new512 scale. Both metrics are saved for every row. All groups selected spatial-RBF multiplier1 and alpha.01, interior points of the prespecified candidates/grid; no extension or evaluation-based reselection occurred. Train-CV normalized losses are .296424(zero) and.202925(reference), using their fold-specific target scales.

| Probe input | Train size | Dev64 MSE, old128 scale | Observed96 MSE, old128 scale |
|---|---:|---:|---:|
| current-only |128|.599409|.566081|
| current + reference mu |128|.489638|.437926|
| current-only |512|.420013|.402629|
| current + reference mu |512|**.321819**|**.316632**|

Reference adds23.38%/21.36% improvement over the corresponding512 current-only model. Both baseline and reference benefit from the larger dataset. The512-trained reference results normalized using its *new* target scales would instead be .303188/.294466; these numbers must not be substituted into the cross-size comparison.

All six xyz-coordinate errors improve with reference on both sets. Right gripper remains worse than current-only on both; left gripper is also slightly worse on dev64 (.029149→.030331) but improves on observed96 (.050444→.048541). Aggregate utility is therefore not uniform fine-control utility.

## Right-gripper physical errors and fixed difficult subgroup

All gripper values remain source units. These are measured endpoint differences, not commands/apertures/velocities. The original128-derived bins remain unchanged: static |delta|≤.01; large |delta|>.7268722. Large-positive evaluation membership is unchanged (7 dev,9 observed96).

| Input / training size | Dev right-gripper MSE | Observed96 right-gripper MSE | Dev large-positive MSE | Observed96 large-positive MSE |
|---|---:|---:|---:|---:|
| current-only /128 |.100462|.094619|.620114|.536882|
| reference /128 |.127678|.112079|.861424|.766326|
| current-only /512 |.085939|.076725|.551072|.483319|
| reference /512 |.089649|.081496|.606409|.544411|

Thus the gripper failure **improves but is not solved**. The reference-vs-current-only right-gripper penalty shrinks from+.027216/+.017460 at128 to+.003710/+.004771 at512. Large-positive reference predictions remain much too small:

| Set | Mean true large-positive delta | Mean reference prediction128→512 | Mean signed bias128→512 |
|---|---:|---:|---:|
| dev64 |.977974|.051954→.205956|−.926019→−.772018|
| observed96 |.959373|.090728→.238065|−.868646→−.721309|

All active examples now have detected changes with correct signs for both512 probes at the fixed .01 threshold. Every large-positive example still underpredicts magnitude by more than.01. The dominant unresolved issue is magnitude regression, not direction detection. Static threshold crossings are40/50(current-only) versus39/50(reference) on dev and46/75 versus59/75 on observed96; these are diagnostics of regression outputs, not issued control events. Full MAE, bias, missed/wrong-sign counts and physical-coordinate MSEs are saved.

The fixed nested selection increases large-positive training examples3→18. That is consistent with improved subgroup coverage, but these results do not isolate a causal explanation: more observations, changed support, refitted target scales and current/kernel geometry all contribute. Kernel regression also has more training basis points at512; this is a same-algorithm data-scale comparison, not a fixed-number-of-coefficients experiment. It does not demonstrate that the frozen IDM itself learned more from512 examples.

## Limits and integrity

Both evaluation sets have already been used during research; neither is an independent final test. This remains deterministic-posterior-mean utility, not a guarantee for one sampled Gaussian latent or an executable future predictor. Stronger current representations and alternate regressors may explain additional information. No future evaluation target was used to select hyperparameters, bins or calibration.

`nested512-displacement-probe/` contains the presaved protocol, observed training labels/states/masks, all fold CV scores, saved selections, fit artifacts, every evaluation prediction and both target-normalized metrics. Existing artifacts were not changed. `verify_nested512_probe.py` checks source/NPZ hashes, exact old128 labels/states/folds, common128 scale, train-only argmin choices, all evaluation targets/row alignment, both MSE normalizations and the gripper decomposition; all checks pass in `nested512-probe-verification.json`. No new sweep was run beyond the prespecified candidates.
