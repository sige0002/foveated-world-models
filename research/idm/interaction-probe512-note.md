# State-dependent readout of frozen μ32

Adding a current–code interaction improves overall measured-displacement prediction with unchanged saved μ32. It **does not explain away the right-gripper failure**: right-gripper improvement is small, remains worse than current-only, and large-positive displacement underprediction persists.

This is an exploratory comparison on previously observed dev64/observed96, not an independent final test. Targets are measured endpoint state differences, not action commands/joint angles. Gripper coordinates retain the source dataset's numerical convention; physical aperture units are unverified.

## Fixed model and selection

The [protocol](interaction-probe512/protocol.json) was saved before fitting. Train512 uses the previous exact five episode folds, frozen reference42 μ32, spatial-current candidates, expanded alpha grid, train-only feature/target calibration, and old128 evaluation target scales. All [CV choices](interaction-probe512/train_selection.json) were saved before evaluation prediction. No new labels, images, encoder updates, or world-model training were used.

Write the training-centered/unit-mean-diagonal current and code kernels as `C` and `Z`. The additive baseline uses the normalized sum `C+Z`. The new model constructs `P=normalize(center(C⊙Z))` and normalizes `C+Z+P` to total mean diagonal one. Cross kernels use only training centering means and diagonal scales. The zero-code control is the exact previous current-only kernel.

The Schur product is a known tensor-product kernel: if `C=φ(C)ᵀφ(C′)` and `Z=ψ(z)ᵀψ(z′)`, their product corresponds to features `φ(C)⊗ψ(z)`. With a linear code kernel this permits state-dependent linear coefficients of z, whereas the previous sum only permits a fixed code coefficient plus a current-only function. Product centering preserves positive semidefiniteness. This is standard kernel/varying-coefficient mathematics, not a novelty claim.

Equal total kernel strength prevents a mechanical scale increase, but does not equate model capacity or regularization: adding the product also changes the relative weight of the two main components after normalization. Improvements therefore establish sensitivity to this specified readout design, not a unique causal attribution to a particular interaction.

## Results

Every group selected spatial RBF multiplier1 and alpha0.01, interior to the fixed grid. CV normalized MSE is current-only0.296424, additive0.202925, interaction0.180235. Additive and zero reproduce every previous512 CV candidate score and selected train/dev/96 aggregate MSE within `1e-12`.

| Model | dev64 common-scale MSE | observed96 common-scale MSE | dev right-gripper MSE | observed96 right-gripper MSE |
|---|---:|---:|---:|---:|
| Current-only | 0.420013 | 0.402629 | 0.085939 | 0.076725 |
| Current + linear μ, additive | 0.321819 | 0.316632 | 0.089649 | 0.081496 |
| Add current–μ product | 0.291834 | 0.275508 | 0.089281 | 0.080588 |

Overall MSE falls9.32%/12.99% relative to additive. This uses exactly the same32D representation, demonstrating that an additive readout understates its predictive utility for this task. It is not evidence that μ identifies actions, excludes camera information, or achieves execution-time prior prediction. μ is a future-conditioned posterior mean.

The fixed right-gripper bins use static threshold0.01 and old128 large-movement threshold0.7268722057342529. The large-positive subgroup has7 dev and9 observed96 examples:

| Readout | dev large-positive MSE | dev prediction mean | observed96 large-positive MSE | observed96 prediction mean |
|---|---:|---:|---:|---:|
| Current-only | 0.551072 | 0.240450 | 0.483319 | 0.279458 |
| Additive μ | 0.606409 | 0.205956 | 0.544411 | 0.238065 |
| Interaction μ | 0.618548 | 0.204118 | 0.535810 | 0.249951 |

True means are0.977974/0.959373. Every large-positive magnitude is still underpredicted by more than0.01, without sign errors or missed movement in these groups. The dev subgroup worsens; observed96 improves slightly but remains worse than current-only. Static-subgroup MSE improves on dev0.023001→0.019472 and worsens on observed960.014007→0.015366. Therefore the small whole-coordinate gain does not establish a general right-gripper repair.

Together with the preceding raw-query probe, the evidence is against attributing right-gripper failure primarily to either the μ32 bottleneck or the additive readout alone. The tests cannot establish absence of useful information: other readouts, target weighting, sparse positive-transition coverage, ambiguous current context, and query representation remain possible explanations. No further model or threshold was selected from these observed evaluation results.

## Reproduction and verification

[Script](probe_interaction512.py), [results](interaction-probe512/results.json), [verification](interaction-probe512-verification.json). The script refuses to overwrite an existing output directory. From repository root:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/probe_interaction512.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m unittest research.idm.test_interaction512 -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/verify_interaction512.py
```

Three meaningful synthetic tests pass: additive/zero kernels exactly match the old implementation; the product kernel is PSD with matching train/cross evaluation and unit mean diagonal; and it fits a balanced state-dependent sign function that an additive kernel cannot fit. Artifact verification checks13 immutable source hashes, previous fit hash/common target scale, fold partition, selected minima, physical/normalized errors, kernel strength, and previous baseline reproduction. All candidate CV errors, fitted kernel normalizers, per-row targets/predictions, and physical-coordinate errors are saved. CPU2 only; no GPU or source-cache mutation.
