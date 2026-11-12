# Prepared A/B/C shared-Gaussian-head experiment

Status: implementation and five synthetic gradient-path tests completed. **No real-data training has been started.** The parent currently has other CPU jobs running. Changes are confined to `research/idm`; the shared decoder training script and production model were not edited.

## The question this experiment can answer

Does allowing the flow objective to update the shared latent head help or harm that head and the direct predictor, when the direct objective and all sampled inputs are otherwise controlled?

It cannot by itself establish that future interpolation causes the effect. B versus C isolates the entire flow-to-head gradient, which includes ordinary multitask regularization/interference, loss scaling, stochastic-gradient changes, and effects of flow's conditioning architecture. Q is cached and frozen, so the experiment concerns the Gaussian **head**, not learning throughout the 24-layer query encoder.

Recent parent diagnostics weaken a teacher-specific explanation: velocity sensitivity to a local z swap fell from about .01665 at t=0 to .00407 at t=.75 on teacher states, but also to .00389 on generated states. Generated states already contain effects of z from earlier integration steps. Thus low partial sensitivity with the state held fixed does not imply an uninformative encoder or prove teacher substitution. The A/B/C study is a controlled follow-up, not confirmation of that interpretation.

## Exactly what is shared

[Implementation](train_joint_objectives.py) reuses `make_decoder` and the reference-query representation from the completed experiment. Each arm instantiates an identical Gaussian head, direct residual decoder, and flow decoder, in the same order and from the same seed. Head: 1024→256→64, yielding 32D mean/log-variance. Primary decoder family: matched `wide_residual`/`wide_flow` from the shared factory. Both use a width-128 two-layer trunk, then the same full-feature-width (768D) readout block with raw state/current bypasses. The optional `--decoder-family narrow` uses the earlier 128→768 linear readout only as a fixed-rank control; it is not the primary mechanism test. The direct branch is the existing one-step residual endpoint predictor trained with MSE; “direct” here does not mean the original large production transformer.

| Arm | Objective | Flow's z input | Active branches |
|---|---|---|---|
| A | direct MSE + beta KL | unused | head + direct |
| B | direct MSE + beta KL + lambda flow loss | sampled z | head + direct + flow |
| C | same scalar objective as B | stop_gradient(z) | head + direct + flow |

Defaults: beta=5e-5, lambda=1, AdamW lr=3e-4, existing weight-decay split, batch8, no dropout. beta matches the coefficient in the earlier head pilot; production reconstruction uses SmoothL1, so objective scales are not equivalent. A lambda sweep would be a separate registered experiment, not a post-result rescue.

All arms draw the same batch indices, posterior epsilon, flow source noise, and flow time from separate seeded generators. A still draws unused flow noise/time, keeping schedules identical. The initial full model and complete sampled schedule are SHA256-checked. No shared encoder parameters, decoder weights, optimizer moments, batch normalization statistics, or running calibration connect direct and flow besides z in B. B and C have equal architecture and forward values at the same parameter state; only flow-to-head differentiation changes.

## Critical clipping and negative control

Global gradient clipping would be a hidden coupling: a large flow-decoder gradient could shrink head/direct updates even after detaching z. The implementation clips head, direct decoder, and flow decoder separately. Loss statistics log each pre-clip norm. AdamW updates are parameter-local; A's unused flow parameters have no gradients and are not updated.

With these controls, **A and C must produce identical head and direct-decoder parameter trajectories** for the same seed/schedule, up to implementation nondeterminism. The script requires exact final hashes to match and stops if they differ. This is stronger than hoping their scalar validation metrics look similar. C's trained flow is then evaluated on the same head trajectory as A. B/C comparisons are still allowed to differ through B's additional gradient into the head.

## Future-information routes

1. All arms infer z from cached `Q(current,true_future)-Q(current,current)`. The posterior itself therefore uses the real future. This is Stage1 reconstruction, not current-only deployed prediction.
2. Direct decoder input is current features plus z. Future features appear only in its MSE target; there is no extra target-dependent decoder input.
3. Flow training starts at `x0=current+0.1*epsilon_flow`, forms `x_t=(1-t)*x0+t*true_future`, and predicts velocity with inputs x_t, t, current, z. The velocity target is true_future−x0. This interpolation is the intended flow-matching training objective, not an accidental software leak. It provides a second route for future content besides z and motivates the hypothesis under study.
4. C stops only the flow input z's gradient. It does not remove future interpolation or change direct-head/KL gradients. Neither direct nor flow reconstruction targets receive gradients because features are frozen cache tensors.
5. Flow evaluation calls the sampler, which accepts current, posterior z and noise but **no future target**; all four Euler states are generated. The z remains a true-future posterior, so the result is still not a deployed action-prior forecast. The target-dependent `training_output.prediction` is never used as validation prediction.

## Necessary tests, completed

[Tests](test_joint_objectives.py), [recorded result](joint-objective-test-result.txt): five tests passed on the updated wide-primary implementation on CPU, without dataset training.

- At identical B/C parameters, direct/KL/flow losses are identical; flow-decoder gradients are identical; B has a flow-to-head gradient and C does not.
- The B-minus-C head gradient equals lambda times the flow head gradient within floating-point tolerance.
- Three synthetic AdamW updates with separate clipping leave A/C head and direct parameters bitwise identical.
- The captured flow training state matches the prescribed future interpolation; the sampler rejects a `target` argument.
- Both narrow and wide factories remain available with nonzero flow-to-head paths after opening gates; small synthetic feature dimensions keep tests cheap.

Tests deliberately make AdaLN gates nonzero when checking positive gradients. Actual gates initialize at zero, so a zero flow/head derivative on the first update is expected and is not evidence of learned collapse. The training script leaves the standard initialization intact.

Reproduce tests:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python research/idm/test_joint_objectives.py
```

Prepared training command — **not executed**:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python research/idm/train_joint_objectives.py --cache runs/frozen-idm-step2900 --queries research/idm/reference-query-full --output research/idm/joint-abc-smoke --decoder-family wide --steps 400 --seeds 42
```

## Measurements and compute limits

At fixed steps, report direct reconstruction, posterior mean/logvar variance, and observed-code swap stress results for all arms; generated-state flow reconstruction for B/C; raw direct/flow head-gradient norms and their cosine before clipping; parameter counts, hashes, and training elapsed time. The recorded cosine excludes KL and reports unweighted flow gradients, so lambda and KL remain explicit rather than hidden in a single alignment number.

A executes one decoder branch; B/C execute two. Allocating the unused flow model in A preserves initialization, not compute equality. B also backpropagates flow through z/head; C omits that path, so even B/C backward cost is not strictly identical. The script reports active and allocated parameters and measured time; equal updates is the primary control, not equal FLOPs. The four-step flow sampler also costs more than the one-step direct predictor.

The existing global cyclic code swap is only a dependency stress test. Our earlier radius-constrained current-neighbor matching failed, so it cannot certify action use. No action/camera semantics are inferred from z variance, KL, gradient norms, or these swap scores.

## Interpretation and falsification

- If A/C hashes differ, reject the experiment implementation until the unintended coupling or numerical issue is explained.
- If B worsens direct held-out performance and reduces useful posterior dependence relative to C, this supports harmful **flow-gradient interaction** at the chosen scale. It does not uniquely identify future interpolation as the cause.
- If B improves direct performance or gives no degradation, the proposed harmful-sharing explanation is unsupported under these conditions. A tiny flow-to-head gradient may also mean the experiment has little power to change the head.
- If C's flow performs better than B's, flow optimization may benefit from a head trained only for direct reconstruction. This still does not imply that the original full IDM would behave the same way.
- A future-interpolation-specific claim needs an additional controlled intervention on the flow's state/gradient information path. Changing to t=0-only supervision or replacing teacher states with generated states also changes the training objective/support; neither is a free causal control. Do not retrofit such an arm and call it equivalent.

Possible degeneracies include nuisance-rich z, a constant head maintained by KL/current predictability, flow ignoring z while exploiting x_t, direct decoding carrying the useful signal while flow adds noise, or B acting merely as a regularizer. Decoder integration/endpoint calibration can impair flow even with a good head. Current evidence that teacher and generated local sensitivities both decline makes these alternatives material.

Novelty is not claimed. This is a standard stop-gradient/multitask ablation designed to test a particular mechanism with reproducible controls. A successful small-cache result would motivate, not replace, full-encoder and action/physical-target evaluation.

## Readout-rank correction before any training

Root review identified the narrow 128→768 flow readout as an existing representational limitation that could make an auxiliary flow branch artificially weak. The prepared experiment therefore defaults to the shared factory's `wide_residual`/`wide_flow`, retaining equal B/C architecture and the A/C negative control. No actual training used the earlier narrow draft. Wide readout removes that particular fixed linear-output-rank ceiling; it does not guarantee a good velocity field, calibrated integration, or action semantics. The narrow family is an explicitly labeled secondary control. Synthetic tests use feature width 8 and trunk width 16 and remain CPU-only.
