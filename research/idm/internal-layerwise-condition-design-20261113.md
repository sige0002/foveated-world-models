# Integrate z through the pretrained visual backbone, not an extra visual tower

2026-11-13. Initially a read-only design. The research implementation and single-step smoke test were subsequently added in [layerwise_condition.py](../layerwise_condition.py) and [check_layerwise_condition.py](../check_layerwise_condition.py). User-authorized GPU fitting subsequently compared five arms across three decoder seeds using rebuilt final10000-encoder inputs; see the GPU validation section of [the research README](../README.md). Historical encoder2900 caches were unavailable, so this is not their exact replication. Layerwise fitting did not reliably beat the constant-code control or match external-correction MSE; production integration remains unperformed. The user prefers an actually integrated predictor if its advantage is clear. Merely wrapping an external correction network in one class does not meet that requirement.

## Concrete architecture

Keep final10000's input projection, positional embedding, all12 learned AdaLN blocks, output projection and final normalization. For block l, replace its shared condition with

`c_l = b_old + U_l z`, where `U_l: R32 -> R1024` has no bias and starts at zero.

The same original projection bias b_old is frozen and reused at every layer. Freeze every pretrained visual/modulation weight in the primary intervention. The external call remains `prediction = model(current_features, z)`; it runs **one** pretrained12-block visual backbone, with no extra visual tower, feature-level correction network, or separate prior forward. z remains the existing reference42 mean throughout this experiment.

At initialization, every z yields the original zero-code prior. After fitting, z=0 still yields that prior, because c_l=b_old for all l. Check these properties under the same precision/backend; do not confuse this anchor with the original posterior prediction. Original decoder weights are preserved, but conditioned behavior away from z=0 is intentionally changed.

Implement the12 maps as one bias-free `Linear(32,12*1024)` and reshape to `[B,12,1024]`, then call each existing block with its corresponding condition. This avoids twelve separate tiny projection calls. Zero-initialize this matrix and preserve original global initialization RNG. Store the immutable base-checkpoint hash, original c0, adapter weights, dimensions and layout version. An inference checkpoint may package them together, but must still use the single shared backbone computation. The former32→1024 projection weights can be retained as a non-executed diagnostic buffer, not applied to reference z as though it were the native posterior.

## Hypothesis, gradients and cost

The current shared adapter imposes U_1=...=U_12. The proposal removes that tying, allowing early/late blocks to use different projections of the same z. If g_l denotes the backpropagated condition gradient at block l, then

`grad_U(shared) = sum_l g_l z^T`, while `grad_U_l(layerwise) = g_l z^T`.

Opposing layer gradients may cancel under the shared constraint. This is a testable optimization/expressivity hypothesis, not a demonstrated explanation. At zero U, pretrained AdaLN sensitivity permits initial gradients; frozen parameters must retain an autograd-enabled forward. Log per-layer condition/adapter gradient norms and the ratio `||sum_l grad_U_l|| / sum_l ||grad_U_l||` on fixed training batches, interpreting zero-denominator cases separately. This ratio describes cancellation, not action information. No additional deterministic mean drift or encoder change is involved.

Stacking12 maps does **not** increase incoming information dimension to384: the stacked12288×32 map still has rank at most32. The output's local z Jacobian is also rank at most32. The proposed flexibility is layer-specific decoding, and a gain cannot establish that z contains more information.

| Increment | Shared U | Layerwise U_l |
| --- | ---: | ---: |
| Trainable condition parameters |32,768 |393,216 |
| FP32 adapter storage |0.125 MiB |1.5 MiB |
| Condition projection MAC / observation |32,768 |393,216 |

Difference:360,448 parameters and MACs. These projections act on one code per observation, not each of256 patch tokens; multiplying this cost by256 would be wrong. Pretrained block modulation already executes separately per block in both designs. Bias additions/nonlinearities and runtime overhead are not included in MAC counts. The original large visual backbone still dominates computation. Compared with the external correction architecture, one extra visual pass is removed in principle, but actual latency/memory and accuracy must be measured. Frozen-weight fitting still needs backward through all blocks to each adapter, so small trainable parameter count does not imply cheap training.

Independent maps can also overfit512 examples, steer the frozen network outside native condition support, or fail because its modulation Jacobians ignore needed directions. The final LayerNorm floor recently measured by the parent (.002263 on observed96) does not by itself explain most baseline error or guarantee this architecture's success.

## Minimum fair comparison

Keep cached current/future features, z0, rows, precision, optimizer convention, objective and update budget fixed. Do not simultaneously change to a new encoder, add physical supervision, alter forecast horizons, or replace MSE because a new evaluation metric looks favorable. Smooth-L1(beta.1), cosine and high-change-region MSE are additional fixed-checkpoint metrics, not an invitation to select whichever metric wins.

1. **Shared frozen-backbone U:** current integration experiment, same prior and z.
2. **Layerwise frozen-backbone U_l:** proposed intervention, same zero initialization and loss. More trainable parameters are intentional and reported; this is not an equal-capacity claim.
3. **Original-backbone continuation:** same pretrained visual weights and zero-initialized shared32→1024 map, but allow the original decoder weights to update too. This tests whether broader decoder adaptation suffices. The initial prior is identical; after training zero-code prior preservation is no longer guaranteed, which must be reported rather than hidden.
4. **Layerwise current-only condition control:** identical U_l implementation and optimizer, with one fixed train-mean code normalized to the train RMS norm, as prespecified for the current shared pilot. It can learn layerwise global conditions, so real-code superiority must exceed this control. Its nominal parameter count matches, but effective constant-input function capacity is lower.

Use the current fixed400/batch8/lr3e-4 protocol, matrix wd.01 and bias/norm wd0, with common minibatch streams across paired arms, and prespecified seeds. Existing completed external-correction results are an engineering reference; their different architecture and prior training history preclude a pure causal capacity comparison. If the old shared experiment used exactly the same source/seed/batches/objective, reuse it with hashes; otherwise label it historical rather than silently calling it matched. No execution is authorized by this design note.

The original-backbone continuation control intentionally relaxes the frozen-weight constraint. If a zero-code-preserving full-finetune control is later required, subtracting a learned zero-code forward and adding the frozen prior would require additional passes and change the efficiency question; do not smuggle that construction into this single-backbone proposal.

## Acceptance, deployment boundary and novelty

Before implementation, declare the primary visual metric and tolerated regressions on the additional metrics. For an integration claim, layerwise must have a clear accuracy/resource advantage versus the existing external correction, while beating its own constant control and surviving paired-seed/episode uncertainty. If it only improves over shared U but materially underperforms the external model, report a useful routing result, not completed integration. If full-decoder continuation matches or exceeds it, prefer the simpler justified explanation that adaptation, rather than a special z-routing principle, mattered. Do not merge merely because the class/API looks cleaner.

Benchmark the complete `(current,z)->prediction` path, with the same batch/thread/backend and warmup, including the current-only prior wherever a comparison requires it. Report correction-only timing separately, never as end-to-end speed. Include parameter memory and peak inference memory. Shared backbone structure alone guarantees neither maintained accuracy nor faster measured runtime.

This changes **how the decoder consumes an unchanged code**. It is distinct from distilling/retraining the IDM, which changes z's content, scaling or deployment prior and needs a separate matched comparison. Here independent measured-effect probe predictions remain unchanged because z remains fixed. Reference z still uses the true future in posterior evaluation; integrated reconstruction is not executable forecasting until a verified action-conditioned code route exists. Camera/appearance information may drive any visual gain. The local10000 backbone is not official released LaWAM weights; all previous reproduction limitations still apply.

Layerwise conditioning is established: [FiLM](https://arxiv.org/abs/1709.07871) uses conditional feature modulation and [DiT §3.2](https://arxiv.org/html/2212.09748v2) describes per-block adaptive normalization. The production [decoder](../../vendor/lawam/latent_action_model/core/utils/lam_decoder.py) already uses separate learned modulation modules with one shared input condition. Learning independent zero-anchored input maps into those blocks is an adapter-style parameterization, not a new conditioning principle. Conference-level novelty would require a broader demonstrated mechanism or algorithmic result, not this integration alone.

Related audits: [shared adapter design](early-condition-adapter-readonly-audit.md), [decoder utilization versus encoder information](local-repair-independent-utility-interpretation.md), [official-source fidelity audit](official-source-audit-20261113/report.md).
