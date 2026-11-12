# Frozen production decoder: early condition adapter audit

2026-11-13. Read-only architecture review; no model implementation, training, or checkpoint modification. Proposed experiment, not an achieved improvement.

## Minimal hypothesis and exact invariants

The original `LAMDecoder_v2` maps the 32D action through `Linear(32,1024)` and sends the same condition through all 12 learned AdaLN blocks. Replace only that projection by `c(z)=b_old+U z`, with frozen original bias and a zero-initialized, bias-free 1024×32 matrix U. Freeze every original weight; preserve final LayerNorm, input/output projections, positional embeddings, and dropout/evaluation behavior. The verified final10000 config has 12 decoder blocks, width1024, FFN expansion4 and dropout0.

This uses 32,768 trainable parameters to test whether the existing nonlinear decoder can exploit reference42 means through its internal modulation pathway. It adds neither a new visual backbone nor the existing approximately13M output correction model. Reference42 means and the entire encoder remain frozen: any gain is **decoder-side reuse**, not an increase in encoder information.

At U=0, every code produces the original zero-code prior. For every learned U, z=0 still produces that same prior mathematically. Test these in the same precision/batch/backend, record numerical tolerance and maximum difference, and verify frozen state hashes after fitting. Replacing the bias with a trainable parameter, adding input centering, or using original `W_old+U` would break the proposed initial/zero-code invariants. Preserve the uncentered cached means in this first comparison.

## Gradients, rank, and manifold risks

For one example, `dL/dU = (dL/dc) z^T`; U=0 does not itself block learning. Frozen weights must have `requires_grad=False`, but the decoder forward must remain in autograd, not `no_grad` or inference mode. All learned AdaLN modulation weights must remain loaded. Their zero initialization at *original training start* is irrelevant unless the trained weights still make the current condition Jacobian zero. Log initial MSE-gradient norm for U, finite checks, and the norm/singular values of learned U. If initial gradients vanish, distinguish zero/constant inputs from decoder conditioning insensitivity before interpreting a failed fit.

The effective local code-to-output Jacobian is `J_c(X,b_old) U`, rank at most32, and can have substantially lower rank. A wide condition vector does not recover missing code information. Across the dataset, the output response depends on X, unlike an additive linear code readout, so the adapter can express some current-dependent effects with the frozen decoder's learned nonlinearities.

Crucially, the old condition was constrained to `b_old + range(W_old)`, a subspace of rank at most32 inside1024D. Arbitrary U can leave it, even though every backbone weight remains unchanged. Neither the bias anchor nor frozen weights guarantees staying near conditions seen in pretraining. The zero-code baseline itself need not be typical under the old posterior. Record train-only `||Uz||`, old native condition displacement norms, and the fraction of adapter displacement orthogonal to `range(W_old)` (rank tolerance declared before evaluation). These are extrapolation diagnostics, not semantic constraints. A later `U=W_old A` restriction would test this explanation, but is a separate experiment and is not automatically required for the first pilot.

## Fixed pilot and controls

Use the existing hash-verified train512/dev64/observed96 caches, fixed reference42 means, and final10000 immutable decoder. All rows, horizon metadata and visual feature normalization remain unchanged. Loss is full feature MSE against the existing future target. Future information enters only through the explicitly posterior-derived cached z and the training target; this is posterior reconstruction, not deployment forecasting.

Fit two predeclared arms for400 updates, common batch stream, optimizer/settings and zero U initialization:

1. Real cached reference42 mean z.
2. One fixed train512 mean z for every training/evaluation example.

Include the frozen zero-code prior and original10000 posterior reconstruction as unfitted references, plus the existing fixed400 output-correction result as a contextual accuracy comparison, not matched capacity/compute. Save train/dev64/observed96 full-row risks at the fixed endpoint, initial/final parameter hashes and initial gradient diagnostics. No evaluation-based update-count, learning-rate, or arm selection. Reuse the current400-step lr3e-4/batch8/AdamW protocol only after declaring it in the launch protocol; do not adjust it here based on outcomes. Use the same gradient clipping rule in both arms.

The constant arm has the same matrix parameter count but only the effective vector `U mean(z)` is observable; it is a global1024D condition adjustment, not equal effective function capacity. Verify the train mean is nonzero. Real-code gradients can be dominated by that same global adjustment because codes have a common component. Consequently real-versus-constant is essential, and zero-code invariance alone is not evidence of transition-specific use. No loss or diagnostic should rely solely on global code swaps.

At fixed inputs the existing physical probe remains exactly unchanged because the encoder/code is frozen. A visual gain cannot be described as improved physical information in z. The model still receives future-derived codes containing possible camera/appearance information; unknown human camera motion is neither identified nor removed.

## Feasibility and rejection criteria

Parameter count is small, but fitting requires backward propagation through all12 large frozen blocks to U. It does **not** have the cost of training only a shallow linear layer; activation memory, attention backward, and total CPU wall time must be measured. Inference removes the extra output correction network but retains the heavy original prior, so large end-to-end speed gains should not be assumed.

- Invalid implementation: frozen state changes, nonfinite values/gradients, initial prior mismatch, or postfit zero-code invariance failure. Stop and repair before interpreting any MSE.
- Optimization/path failure: initial gradients are zero or numerical updates negligible. Reject the chosen pilot path as evidence of successful reuse; do not infer information absence in z.
- Utility failure: real-code held-out reconstruction does not improve over the frozen zero-code prior and the matched constant arm. Reject the claim that this adapter has demonstrated useful transition-specific conditioning. Report paired episode differences; an inconclusive interval is insufficient evidence, not proof of equality.
- Practical failure: it improves the prior but remains materially less accurate than the existing output-correction model without a measured resource benefit. Keep it as a mechanistic result rather than adopt it as the better predictor.
- Generalization failure: training improves substantially while both evaluation splits degrade. Treat this as insufficient512-example evidence or out-of-manifold adaptation, not automatically a code-information failure.

These evaluation sets are already observed and repeatedly used. Report descriptive fixed-protocol outcomes and conditional uncertainty, never an independent final-test claim. This adapter preserves weights, not their behavior away from zero code, and no proposed constraint guarantees camera invariance, action identification, or bounded posterior drift.

## Overlap and novelty

The previous D arm already established decoder adaptation with unchanged codes, but adapted an added output correction network. This proposal instead learns a new input condition map into the original decoder while preserving its visual backbone. It is a different location/capacity test of the same utilization question; it does not supersede or independently establish encoder improvement.

Feature-wise conditioning is established by [FiLM](https://arxiv.org/abs/1709.07871); its abstract was verified, while the publisher PDF was access-blocked in this audit. [DiT §3.2](https://arxiv.org/html/2212.09748v2) describes adaptive-normalization conditioning and zero initialization, already reflected in the production source. Learning a small condition map with a frozen backbone is an adapter-style construction, not a claimed new learning principle. The potentially useful result would be evidence about where this particular pretrained system can use its fixed transition representation, with an honest accuracy/resource tradeoff. This alone is not sufficient novelty for a top-conference paper.

Code inspected: [lam_decoder.py](../../vendor/lawam/latent_action_model/core/utils/lam_decoder.py), especially AdaLNBlock.forward and LAMDecoder_v2.forward; [resolved final10000 config](../../runs/frozen-model-step10000/resolved_config.yaml). Existing evidence: [independent utility addendum](local-repair-independent-utility-interpretation.md).
