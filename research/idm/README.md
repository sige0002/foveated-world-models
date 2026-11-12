# IDM: evidence, diagnostic protocol, and research decision

Updated 2026-11-13. Investigation only; production training and branch unchanged. Diagnostic implementation: [diagnose_posterior.py](diagnose_posterior.py). Checkpoint: `runs/frozen-idm-step2900/`, warm-up ends at step 5000. A result at step 2900 is not evidence that the final training run will fail.

## What the existing evidence establishes

The cached inference code is the Gaussian posterior mean, not a sampled latent. The between-example mean variance is tiny, and replacing it by another episode's mean barely changes reconstruction. Zeroing the latent does change reconstruction, but removes a learned constant conditioning bias and is out of distribution. This does not establish useful action conditioning. A posterior can encode information in variance even when its mean is constant; inspect both before using the word collapse.

The implementation uses a 24-layer query encoder and 12-layer AdaLN decoder. `VAEQuantizer._encode` applies LayerNorm, distinct mean/log-variance projections, and bounded log-variance. Training samples the posterior; default inference returns its mean. Decoder conditioning can exploit a constant mean. Current-state input to the state auxiliary predictor also permits a state-only shortcut; its loss alone does not certify latent action information.

## Diagnostic protocol and interpretation

1. Reproduce cached means from cached current/future DINO features with the original precision. Reject mismatched extraction before interpreting tiny changes. Use float64 stable Gaussian KL arithmetic.
2. Hold current fixed and swap future; separately swap current, duplicate current, and reverse endpoints. Measure mean, log-variance, pre-VAE nodes, and symmetric Gaussian KL. These synthetic pairs diagnose dependence, not valid counterfactual actions.
3. Hold current and random epsilon fixed; exchange posterior mean only, log-variance only, and both. Record prediction change and paired reconstruction risk. Separate posterior information from decoder utilization. Prefer observed posterior values over zero vectors.
4. Optimize per-example z with the frozen decoder, a small trust region, and a prior penalty. A large improvement over inferred z identifies an inference/optimization gap. No improvement suggests weak decoder access or target limitations. This is an oracle diagnostic; it is not an executable policy and can fit camera information.
5. Test held-out incremental action prediction: current-only versus current-plus-z, matched capacity and splits. Test camera pose separately. Global variance and future dependence are not action semantics.
6. Repeat at a post-warm-up checkpoint. Use episode-level uncertainty, at least three training seeds for intervention comparisons, and actual control/planning evaluation before an improvement claim.

## Primary literature and novelty exclusions

| Work and primary source | Sections inspected | Relevant constraint |
|---|---|---|
| [ACT-LAM](https://arxiv.org/html/2609.15189v1) | 3.1–3.3, 4.3 | Residual transition features, action tokens, reconstruction/cycle/reversal/variance objectives, and decoder utilization analysis already exist. Its rollout results do not uniformly beat rollout-tuned alternatives. |
| [Why Latent Actions Fail, and How to Prevent It](https://arxiv.org/html/2605.20223v3) | 3–4.2, Proposition 4.2 | Reconstruction can prefer future nuisance. Linear additive assumptions and independent exogenous dynamics constrain transfer to active human cameras. |
| [On Identifiability of Latent Action Policies](https://arxiv.org/html/2510.01337v1) | 2, Example 3, main theorem | Deterministic behavior admits z-independent exact reconstruction. Discrete-action identifiability requires continuity, support overlap/connectedness, and distinct effects; it does not certify this continuous Gaussian IDM. |
| [Beyond Vanilla VAEs](https://arxiv.org/html/2306.05023v3) | 4.1, Theorem 2 | Conditional collapse depends on unpredictable residual covariance relative to regularization/noise scales in a linear model. The theorem's scales cannot be substituted directly into LaWAM's SmoothL1 objective. |
| [Lagging Inference Networks](https://arxiv.org/html/1901.05534v2) | Abstract, introduction | Aggressive inference updates are an existing remedy. If the decoder is exactly independent of z, reconstruction supplies no encoder gradient. |
| [JEVA](https://singhularadi.github.io/jeva/static/jeva.pdf) | 3.2–3.5, equations 9–13, 5.2 | Posterior samples penalize decoder sensitivity in the affine case through tr(AΣAᵀ). Mean routing on labeled transitions is already proposed. Its own fine-control result is weaker on Reacher: coarse pose representation does not ensure small joint deltas. |
| [WING](https://arxiv.org/html/2610.03607v1) | Introduction; detailed method reviewed in parent investigation | Generic camera/interaction decomposition and camera robustness cannot be claimed as new. Real head motion and parallax need evaluation beyond synthetic translation. |
| [RLA-WM](https://arxiv.org/html/2605.07079v1) | 3, reviewed in parent investigation | Residual DINO compression with learned queries and separate action-conditioned latent flow already exists. |

## Candidate with a defensible problem statement

**Make action information necessary through verified alternative outcomes at matched current states.** The hypothesis is that current-only predictability and data coverage, rather than dimensionality alone, allow the shortcut. Mine or collect short branches with approximately the same complete state but different demonstrated controls/outcomes. The useful branches must differ in interaction effect, not merely camera motion. Robot resettable simulations provide a first controlled test; human transfer comes later.

For two valid branches, assume an L-Lipschitz current-only predictor, current distance at most δ, and prediction error at most ε on each. The triangle inequality gives

`||y_i - y_j|| <= L δ + 2 ε`.

Therefore separated effects exceeding this bound cannot both be fitted accurately by that current-only predictor. This elementary observation supports a falsifiable data/optimization hypothesis; it is not a novel identifiability theorem. Contact discontinuities, incomplete visual state, and unrestricted decoder Lipschitz constants break the practical force of the argument.

Train on such branch groups with ordinary positive reconstruction plus an action-grounded interaction target. Do not reward making incorrect latent swaps arbitrarily bad: that margin is gameable. First compare ordinary sampling and branch-balanced sampling at identical architecture, data count, compute, and regularization. Measure action-conditional effect errors, then planning. Test transfer from a small robot-anchored branch set to unlabeled human clips, with explicitly measured camera/embodiment shift.

Potential research contribution, still unproven: a measurable coverage criterion predicting when IDM learns actions; a branch acquisition/training strategy that repairs this; and improved fine manipulation and human-video transfer under controlled camera shifts. The novelty claim requires a further targeted survey of branch collection and conditional representation learning. Neither the inequality nor pair mining alone is enough for a top conference.

## Fastest useful experiment and stop conditions

- First complete posterior factor diagnostics. If future changes posterior but decoder ignores it, compare aggressive encoder updates, mean routing with sparse action anchoring, and a z-only action-delta auxiliary as known baselines.
- Construct a small resettable branching benchmark with independent camera changes. Cross action diversity (one/multiple action per state) with camera diversity (fixed/moving) while keeping transition count fixed. Evaluate ordinary LaWAM, branch-balanced LaWAM, and relevant ACT-LAM/WING or their explicitly scoped ablations.
- Reject the coverage proposal if branch-balanced sampling has no benefit after matched optimization, if the effect vanishes with a straightforward known anti-collapse baseline, or if gains consist only of camera information.
- Reject a fine-control claim unless held-out small-motion/contact errors and control success improve; image-feature MSE alone is insufficient.
- Increasing 32 dimensions, maximizing variance, generic InfoNCE, latent cycling, or forcing decoder sensitivity are not primary proposals. Each can strengthen scene/camera leakage or create useful-looking but non-executable codes.

## Confidence

High: mean-only diagnostics cannot establish posterior collapse; zero-z perturbations are confounded; listed overlapping methods exist in the inspected primary texts. Moderate: the present checkpoint exhibits weak transition information in both posterior parameters on the measured eight-pair subset; broader and later-checkpoint coverage is required. Hypothesis: action-branch coverage is a major cause and repair mechanism in this dataset. Unknown: whether this direction beats the strong existing methods or merits top-conference acceptance.

## Completed factor diagnostic

[Result JSON](posterior-diagnostic-step2900.json): eight uniformly selected validation pairs, two shared-epsilon samples, CPU four threads, batch size one, original medium matmul precision. Cached posterior means reproduced exactly (maximum absolute error zero). Batch size two changed numerical results enough to fail initial reproduction; those discarded runs produced no result artifact.

| Quantity | Value |
|---|---:|
| Mean between-pair variance, summed over dimensions | 8.062e-6 |
| Log-variance between-pair variance, summed | 9.792e-6 |
| Future-swapped symmetric posterior KL | 2.823e-5 |
| Current duplicated as future: symmetric KL | 1.952e-6 |
| Reversed endpoints: symmetric KL | 8.897e-7 |
| Decoder MSE, true mean | 0.164412221 |
| Decoder MSE, another pair's mean | 0.164412756 |
| Decoder MSE, sampled true posterior | 0.165121400 |
| Decoder MSE, only mean exchanged, same epsilon | 0.165118039 |
| Decoder MSE, only log-variance exchanged, same epsilon | 0.165118616 |
| Decoder MSE, both exchanged, same epsilon | 0.165118143 |

The hypothesis that useful transition information is hidden only in log-variance is unsupported on this subset. Pre-VAE nodes also have a large common component: mean norm squared 1.720e6 versus between-pair total variance 0.0909. This motivates layerwise diagnostics before blaming the final VAE heads. These are dependence measurements on a small early checkpoint; they neither establish action identifiability nor prove every latent pathway is useless. The eight-pair MSE is not directly comparable to the earlier 64-pair mean. No heavy training was run.

## Follow-up: layer localization and existing-data interventions

[Layer/oracle diagnostic](diagnose_layers_oracle.py) uses transparent forward wrappers rather than module hooks, because hooks can disable PyTorch Transformer fast paths. Cached posterior means must match before continuing. For eight pairs, holding the current fixed and replacing the future yields:

| Representation | Common squared norm | Future-swap squared difference | Difference/common |
|---|---:|---:|---:|
| Projected visual input | 148645 | 31396 | 0.211 |
| Initial cross-attention query | 1207 | 0.772 | 6.39e-4 |
| Query after layer 1 | 2087 | 1.026 | 4.91e-4 |
| Query after layer 24 | 1724164 | 1.692 | 9.81e-7 |
| Output projection to 32 dimensions | 1720087 | 0.0930 | 5.41e-8 |

These metrics compare different representations, so the ratios are diagnostic, not mutual information or a causal decomposition. Nevertheless, within the 24 query layers the absolute response to the future is retained while a common component grows much faster. This localizes a testable mechanism beyond a generic VAE-collapse story.

Code review: `QFormer_att` uses pre-normalized residual Transformer layers, returns the final query directly, and has no final normalization before the 32-dimensional projection. The VAE applies its normalization after that projection. The modality mask allows vision-to-vision attention across both times and allows query-to-all attention. It blocks visual-token attention to the query. There is no mistaken mask sign preventing future access. Initial cross-attention adds a learned query, normalizes, then adds an MLP residual.

Mechanistic hypothesis: write a query/output as `c + delta_i`, where c is a large shared vector. For small delta, `LN(c + delta) ≈ LN(c) + J_LN(c) delta`. The Jacobian removes the coordinate-constant and radial components and scales remaining directions inversely with the coordinate standard deviation of c (up to epsilon and learned gain). Thus large nonconstant shared components can suppress transition distinctions after normalization. This is a local Taylor statement, not proof of the cause of training collapse. Variance ratios alone cannot distinguish informative retained directions from numerical artifacts.

The immediate interventions below all use existing data; branch collection is optional further work.

1. **Query accumulation ablation.** Compare 4 versus 24 encoder layers with the same decoder; separately compare final normalization before output projection and residual scaling/gating. Track common norm and matched-future response throughout training. These are standard architecture/optimization controls, not novel algorithms. Do not insert normalization into a trained checkpoint and interpret the resulting out-of-distribution degradation as a fair training comparison.
2. **Current-conditioned reference subtraction.** Compute a transition query `q(x,y)` and a no-change reference `q(x,x)` with shared weights; feed their difference, before VAE normalization, to the posterior head. Both branches receive gradients. This cancels a common current-conditioned offset and encodes no-change at a consistent reference, using available frame pairs. It costs a second encoder pass and may cancel useful state-dependent action context. Direct DINO differences and ACT-LAM/RLA-WM residuals are essential baselines; a different subtraction location alone is not enough novelty. If this is pursued, the mechanism must explain benefits over those baselines.
3. **Ground small interaction changes explicitly.** Add a z-only decoder for robot joint/relative pose increments on already labeled transitions, normalized by train-set per-joint scale with a floor. Unlike a predictor supplied with current state, this gives a direct pressure for usable latent motion. Evaluate incremental prediction over current-only and small-motion bins. Human clips require an independently validated motion teacher or a separate unsupervised branch; do not invent joint labels. Existing JEVA grounding and ACT-LAM transition losses limit novelty. This is an immediate practical repair baseline.
4. **Separate decoder access from amortization.** Use the constrained latent oracle below. If it improves, distill its regularized targets into the IDM or do extra IDM updates with the decoder fixed, compared to lagging-inference and semi-amortized baselines. Distillation is not automatically an action solution: future-only nuisance remains a confound. An exact z-independent decoder supplies no useful oracle gradient.

A viable research claim would connect the measured common-component accumulation to failure on fine interaction prediction, show the condition under which the proposed repair preserves action signal, and beat existing residual/grounding/anti-collapse controls without extra labels or compute. At present the evidence supports running those experiments; it does not establish that claim.

## Completed constrained latent oracle

[Result JSON](layers-oracle-step2900.json). All weights fixed; eight validation pairs; 12 Adam updates on a 32-dimensional latent offset per pair; total Mahalanobis radius at most 1 relative to the inferred diagonal posterior; squared-radius penalty 1e-4. Every pair reached the radius boundary. Mean MSE changed from 0.164413850 to 0.162360745, a 1.25% relative decrease. This supports a small available decoder response beyond the nearly constant inferred mean; it does not imply a fully useful action interface. All eight final errors decreased. These targets use the true future and are an oracle diagnostic, not a deployable prediction improvement. Gradient-enabled computation can choose a different numerical path from inference mode, so compare initial/final values from this same run. The latent move (L2 about 0.44) greatly exceeds natural between-pair mean differences. A tighter radius relative to the empirical mean-code support would be a useful conservative follow-up.

## Adversarial review of the common-component explanation

The layer measurements do **not** identify common-component growth as the cause of low posterior information. Alternative explanations remain:

- An invertible affine change of representation can make a common vector arbitrarily large without losing information. Uncentered rank or difference/common ratios are not information measures.
- At the final query, absolute future response still exists. The learned projection/head might have selected directions that ignore it because the reconstruction optimum is current-predictable, rather than normalization causing the collapse.
- Queries may learn a constant prototype because sampling and KL make transition distinctions expensive. Then norm growth is a symptom.
- Global future swaps are off-manifold. They can expose scene or camera changes rather than meaningful action sensitivity. Eight samples cannot estimate a 1024-dimensional representation rank reliably.
- Deep pre-LN accumulation is established prior art. [Attention Residuals](https://arxiv.org/pdf/2603.15031), abstract/method, addresses uniform residual accumulation and contribution dilution; [Curse of Depth](https://arxiv.org/html/2502.05795v1), sections 3–4, studies variance growth and depth-dependent LN scaling. Neither certifies an action representation.
- [Dong et al., ICML 2021](https://proceedings.mlr.press/v139/dong21a/dong21a.pdf), introduction/theory, analyzes token rank collapse for pure attention and the role of residuals/MLPs. Our cross-example query variation is a different object, and this encoder has residuals and MLPs. Its theorem cannot be imported directly.

### Why normalization before the projection is not the main intervention

Let q be the query, W/b the final projection, m the coordinate mean of q, and s its coordinate standard deviation. Ignore epsilon and use non-affine input normalization. Then

`LN(W LN(q) + b) = LN(Wq - m W1 + s b)`.

The original head normalizes `Wq+b`. Any uniform scale reduction disappears under the second LN. Only coordinate centering through W1 and bias reweighting change this expression; learned per-coordinate affine gains add a change of projection, not guaranteed information recovery. With epsilon retained, the right-hand normalization uses epsilon multiplied by s².

[Measured algebra check](normalization-step2900.json), eight examples using the real projection: equality error 1.11e-15 in float64; pure scaling without bias also cancels to 1.11e-15. Query coordinate standard deviation is 41.03. The normalized projected representation's between-example variance is 1.113295e-6 originally and 1.112042e-6 after adding pre-projection LN. Thus this intervention does not restore variation here. This is an untrained representation comparison, not a performance result. [Script](diagnose_normalization.py).

### Selected intervention for the next mechanism test

**Choose current-conditioned reference subtraction, with a fixed train-calibrated scale, before the learned latent head.** Define

`d(x,y) = Q(x,y) - Q(x,x)` and `z = posterior_head(d / s_train)`.

Use the 1024-dimensional query before the current 32-dimensional projection. Reinitialize/train the posterior head; calibrate s_train only on training transitions with a nonzero floor. Do not normalize every example to unit length: that destroys motion magnitude and magnifies near-static noise. Do not interpret applying this input to the old decoder without training as improvement.

Why this one: unlike scalar rescaling, subtracting a current-conditioned reference can remove a vector offset before the posterior bottleneck. It needs only the existing pair (x,y), has an exact no-change anchor d(x,x)=0, and supports a cheap first test by caching both queries with the frozen encoder. The structural guarantee is only offset cancellation when `Q(x,y)=b(x)+r(x,y)`; useful action recovery additionally requires r to retain action effects rather than nuisance. Both branches should receive gradients in a later end-to-end version. Its cost is approximately two encoder passes; a train-cache pilot avoids this repeated cost.

Minimal pilot: cache train/validation d, original queries, and globally centered queries; train the same small posterior head and the same decoder adaptation for each. Include direct DINO difference as the ACT-LAM/RLA-WM-related baseline, and known z-only action-increment grounding as a separate control. Split by episode. Report future-prediction error, posterior swaps, and incremental held-out action prediction over current-only. Reject if subtraction helps only camera/scene probes, fails to beat global centering or raw difference, or loses fine-motion amplitude. No action labels are needed to build d; existing robot states/actions are used only where actually provided for grounding/evaluation.

Residual/LN depth scaling is the next **known baseline**, with lower runtime overhead but a weaker direct guarantee: scaling can reduce both the shared component and action response. Future-direction projection is deferred: PCA/SVD on future swaps preferentially captures high-energy camera/scene changes, and eight held-out samples are not a training basis. A supervised interaction-effect subspace could help, but then the gain must be compared to ordinary action grounding and cross-exogenous/CCA methods.

This recommendation selects a feasible **experiment**, not a validated top-conference method. Input residuals, cycle constraints, variance regularization, mean routing, and grounding are already covered by ACT-LAM, RLA-WM, JEVA, and VAE literature. A contribution would require evidence that current-conditioned representation offsets are a distinct failure mode and that reference subtraction repairs it beyond these controls. That novelty and performance remain unverified.

## Completed matched-head experiment

The subsequent full train128/validation64, three-seed, 2000-update comparison is complete. Reference subtraction achieved validation mean-code MSE 0.160962±0.000721 versus global centering 0.170286±0.002487. These are seed mean±SD within the fresh small-decoder experiment, not production Stage1 results. [Full experiment and controls](reference-experiment.md), [summary](reference-comparison2000/summary.json), and [semantic/OOD audit](conditional-utility-audit.md). All primary processes have completed; the seed42 reference posterior cache was exported for the parent's decoder comparison.
