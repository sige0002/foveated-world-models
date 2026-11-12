# Next experiment: ground the extra information supplied by z in measured motion

## Decision

Run **one matched three-arm objective comparison** on the existing frozen reference-query cache: reconstruction only; ordinary measured-increment grounding; and current-residual increment grounding. The goal is to test whether posterior information improves prediction of a physical measurement beyond current information, rather than merely increasing image reconstruction or swap sensitivity. This is a testable proposal, not a novel-method or camera-invariance claim.

Reference centering improved our small-decoder reconstruction, but its global swap gap fell under smaller-current-distance swaps and a genuine radius-constrained matching was infeasible. Those results justify a different target for learning and evaluation; maximizing the existing swap gap would reward distribution shift and future-image coding.

## Available labels and implementation

`vendor/dataset/robomind.py` defines each arm as xyz, two rotation columns, and gripper, concatenated into 20 dimensions. It is **not a joint-command or action-chunk label**. Use an eight-dimensional initial target A: endpoint xyz displacement and gripper change for each arm, over the exact existing approximately 1.2-second pair. Do not claim that these measurements uniquely identify the executed controls. Preserve each arm's source coordinate frame. Orientation is excluded from this first experiment rather than treating rotation-column subtraction as an angular action.

Labels can be recovered without video decoding: instantiate the same dataset with the resolved config and `require_observed_endpoints=True`; use each parent cache row's dataset index to look up `dataset.pairs`, then `_arrays` for the two endpoints. Verify episode id, timestamps/dt, source manifest SHA and original-valid flags against the parent extraction. Select indices [0,1,2,9,10,11,12,19] from future minus current prepared state. The cached current/future image features remain unchanged.

Let X be all available current information and m(X) an independently fitted current-only predictor of A. Fit m by episode-level five-fold cross-fitting on train only, freeze it, and generate out-of-fold training predictions. Validation uses the average of train-fold predictors. Its inputs include current visual features and current 20D state; no future, query, z, or future-derived masks. The precise current representation and baseline capacity must be reported: a pooled feature baseline conditions on less than the full image, so its guarantee applies only to that summary, not all information available to IDM.

Start with ridge regression on standardized current mean-pooled DINO plus current state, choosing its penalty by train-only inner folds. Then check residual predictability with a separately fitted nonlinear current-only probe. If that probe explains the residual, the baseline is inadequate and the result must be called baseline correction, not future-specific motion information. The 128-pair cache may be too small to pass this check; that failure is informative and must not be hidden by changing the held-out test.

Use the same train-only per-coordinate scale D for both grounding objectives: observed-increment standard deviation, floored at 1 mm for translation and 0.01 for gripper. Do not whiten every sample separately. Define

`R = D^-1 [A - stopgrad(m(X))]`.

With sampled z from the existing Gaussian posterior, train a small auxiliary MLP H(z) to predict R:

`L = L_visual + 5e-5 KL(q(z|x,y)||N(0,I)) + 0.01 E ||H(z)-R||²_mean_dimensions`.

For ordinary grounding, replace R by `D^-1[A-train_mean(A)]`. Both arms use the same auxiliary architecture, initial weights, samples, batches, learning rate, label mask and coefficient. Reconstruction-only keeps the same auxiliary module with coefficient zero. Keep posterior sampling in all arms; do not introduce JEVA's mean routing in only one arm. Keep the existing reference representation fixed across arms so this experiment isolates the objective.

The reported gain `E||R||² - E||R-H(z)||²` is a diagnostic. Minimizing its negative is just proper squared prediction loss plus a constant, so there is no incentive to deliberately worsen a wrong-code prediction. [Loss helper](conditional_gain_objective.py) implements detached targets/baselines/scales and was checked on balanced branches, nuisance-only codes, a baseline-error counterexample, and stop-gradient behavior. Baseline fitting and real-data training have **not** been run.

## What the theory actually says

Assume m(X)=E[A|X] and finite second moments. Then E[R|X]=0. Any z that is only a deterministic function of X cannot obtain positive optimal squared-error gain over H=0. More generally, a camera-only code cannot help if its nuisance variable N satisfies E[R|X,N]=0. This is a conditional-mean statement; it does not require or establish full statistical independence.

If z contains additional predictive motion information, its optimal gain is

`Γ_R = E ||E[R|Z]||²`.

For a context-aware evaluation head, replace the inner mean by E[R|X,Z]. This encourages prediction of the unpredictable component of a **measured** physical increment. It does not identify a true action coordinate system.

If the baseline is inaccurate, a current-only code can obtain gain up to the scaled baseline error `E||D^-1(E[A|X]-m_hat(X))||²`. Cross-fitting limits own-example fitting but does not make m_hat correct. Our loss is not a Neyman-orthogonal causal estimator; DML's causal guarantees do not transfer. The included synthetic check explicitly shows positive apparent gain from current-only information when the baseline is wrong.

For two genuinely identical currents X with different measured outcomes, R_i−R_j=D^-1(A_i−A_j): residualization preserves their difference. In contrast, if the behavior is deterministic given X, R can be zero. The objective then supplies no additional action-learning signal. It removes predictable behavior from the auxiliary target; it must not be interpreted as a universal representation of all actions. The combined prediction m(X)+D H(z) retains the baseline component, while z alone is an innovation representation.

## Leakage and nuisance failure modes

- Independent future lighting/camera noise does not reduce the physical-increment loss under the conditional-mean assumption. However, the visual reconstruction term can still put it into unused latent dimensions. This objective prioritizes useful content; it does **not** certify no leakage.
- If head motion or scene changes correlate with the measured motion after conditioning on X, a camera code can predict R. No observation-only residual loss distinguishes this from action information. Paired views or controlled camera interventions are required for the stronger claim.
- A robot can move without moving an object, or different control chunks can reach the same endpoint. Eight endpoint increments do not certify contact or fine manipulation. They are a feasible first measured target; later trajectory/contact/object labels are a separate extension.
- Sensor noise, near-static transitions, and an underestimated scale can make the objective reward noise. Floors, observed-valid endpoint filtering, and small-motion-bin reporting are required.
- A residual target can be less transferable across embodiments or behavior policies than ordinary grounding. It depends on the current-only behavior baseline and cannot silently be treated as universal action semantics.

## One prespecified comparison and its gates

Use the existing train128/validation64 pairs and the same reference-query Gaussian head plus width-128 two-layer decoder. Train the three objective arms with seeds 42/43/44, first a 400-step implementation gate and then the already-used 2000-step budget if stable. No new large model or camera estimator is needed. Recovering the labels and baseline quality check come first; do not launch training if alignment fails.

Evaluate **all arms** with matched current-plus-z versus current-only probes: same network capacity and current inputs, replace the z input by zero for the current-only control, train independently with matched seeds. Predict A itself, not only the residual training target. Also compare frozen m(X) against m(X)+D H(z) in the residual arm. Distinguish these two measurements; the first is a representation audit, the second evaluates the auxiliary predictor.

Primary success condition: residual grounding beats ordinary full-increment grounding on held-out incremental physical prediction beyond current information, with the benefit present in small-motion bins. Visual feature error should also be reported; if it worsens, the result is a tradeoff, not improved world-model prediction accuracy. Preserve the current-distance swap audit as an exploratory secondary metric, not the success criterion.

Reject the idea if it only improves the fitted baseline's residual while a stronger current-only probe explains the effect; if ordinary grounding matches or beats it; if gains occur only on large pose changes; or if visual and physical prediction both deteriorate. A reconstructed-image or latent-cycle auxiliary is not an acceptable substitute for the measured target in this test.

## Primary literature checked and precise overlap

- [JEVA](https://singhularadi.github.io/jeva/static/jeva.pdf), sections 3.4–3.5 and fine-control results: already grounds Gaussian latents with a z-only action-chunk decoder and changes sampling routes on labeled data. Ordinary grounding is therefore a required baseline. Our proposed distinction is a frozen current-only residual target with a baseline-error audit; its value is unproven.
- [AD-WM](https://arxiv.org/html/2609.30264v1), section III-C, equations 3–4: already recovers action embeddings from predicted endpoints and motivates standardized Gaussian recovery via conditional MI. It also evaluates action-ranking/planning metrics separately from factual MSE. Merely recovering action/z from predicted features is not new. Our first experiment uses observed motion increments and an out-of-fold conditional baseline, not a learned action embedding target or decoder-generated endpoint.
- [Why Latent Actions Fail](https://arxiv.org/html/2605.20223v3), sections 4.2 and 4.4: reconstruction can prefer nuisance; exogenous-swap reconstruction assumes synchronized physical transitions under different exogenous states and connects to CCA. Those synchronized interventions are not present in this cache. We cannot claim its nuisance separation guarantee from residual grounding.
- [Double/Debiased Machine Learning](https://arxiv.org/pdf/1608.00060), introduction/partialling out, section 2 orthogonal scores, and section 3 cross-fitting: supplies established residualization and sample-splitting ideas. Our simple outcome-residual MSE is not its orthogonal score and has no inherited causal or root-n inference guarantee.
- [ACT-LAM](https://arxiv.org/html/2609.15189v1), sections 3.2–3.3: transition residuals, action-centric losses and cycle/reversal/variance constraints are established. [RLA-WM](https://arxiv.org/html/2605.07079v1), section 3: compressed visual residuals are established. The proposed novelty cannot be the word residual or the query subtraction already tested.

## Human-video transfer boundary

On unlabeled human clips, the physical-increment objective is unavailable. Keep the original unlabeled visual objective there and apply the new term only to genuinely labeled robot samples, using the identical labeled fraction for all baselines. Shared latent capacity alone does not guarantee that the robot anchor transfers to humans. Transfer would require comparable observable interaction effects, adequate support, and resistance to embodiment/domain shortcuts, then a held-out transfer evaluation. Human pose pseudolabels have different coordinates and errors; adding them would be a new assumption and experiment. We do not claim few-label human transfer from the present proposal.

## Status

Completed: primary-method inspection, code/label-schema verification, objective implementation and synthetic falsification checks, and a fully specified next comparison. Not completed: aligned label extraction, baseline residual calibration, real-data objective training, camera-controlled evaluation, or novelty establishment. The proposal is deliberately a falsifiable next experiment, not a promise of top-conference acceptance.

## Why predicted-image-to-z recovery is not the selected objective

AD-WM already applies action recovery to predicted endpoints, and ACT-LAM includes latent-cycle consistency. Moreover, an unconstrained recovery cycle permits two failures: a constant z can satisfy a constant recovery target, and a decoder/recognizer can hide z in tiny prediction perturbations that barely affect reconstruction (steganography), with a high-gain recovery head reading them. A nonconstant measured target helps rule out the first failure; an independent physical target/control evaluation is still needed to reject the second. We therefore do not adopt a latent reconstruction cycle alone or treat its accuracy as action evidence.

## Label-free latent refinement: adopt as a diagnosis, not the semantic solution

Use future-target latent refinement **before** deciding whether additional grounding is needed. Freeze the decoder and, for each observed transition, optimize the posterior mean locally while holding its diagonal variance fixed. If q_old=N(mu,sigma²) and q_ref=N(mu+delta,sigma²), then `KL(q_ref||q_old)=0.5 sum(delta²/sigma²)`. A fixed Mahalanobis radius and a KL penalty make the optimization budget explicit. A few gradient steps distinguish an accessible but poorly inferred latent from a locally unresponsive decoder.

We already performed a limited version on the original step2900 decoder: eight examples, 12 steps, radius 1, penalty 1e-4; MSE decreased 1.25%, and all examples hit the radius boundary. This shows a small local correction opportunity, not large hidden action capacity. The moves greatly exceeded natural posterior-mean variation. A single shared latent offset is an important additional control: if it explains the gain, the refinement mostly repairs a common bias rather than example-specific future information.

For an eventual repair test, refine **training examples only**, then distill those targets into the IDM with the decoder frozen. Compare to the same number of ordinary encoder updates and a shared-offset target, with matched optimization budgets. Evaluate untouched validation episodes using the distilled encoder without validation-time optimization. If one optimizes z against a validation future and reports that same future's error, that is an oracle diagnostic, not held-out deployable prediction performance. Even the distilled posterior still consumes the true future and is distinct from the current-only action prior.

[Semiamortized Variational Autoencoders](https://arxiv.org/abs/1802.02550) and [Lagging Inference Networks](https://arxiv.org/html/1901.05534v2) are the relevant established family; the latter's introduction explicitly discusses per-example refinement and amortization. This proposal does not claim novelty over them. Future-only camera information remains a valid optimization shortcut. Thus **adopt refinement for inference-lag versus decoder-access diagnosis and label-free optimization repair; do not adopt it as evidence of motion/action semantics**. The measured-increment comparison above remains the next objective test for that semantic question.

## Reproducible synthetic checks

Run from the repository root:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python research/idm/test_conditional_gain.py
```

[Five-test script](test_conditional_gain.py) and [recorded successful result](conditional-gain-test-result.txt) cover the exact balanced finite-support examples: null gain, useful physical innovation, independent nuisance, a deliberately misspecified current-only baseline that permits a shortcut, and frozen-baseline gradient routing. These are algebra/implementation checks, not empirical validation on robot or human data.

Source precision for refinement: Semi-Amortized VAE section 1 also explicitly discusses earlier methods that train an inference network toward refined variational distributions using KL. Its own main method differentiates through local optimization; the proposed frozen-decoder target distillation is a simpler diagnostic/repair variant and should not be described as that exact algorithm or as a new refinement principle.
