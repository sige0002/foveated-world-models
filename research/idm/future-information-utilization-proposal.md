# Future information that is useful and actually used: a falsifiable IDM objective

**Decision:** test an effect-conditioned information budget against ordinary grounding, using existing robot labels, before attempting a larger IDM redesign. This is a small, implementable falsification experiment, **not yet a recommended replacement for the human-video world model**. The hypothesis is that the posterior can retain useful transition effects while transmitting less future-specific information that an effect-conditioned code generator cannot reproduce. Its adoption requires better image-feature prediction as well as physical utility; sacrificing prediction accuracy merely to obtain cleaner-looking latents does not meet the user's objective.

No training was launched. This note applies the deep-research workflow to local evidence, implementation constraints, primary methods, and explicit counterexamples. Novelty is not established: the components are conditional VAE objectives, grounded latent prediction, and shared-decoder training. A conference contribution would require a distinct, demonstrated mechanism and broader validation, not this combination alone.

## What the completed experiments actually constrain

The frozen reference μ is useful beyond a strong spatial-current probe on512 examples. Allowing a current–μ product kernel improves aggregate measured-displacement MSE from0.321819/0.316632 to0.291834/0.275508 on already observed64/96. Therefore the earlier additive probe underestimated accessible utility. Raw1024D query differences also improve aggregate readout, but neither bypassing μ nor adding interactions repairs right-gripper large-positive underprediction. Neither result proves information-theoretic loss or absence. See [query comparison](query-accessibility512-note.md) and [interaction comparison](interaction-probe512-note.md).

The production posterior/decoder previously showed weak natural-z sensitivity despite substantial upstream query response; adapted heads/decoders improve posterior reconstruction. These observations separate representation, normalization, and decoder-training issues, but do not establish camera invariance or action semantics. The final-prior direct predictor's improvements remain posterior reconstruction; they are not action-conditioned deployable forecasts.

## Existing production supervision: do not call grounding new

Code inspection confirms that production already predicts a robot state delta from **current state plus z**: `vendor/lawam/latent_action_model/core/lam_model.py:474` calls `state_decoder(z_t=quantized, state_0=dec_states, embodiment_id=emb_tensor)`. `core/lam_lightinng.py:679–717` masks robot examples, uses `delta_proprio` and `eef_reconstruction_loss`, then adds `lambda_aux * state_loss`; the immutable final10000 config sets `lambda_aux: 1.0`. Thus neither adding measured-state supervision nor making its decoder context-aware is a new intervention relative to production. Current state alone can explain part of that auxiliary, which is why matched current-only physical-risk comparisons are essential.

The frozen-cache small-head harness lacks that full production auxiliary. Any8D pilot below is a **controlled harness experiment**, not evidence of improvement over an otherwise identical production loss. Before production integration, reproduce the original masked state-delta head/loss and keep its dimensions, weights, and inputs unchanged in G/B. The8D labels/reporting subset does not silently replace the production state target. If extra endpoint coordinates are needed for that reproduction, recover and validate them from existing NPZ state arrays before fitting; do not infer them from the8D probe labels. The actual candidate change is posterior alignment to an effect-conditioned prior, and its potential cost is removal of future content needed for Y.

## Three distinct quantities

Let `C` be current visual context (and explicitly declared current state where available), `Y` future visual features, `T` a measured physical effect, and `Z~qφ(Z|C,Y)` the posterior. Use train-only standardized T throughout.

1. **Future dependence:** q changes when Y changes with C held fixed. This admits camera motion, lighting, appearance, and leakage. A future-swap KL is a sensitivity diagnostic, not an action counterfactual.
2. **Available conditional predictive utility:** under squared loss, the ideal Bayes-risk reduction is

   `G_T = R*(T|C) - R*(T|C,Z) = E ||E[T|C,Z] - E[T|C]||²`.

   A camera-only code need not have positive gain when its nuisance variable N satisfies `E[T|C,N]=E[T|C]`. This is an assumption, not guaranteed by observational data. Camera movement correlated with manipulation can violate it. Finite probes only measure accessibility within their function class; weak or misspecified current baselines can exaggerate gain. Failure of a probe does not prove zero information.
3. **Actual decoder use:** `U_Y = E||Y-B(C)||² - E||Y-D_Y(C,Z)||²`, where B is a separately trained, matched current-only model. U is not the Bayes gain and may be negative. A new auxiliary physical decoder learning T from Z does not show that the visual decoder uses it. Require actual visual-risk improvement against both ordinary grounding and the matched constant-code model. In-distribution/near-current code sensitivity remains secondary; a large global-swap gap can be OOD sensitivity.

For the Markov relation `T — (C,Y) — Z`, task conditional-mean sufficiency means `E[T|C,Z]=E[T|C,Y]`. Its Bayes-risk gap is `R*(T|C,Z)-R*(T|C,Y)>=0`. This is weaker than full distributional sufficiency, does not identify actions, and cannot be certified by two imperfect probes. Full-query readouts are finite empirical references, not the Bayes oracle.

Subtracting a frozen baseline risk from an MSE loss adds a constant; it does not create a new learning algorithm. Conditional gain is the evaluation target. The intervention below changes which information is penalized and exposes the same decoder to two code-generating routes.

## Priority experiment: preserve measured effects, budget residual future information

Reuse frozen Q differences, the1024→256→64 Gaussian head, and the current strong fixed-baseline residual visual decoder. Train only small heads/decoder initially. Keep32D, logvariance clamp[-10,10], identical initialization/batches/posterior noise across arms. No new image extraction is necessary for this first gate.

Add a diagonal Gaussian `pψ(Z|C,T)`; retain/reproduce the existing context-aware effect decoder `D_T(C,Z)` in matched arms (a small replacement is only a cache-harness approximation). Both the visual and effect decoders receive C and Z, **never T directly**. The conditional code generator alone sees T. Context-aware effect prediction is important because the product-kernel experiment showed the inadequacy of assuming fixed linear coefficients of Z. A separate physical head is a training signal; visual use is still tested separately.

For labeled robot samples, draw `zq~qφ(C,Y)` and `zp~pψ(C,T)` using matched random numbers across experiment arms. Define

```
Lq = MSE(Y, DY(C,zq)) + λT MSE(T, DT(C,zq))
Lp = MSE(Y, DY(C,zp)) + λT MSE(T, DT(C,zp))
LK = KL(qφ(Z|C,Y) || pψ(Z|C,T))
L  = Lq + η Lp + β LK + β0 KL(qφ || N(0,I)).
```

The shared decoder learns both posterior and effect-conditioned routes. KL gradients reach q and p in the proposed arm; the critical control stops gradients only at q in LK. All arms retain the same original β0 and sampling route, so a mean-versus-sample or KL-removal change cannot silently explain results. A simple Gaussian prior is deliberately sufficient for the initial controlled test; flow is not needed to test this hypothesis.

The exact population decomposition, for the joint distribution induced by data and q, is

`E[C,T,Y] KL(q(Z|C,Y)||p(Z|C,T))`
`= I_q(Z;Y|C,T) + E[C,T] KL(qagg(Z|C,T)||p(Z|C,T))`.

Here `qagg` averages q over `Y|C,T`. This identity explains the intended pressure: penalize future-specific bits beyond C,T while fitting a conditional prior. It does **not** say those bits are useless. A restricted diagonal Gaussian can have substantial aggregate-matching error; minimizing the term can then distort useful codes rather than remove nuisance. KL zero permits a constant code, so reconstruction and effect losses must demonstrably retain conditional utility. Any claim about this information bound concerns sampled Z; it does not automatically constrain deterministic μ, which can encode distinctions in arbitrarily small mean differences.

## The important incompatibility with image prediction and human video

**Moving the head is real motion, not merely jitter.** Future camera pose can be necessary to predict Y. A decoder conditioned only on current C and a camera-invariant action code cannot reconstruct unpredictable future viewpoint faithfully. Without a separate adequately modeled camera/context route, suppressing that information can increase image-feature MSE. This is a concrete reason to reject the proposed budget if its semantic metric improves while visual risk worsens. We are not proposing to remove all global motion or claiming that this objective solves human egomotion.

Existing T comprises8 measured endpoint differences: source xyz/gripper coordinates, not commands. The adapter returns frames and proprioceptive endpoints (`vendor/dataset/robomind.py:104`), and no verified command chunk exists in these caches. Two different contact sequences or gripper trajectories may have the same endpoints. Rotations, object effects, contact forces, slip, and path timing are not covered. Penalizing information beyond T may delete exactly the information needed for fine manipulation or visual prediction. This failure is particularly plausible given the unresolved right-gripper tail.

On unlabeled human clips T is unavailable. The new terms cannot simply be applied there; retain the existing visual objective and mark the robot-grounded subset explicitly. Human pose/flow pseudolabels would introduce additional assumptions and noise, and would not be equivalent to the current measured T. Shared robot/human latent weights do not guarantee transfer. A positive robot-only pilot is insufficient for the main human-video claim.

Observational pairs alone cannot generally identify which variation is action and which is camera. Conditional camera correlation can remain useful for T, and separate factor branches can mix or collapse without anchors/interventions. WING's camera intervention is relevant, but adopting a crop/warp proxy does not model general SE(3), parallax, occlusion, or newly visible surfaces. A later camera stress test must use validated paired observations/interventions with preserved physical effects; it cannot be claimed from the present cache.

## Deployment boundary

`p(Z|C,T)` in this pilot is an **effect-conditioned training bridge**, not an executable action prior: its input T is measured after the transition. Evaluate its Y risk as an oracle-effect diagnostic, separate from posterior reconstruction. Neither is deployment prediction.

Once verified commands/chunks A are available, the executable branch must be `p(Z|C,A)` or a validated controller that turns requested effects into commands. Train that branch and the same decoder jointly, with future Y unavailable on the execution branch. The corresponding KL budgets `I(Z;Y|C,A)`, which also includes genuine stochastic action outcomes; suppressing it indiscriminately can hurt calibrated stochastic forecasting. Compare sampled-prior risk/distribution metrics and command-conditioned action ranking, not merely q/p mean agreement. No command-level claim is available from current endpoint labels.

## One minimal comparison, with rejection criteria

Use existing train512 and fixed observed64/96 only as a development experiment. Save the protocol before any fit. Four matched arms distinguish the specific intervention:

- **V:** visual objectives only; retain the auxiliary networks and computations for capacity accounting but no physical loss/conditional-KL gradient into q.
- **G:** ordinary effect grounding (already present in production) plus both decoder routes; LK trains p toward detached q only. This is the most important existing-method control.
- **B:** exactly G plus LK gradients into q: the proposed information budget. Keep p-gradient weight, decoder paths, data, initialization, and other losses identical to G.
- **Constant:** same current/decoder capacity with a trainable but example-independent Gaussian code and the same physical supervision. No T shortcut into decoder inputs.

For the initial implementation gate, fix400 updates/batch8/lr3e-4, seeds42/43/44, β0=5e-5 as in the existing harness. Set η=1, λT=0.1, β=5e-5 as explicit pilot values, not estimated optima. Any later adjustment must use train-only nested selection and a newly declared protocol, not these observed evaluation errors. Log physical/visual/KL gradient norms to detect an initialization gate blocking learning; report sample and μ routes separately. This is a small-head feasibility gate, not an end-to-end encoder claim.

Primary adoption condition: B improves **visual prediction risk and actual matched-control visual utilization** beyond G, while retaining or improving independently refitted, context-aware measured-effect utility. Report right-gripper static/large-positive bins using the existing fixed thresholds, and KL/prior-route mismatch. Do not accept lower KL alone, better fitted physical-head loss alone, or a larger swap gap. For human-video relevance, a later genuinely held-out human/moving-camera evaluation with usable transfer anchors is additionally required.

Reject the mechanism if B merely shrinks q variance/means, if the prior-only G matches B, if a stronger current-only model removes the gain, if camera/appearance predictability alone explains the effect, or if effect utility improves while visual MSE worsens. If q utility improves without DY risk improving, the encoder signal is useful but decoder utilization remains unresolved. If DY improves without physical utility, report predictive compression, not better action semantics. Larger query dimensionality and current–code interactions are separate ablations, not bundled into this first objective change.

## Why this candidate, and why not the alternatives

This candidate introduces a falsifiable pressure against **future content unavailable from the chosen condition**, while preserving measured utility and testing the actual decoder. Latent refinement only asks whether future-dependent code optimization can fit Y better; it also optimizes camera/appearance leakage. Earlier trust-region refinement showed limited local gains and no action guarantee. Refinement remains a useful optimization diagnostic, but it does not address the information-selection question.

Drop unconditional variance/InfoNCE maximization: it can reward episode/camera differences and says little about decoder use. Drop predicted-image→z recovery alone: a constant target or steganographic residual can satisfy it, and predictor-level recovery already exists. Drop simply enlarging z: raw queries improve aggregate probes but not right-gripper behavior, and larger capacity can carry more future leakage. Drop a standalone camera-invariance constraint as the first cache experiment: valid human camera interventions are absent, and camera motion may be required for Y. Drop the claim that subtracting current-only loss creates a new conditional-gain objective: it is the same gradient when the baseline is fixed.

## Primary-method checks and overlap

- [Sohn et al., Conditional Generative Models, NeurIPS2015](https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf), §4 Eq.4–5 and §4.2 hybrid objective: conditional posterior/prior KL and combining recognition/prior prediction routes are established. Our conditional bottleneck is not a new variational principle. This is the strongest novelty warning.
- [JEVA](https://singhularadi.github.io/jeva/static/jeva.pdf), §§3.2–3.5: Gaussian transition posteriors, shared visual/action likelihoods, sparse action grounding, and labeled mean routing are already developed. Ordinary grounding and unchanged sampling routes are essential controls; our pilot targets measured effects and a learned effect-conditioned reference instead of asserting that grounding itself is new.
- [Why Latent Actions Fail](https://arxiv.org/html/2605.20223v3), §4.2 Proposition4.2 and §§4.4–4.5: future exogenous leakage and robust-target/exogenous-swap remedies are explicitly analyzed under stated assumptions. The proposed budget does not inherit their identifiability or CCA guarantees, and current data lacks synchronized nuisance interventions.
- [AD-WM](https://arxiv.org/html/2609.30264v1), §III-C Eq.3–4: predicted-endpoint action recovery already regularizes actual dynamics, with action-ranking diagnostics distinct from factual MSE. Recovering z from generated output alone is neither novel nor sufficient for this proposal.
- [RLA-WM](https://arxiv.org/html/2605.07079v1), §3: residual-feature query compression plus a current/action-conditioned latent flow and shared decoder is established. Our investigation concerns whether training the posterior toward a conditional information budget helps; residual latents and compact flow are not new contributions.
- [WING](https://arxiv.org/html/2610.03607v1), motion routing/camera-intervention subsection, Eq.26: camera-compensated interaction representations and intervention consistency already exist. Camera separation cannot be claimed from simple reference subtraction or the proposed effect loss.

Evidence strength: mathematical decompositions are exact under the stated induced distributions and finite-risk assumptions; existing empirical results are finite fixed-data readouts/posterior reconstruction; the proposed objective's visual benefit, camera robustness, command compatibility, human transfer, and novelty remain untested. **If the goal is a deployable human-video prediction improvement today, this evidence is not yet enough to adopt it.** The justified next action is the bounded G-versus-B falsification test, not a broad new training campaign.
