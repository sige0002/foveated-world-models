# Distinguish future encoding, decoder use, and action-specific utility

The current query experiment measures a **conditional-centering intervention**. Q(x,x) is a zero-transition input that may be outside the training distribution. Subtracting it does not identify motion, action, or camera-invariant features. Even if subtraction improves held-out reconstruction, those stronger interpretations require separate tests.

## A mathematical target with explicit assumptions

Let X be current information, Z an inferred latent, and Y a target with finite second moment. Under squared error, define the Bayes predictors m(X)=E[Y|X] and f(X,Z)=E[Y|X,Z]. Orthogonality of conditional expectation gives

`R(m) - R(f) = E ||f(X,Z)-m(X)||² = Γ_Y >= 0`.

This is a predictive-utility identity, not a causal identification theorem. If Y is a future visual feature, Γ_Y may be entirely camera, lighting, or scene information. Replace Y by measured robot action/increment A to define Γ_A. Finite trained probes only approximate these Bayes risks; a larger/currently better-trained head can create an apparent gap.

If Z' is drawn independently of (Y,Z) from the **true** conditional distribution p(Z|X), a Bayes-optimal decoder satisfies

`E ||Y-f(X,Z')||² - E ||Y-f(X,Z)||² = 2 Γ_Y`.

Proof: conditional on X, the wrong-code predictor and Y are independent, while f(X,Z) is the conditional mean. The wrong-code risk adds Var(f|X) to Var(Y|X); the correct risk subtracts it. This identity fails as an interpretation if the decoder is not Bayes-optimal, the sampler is wrong, or arbitrary cross-scene swaps are used. It supplies a falsifiable calibration target in a simulator with known distributions, not a new statistical theorem.

## Why ordinary swaps are confounded

The [conditional permutation test paper](https://arxiv.org/pdf/1807.05405), sections 2.1–2.2 and robustness analysis, explains why unconditional permutation breaks dependence with conditioning variables. Within-bin permutation is not generally an exact conditional-independence test. Validity depends on the conditional distribution approximation; nearest-current swaps alone do not justify formal p-values.

[Why Latent Actions Fail](https://arxiv.org/html/2605.20223v3), sections 3–4, gives a reconstruction-based nuisance problem under its linear/exogenous assumptions. Human head motion can correlate with the action, so marginal independence from camera is not a universally appropriate target. [ACT-LAM](https://arxiv.org/html/2609.15189v1), section 4.3, already studies latent utilization. [On Identifiability of Latent Action Policies](https://arxiv.org/html/2510.01337v1), Example 3 and theorem, requires coverage/structural assumptions beyond reconstruction. A swap metric by itself is not a new contribution or certificate.

## Feasible audit on existing data

1. **Global swaps as an upper stress test.** Keep the currently measured result, explicitly named a dependency stress test. Report current-state distances between each source/destination pair. It is not the main estimate of action utility.
2. **Current-matched swaps.** Build neighbor distances using only current visual features plus available current robot/object state. Choose feature preprocessing and radii on train data only. Match embodiment/task and exclude same-episode temporal neighbors. Restrict swaps to small-radius matches and report radius, match rate, and remaining action diversity. Plot the risk gap against radius rather than declaring one nearest-neighbor radius sufficient. Tight radii with no alternatives mean insufficient coverage, not a negative action result.
3. **Separate pseudo-counterfactual construction.** Evaluate (a) swapping latents from real nearby transitions, and (b) recomputing Q(x_i,y_j) at the fixed current. The latter creates a synthetic mismatched transition and can be more OOD. Agreement is informative; neither is a ground-truth action intervention.
4. **Learn conditional resampling only if data supports it.** Fit p(Z|X) on train folds and assess held-out conditional calibration, distance/range coverage, and failures by state region. Use held-out folds for the utility statistic. A model trained on 128 examples in a high-dimensional current space is unlikely to justify exact CRT claims; report sensitivity to sampler choice. Current-neighbor swaps remain descriptive when calibration is not credible.
5. **Measure Γ_A separately.** Recover aligned robot labels from the original dataset indices; compare matched-capacity current-only and current-plus-Z predictors of actual action or joint increments. Include current state in both. Report held-out episodes, within-current-neighborhood results, and small-motion bins. Repeat after shuffling Z conditionally. Future-feature reconstruction gains without action-label gains support a future-information shortcut, not action learning.
6. **Test visual nuisance with paired interventions.** On recorded/simulated transitions with the same physical action and state, change only the view using multiview data or calibrated rendering. Measure action decoding and interaction-effect predictions across those views, with occlusion/observability stratification. Synthetic translations are a limited stress test, not real 6D head-motion invariance. If camera motion is behaviorally coupled, test controlled changes rather than demanding zero marginal correlation.

## Negative controls and kill criteria

- Construct synthetic data where future appearance noise is independent of action but visible to IDM. A valid action audit should allow Γ_Y>0 while rejecting Γ_A>0. If it labels this a better action model, the audit fails.
- Construct a z that encodes only scene identity. Global swap gaps may be large; current-matched/calibrated results should reduce that false signal.
- Construct z=A with known dynamics and varied camera nuisance. The audit should detect action utility and its loss under conditional action swaps.
- In a resettable environment, compare measured conditional-swap risk with the 2Γ identity using known conditional distributions and a fitted decoder. Disagreement should track decoder/sampler error; unexplained disagreement falsifies the interpretation.
- For reference centering, reject an action claim if visual reconstruction improves while Γ_A estimates do not, or if gains vanish after camera/scene-controlled evaluation. Do not tune the audit on the same held-out examples used to claim the improvement.

## What could constitute research value

A possible contribution is a calibrated three-stage audit (future dependence → decoder use → action-specific incremental utility), together with an intervention that improves the final stage and downstream control. The proposed reference subtraction is only one candidate intervention. The conditional-risk identities and conditional-permutation ideas are established tools. Novelty would have to come from demonstrated failure modes of existing latent-action evaluations, a practical calibration/coverage procedure, and replicated action/control gains beyond ACT-LAM/WING/JEVA-related baselines. None is established by the current small-cache experiment.

An additional control for the zero-transition reference is an independently cross-fitted predictor of E[Q(X,Y)|X], trained only on real transitions. Subtracting this estimate avoids feeding the encoder Q(x,x), but removes behavior predictable from X as well as nuisance and may overfit on a small cache. It represents an innovation relative to the behavior distribution, not an action definition. Comparing it with global centering and Q(x,x) would reveal whether the observed gain specifically needs the synthetic no-change reference. This control is proposed only; it has not been trained or validated here.
