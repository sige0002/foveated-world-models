# Local repair: independent utility and decoder sensitivity addendum

2026-11-13. This adds interpretation after all 12 fixed fits and all independent probes/swaps completed. It does not change their protocols, checkpoints, or selection rules. No further training was performed for this note.

## Decision

Decoder-only continuation (D) is the clearest positive result: visual reconstruction improves with exactly unchanged codes and unchanged measured-effect probe predictions. The additional benefit of changing the IDM head is unresolved. All H and HD runs fail the prespecified train-mean posterior-drift gate of 1 nat; small HD improvements do not override that failure. Do not adopt HD as a successful *local* IDM repair.

F is the frozen common pretrained head/decoder; H updates the mean head only; D updates the decoder only; HD updates both; K updates the decoder with the fixed training-mean code. All three seeds share pretrained H0/D0 and differ in optimization streams, rather than constituting independent pretrained representations. The trained path uses deterministic means, with frozen reference variances for drift measurement, not posterior sampling or continuation of the original VAE objective.

## Completed observed96 results

Values for H/D/HD/K are arithmetic means of individual model risks, not ensemble predictions. Physical probes use the same train-only five-fold selection, interaction kernel, old128 target scales, and fixed measured endpoint displacement labels. Every independently fitted probe selected spatial RBF multiplier 1 and alpha .01, an interior value. D codes are exactly F on train512/dev64/observed96, so D uses the identical F probe without refitting. Historical zero-code CV and evaluation reproduction passed.

| Arm | Visual feature MSE | Measured-effect normalized MSE | Right-gripper coordinate MSE | Local-drift gate |
| --- | ---: | ---: | ---: | --- |
| Current-only probe | — | .402629 | .076725 | — |
| F | .085572 | .275594 | .080620 | Reference |
| H | .085189 | .277415 | .081913 | Fail 3/3 |
| D | .083170 | .275594 | .080620 | Unchanged head |
| HD | .082970 | .271727 | .079863 | Fail 3/3 |
| K | .094909 | — | — | No head adaptation |

H reduces visual MSE by about .45% relative to F, but its observed96 effect-probe risk worsens in all three seeds. Its dev64 effect-probe risk improves, so the physical result does not transfer consistently between these already observed evaluation sets. HD improves the aggregate effect probe by about 1.40% relative to F, while its visual risk improves only about .24% relative to D. HD's visual difference from D changes sign for seed43. No physical-probe uncertainty interval or population-level significance claim is established by these means.

The saved conditional paired visual intervals for HD minus D are [-.000779,-.000037], [-.000363,.000919], and [-.000955,.000026] for seeds42/43/44 respectively. These intervals condition on fitted models and the observed evaluation data; they do not include research selection uncertainty. Only one excludes zero. In contrast, all three saved D-minus-K intervals exclude zero, supporting transition-specific code utility beyond the particular constant-code continuation control. This does not identify action semantics or exclude camera/appearance information.

The right gripper remains worse with HD than with the strong current-only probe. Its small improvement over F does not solve the previously identified weakness. These targets are measured endpoint state differences, not action commands, joint angles, or contact trajectories; the gripper coordinate has dataset-defined scale rather than a verified physical aperture unit.

## Information versus use: the swap counterexample

For the same eight fixed current observations, replace the encoder's future input with the predetermined alternate future. With F, the decoder output difference MSE is .031095. With D42/43/44 it becomes .048465/.052012/.050372, even though the posterior means, reference variances, and all independent probe predictions are exactly unchanged. Thus a larger decoder swap response can arise entirely from decoder adaptation; it is not evidence that more future information entered the encoder.

All these swaps are exploratory out-of-support sensitivity measurements, not physical interventions or a conditional independence test. The fixed eight examples are not the full evaluation risk. The large symmetric posterior KL for F (2033.18 mean) reflects this fitted reference head, its small variances and swapped inputs; it must not be equated with action information or compared causally with the original production6501 head's much smaller KL. H and HD do not monotonically increase this KL, despite HD's modest probe improvement. Reference log-variance remains a frozen function of each input, so it can differ between true and swapped inputs without being learned during repair.

## Why the soft constraint failed and what remains justified

The completed [drift analysis](local-repair-soft-drift-and-hard-bound-addendum.md) shows train-mean drift 5.61–7.95 nat across all six H/HD fits, rather than the allowed 1 nat. The upper 5% of examples account for approximately 67–75% of drift, but removing the five largest still leaves means 3.61–4.23. This is not explained by one pathological example. The initial MSE gradients reach the head, while the quadratic drift gradient is zero at its anchor as expected; subsequent penalty gradients can dominate MSE gradients. A nonzero soft penalty therefore neither proves a pointwise constraint nor satisfies the observed train-mean gate.

A later hard-bounded residual head is a mechanism test, not a novel algorithm:

`mu = mu0 + sigma0 * sqrt(2*delta/32) * tanh(g(q))`, with frozen `mu0`, `sigma0`, and a zero-initialized final layer of `g`.

For fixed diagonal reference covariance, `KL = delta/32 * sum(tanh(g)^2) <= delta` in exact arithmetic. Numerical tolerance, finite values and initial equality must be checked. The coordinate-wise box is more restrictive than a Mahalanobis ball, and the changed architecture requires a matched unbounded residual-head control. A joint decoder can amplify small mean changes; this bound does not guarantee semantic preservation, removal of camera information, or retained information content. The current results justify keeping this as a possible controlled follow-up, not starting it or calling it a top-conference contribution.

## Scope and evidence

These are posterior reconstructions: the encoder sees the true future to obtain the code. Neither the visual scores nor physical probes demonstrate executable action-conditioned future prediction. Observed96 and dev64 have already informed this research, so neither is an independent final test. A finite probe only measures accessibility under its specified readout, not conditional mutual information or the absence of additional information. Unknown human camera motion and unlabeled physical effects remain unresolved.

Source artifacts: [visual results and paired intervals](local-repair512-summary-20261113.json), [independent effect probes](local-repair-probes-20261113/results.json), [fixed-current future swaps](local-repair-swaps-20261113/results.json), and [all-seed drift summary](local-repair-soft-drift-all-seeds-summary.json). The new [descriptive aggregation](local-repair-independent-utility-aggregation.json) records SHA256 hashes of all four sources and arithmetic group means. It adds no model selection or new statistical test. The [original implementation specification](future-information-utilization-minimal-experiment-addendum.md) and [prior-method review](future-information-utilization-proposal.md) retain the existing grounding/CVAE comparisons; this addendum asserts no new method novelty.
