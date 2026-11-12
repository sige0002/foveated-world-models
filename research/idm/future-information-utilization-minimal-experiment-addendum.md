# Addendum: separate accessible information from decoder use before changing the bottleneck

Status: design only; no training launched, no existing artifacts changed. This supplements [the conditional-information proposal](future-information-utilization-proposal.md). The next experiment should be a **controlled repair attribution**, not a new claim that conditional VAE training or physical grounding is novel. In particular, an effect-conditioned information budget can discard camera information needed for image prediction and cannot be applied directly to unlabeled human clips. Establish which trainable component is limiting the present visual objective first.

## Why this experiment now

Existing512 probes show useful information accessible in frozen reference μ beyond a strong spatial-current baseline. Allowing current–z interactions improves measured-effect prediction substantially. Neither replacing μ with1024D query differences nor adding those interactions repairs the right-gripper tail. Meanwhile changing the visual decoder with a fixed μ improves reconstruction. Therefore “z contains no future information” is already too broad; neither a small variance nor a weak additive readout can justify it.

The codec experiments address output compression, not whether the IDM lacks useful information. Per-patch codec w32 has8192 coordinates across256 patches and is distinct from global IDM z32. Future-encoded codec reconstruction and a predictor conditioned on posterior z remain reconstruction diagnostics. Do not mix their lower errors into an execution-time forecasting claim.

## Fixed starting point and inputs

Use the existing train512, observed dev64/96, frozen2900 Q differences, fixed128 calibration, reference Gaussian head42, and **the completed fixed400 direct prior-residual decoder42** on the final10000 current-only baseline. Do not choose a new starting checkpoint based on the latest observed errors. Freeze the vision/query encoder and baseline throughout. Only the small IDM head and/or correction decoder can update. This identifies head/decoder optimization effects, not the entire production encoder's capacity.

The head consumes `Q(C,Y)-Q(C,C)`; the decoder consumes current C, its fixed current-only baseline, and μ. Future Y enters the IDM and reconstruction target, not a decoder bypass. Keep the already successful μ route for every arm: adding posterior sampling here would confound the diagnosis with a distribution shift relative to the pretrained decoder. Consequently conclusions concern μ-based reconstruction, not the information capacity of a single sampled Gaussian z.

Prespecify CPU float32 `highest` for the small-head/decoder experiment and minibatch8. Recompute the untouched starting point using that same path. Record its difference from historical cached-μ evaluation; do not demand that a different batch/precision path reproduce historical MSE exactly or choose precision by evaluation performance. All arms must start with exactly the same newly computed μ and decoder outputs. Historical source Q was extracted using its recorded medium/B1 encoder path; do not re-extract it.

## The minimum repair comparison

Let H0,D0 be the common completed start; Hθ changes μ, and Dω includes the learned residual correction while the current-only baseline remains fixed.

| Arm | Trainable component | Purpose |
|---|---|---|
| F | None | Untouched common baseline; evaluation only |
| H | IDM head only; decoder exactly D0 | Can a shared learned encoder map recover better codes for the existing decoder? |
| D | Decoder only; μ exactly H0 | Can the existing representation support better visual reconstruction? |
| HD | Both head and decoder | Is joint adaptation needed beyond either isolated repair? |
| K | Decoder only; fixed train-mean μ for every example | Matched current-only continuation; separates extra decoder optimization from using transition-specific code |

H, D, HD, K each receive400 updates, batch8, AdamW lr3e-4, the same train index stream. Use seeds42/43/44 for minibatch streams while all start from the same frozen H0/D0 weights; these are optimization-stream replicates, **not independent pretrained-IDM seeds**. Do not mix them with previous head-seed replication. Different trainable subsets have different cost/parameter counts intentionally; report wall time and trainable counts rather than claiming equal FLOPs.

No rank, flow objective, horizon feature, spatial basis, physical target weighting, or output codec change is bundled in this comparison. K uses the train mean once, not the mean of an evaluation batch. Its initial loss may differ from F because the code changes; report that initial loss, then compare completed D and K as matched continuation controls.

## Training loss and gradient routing

For H and HD, use mean reconstruction plus a fixed drift penalty:

```
Lvisual = mean || Dω(C, μθ(C,Y)) - Y ||²
Ldrift  = mean_examples 0.5 * sum_dimensions
          ((μθ(C,Y) - stopgrad(μ0(C,Y)))² / exp(logvar0(C,Y)))
L = Lvisual + 1e-3 * Ldrift.
```

Freeze the reference logvariance at its existing[-10,10] clamp. The updated Gaussian for this diagnostic is defined with μθ and **the frozen reference variance**; disregard the trainable head's unused logvariance output. Then Ldrift is exactly `KL(N(μθ,Σ0)||N(μ0,Σ0))`. It is an explicitly local repair guard, not a new conditional prior, an action-identification loss, or a claim to preserve all posterior behavior. The weight1e-3 is a prespecified pilot setting, not an optimum. D and K have zero mean drift and train with visual MSE alone.

For H, set decoder parameters `requires_grad=False` but **do not use no_grad around decoder forward**: gradients must reach μ through D0. For D/K, detach μ and train the decoder. For HD, propagate through both. Optimizers must contain only intended trainable parameters, with all four arms given identical weight-decay conventions. Check clipping norm finiteness, unchanged frozen tensor hashes, and head/decoder MSE-gradient norms at fixed steps1/10/100/400. Freeze masks must not accidentally suppress the head gradient or update unused reference variance.

Report train and evaluation drift distributions. If mean train drift exceeds1 nat or any nonfinite gradient occurs, flag the pilot as failing its local-repair scope; do not choose a lower-error checkpoint or change the weight using observed64/96. A new predeclared training-only protocol would be required. This rule is a scope gate, not a proof of distributional support.

The mathematical KL guards sampled distributions, while optimization/evaluation here uses μ. Small KL alone does not prove a deterministic mean has little information or that the decoder cannot amplify it. Also, coordinate rescaling and co-adaptation can change finite-readout accessibility without adding information. The drift guard limits but does not eliminate that interpretation.

## Exact relationship to the head's original objective

`train_reference_heads.py:85–91` trained the reference42 head with `z=μ+exp(0.5 logvar)ε` and `MSE + 5e-5 * mean(sumdim KL(q||N(0,I)))`. The subsequent strong prior-residual decoder was trained on exported **μ**, not posterior samples. Those two objectives are different already.

For this repair experiment choose μ-only forwarding to match the actual pretrained decoder, freeze reference variance, and use the drift KL above. **The original standard-normal KL is not optimized in this primary test**; log it diagnostically for all changed heads. This is a local reconstruction adaptation, not continued maximization of the original VAE objective and not a new collapse-prevention ELBO. Keeping a learnable logvariance while only feeding μ and penalizing KL would allow variance changes unrelated to decoder use, so the reference variance is fixed explicitly.

If the intended next task is production Gaussian-posterior training rather than this repair diagnosis, use a separate preregistered experiment that restores sampling plus the original5e-5 standard-normal KL in every applicable arm, gives the starting decoder the same sampling adaptation budget, and reports μ and sample risks separately. Do not merge that change into H/HD and then attribute the result solely to updating the head. JEVA's routing result is precisely why this choice cannot remain implicit.

## Concrete cache and trainer reuse

No new image decode, vision forward, physical labels, or GPU is needed to implement this specification:

- Feature/current-only-prior data: `runs/reference-idm42-nested512-prior10000/{train,validation}.pt`; observed evaluation: `runs/reference-idm42-observed96-prior10000/validation.pt`.
- Train Q: concatenate old128 `research/idm/reference-query-full/train.pt` with added384 `runs/reference-idm42-nested512-prior10000/additional_queries.pt` in fixed selection order. Dev Q: `research/idm/reference-query-full/validation.pt`; observed96 Q: `research/idm/observed96-reference-queries2900/validation.pt`.
- Calibration: `research/idm/reference-query-full/train_calibration.pt`, unchanged. Reuse `train_reference_heads.representation(...,'reference')` and load only `head.*` from the completed `research/idm/reference-comparison2000/reference-42.pt`.
- D0: `runs/reference-idm42-nested512-comparison400-20261113/prior_residual-seed42.pt`. Instantiate through `research.train_cached_decoders.make_decoder('prior_residual',128,2,768,32,16)` and load strict state. The latest codec capacity results motivate the question but are not used to switch this starting checkpoint after specifying it.
- Reuse source/row/head/calibration verification logic from `probe_query_accessibility512.py` and `verify_nested512_cache.py`; use full source-file hashes plus episode/index/dt/optional endpoint identity. **Legacy128 rows lack endpoint_indices.** Check real cache schemas before the first optimizer step, and verify all three episode sets are disjoint and share the final10000 prior checkpoint/weight hashes.
- Reuse immutable batch schedules, initial/final parameter hashes, and finite-gradient checks from `train_postprojection_heads.py`; reuse the separate gradient diagnostics from `train_joint_objectives.py`. Do not reuse their34D head, sampled objective, or flow branch inadvertently. Reuse original evaluation arithmetic but compute each arm's current μ from Q at its declared precision/batch path instead of blindly loading original codes.

Implement in a **new dedicated research/idm trainer**, leaving running shared trainers unchanged. Before any fit, a synthetic test should prove: H retains a nonzero gradient through the frozen decoder while D0 tensors stay bitwise fixed; D has no head gradient; HD updates both; K ignores per-example query; unused variance does not drift; at step0 all non-K arms have identical outputs. A real-cache preflight must verify512/64/96 row alignment, train-only calibration, checkpoint provenance, finite features and posterior parameters, and report initial risk without using it to change settings. Save the protocol before fitting and fixed400 results/per-row errors for every arm, not only the winner.

## Measure three things separately

**A. Future responsiveness.** Use indices `[0,9,18,27,36,45,54,63]` from the **step2900** `reference-query-full/validation.pt` query/future_swapped/reference fields, retaining each current reference. Do not feed the separate6501 projection diagnostics into this2900 head. On these fixed same-current future-swap pairs, measure μ/logvariance differences, posterior KL using the frozen variance, and native-direction decoder finite differences. These detect sensitivity, including camera/scene leakage. A swapped pair need not be a physically realizable action counterfactual. Do not optimize this metric or rank models by it.

**B. Accessible utility of the representation.** Independently refit the already declared spatial-current plus linear-code **interaction** kernel probe for H0, H and HD. Keep the exact512 train-only five folds, alpha grid, target scales, and fixed right-gripper bins. D has identical H0, so its representation probe should match exactly and need not be refitted. Evaluate measured8D effects and a same-capacity current-only control. A gain is evidence of predictive accessibility under this function class, not conditional mutual information or action identity. A negative result cannot establish absence of information. Do not substitute the co-trained state decoder's own training error for an independent representation evaluation.

For squared loss the ideal population quantity is

`G_T = R*(T|C) - R*(T|C,Z) = E||E[T|C,Z]-E[T|C]||²`.

Our finite probes do not equal these Bayes optima. The existing raw-query probe is a useful richer reference but not an upper bound or oracle. On unlabeled human data this physical-utility channel is unavailable; report that absence rather than manufacturing a semantic score from visual reconstruction.

**C. Actual visual decoder use.** Evaluate full-feature MSE for F/H/D/HD/K under the same forward precision, source inputs, and fixed endpoint. The key D-versus-K difference asks whether continuing the real decoder with transition-specific μ helps beyond continuing the same-capacity current-only model. H-versus-F asks whether an amortized head repair helps the existing decoder. Global z swaps are secondary because they can create current/code or horizon/code incompatibility.

A finite decoder gain is not equal to G_T. Auxiliary physical-probe improvement does not show that the visual decoder uses that information. Conversely, visual improvement may represent camera/appearance prediction without improved physical grounding. Both outcomes are informative but require different claims.

## Interpretation and falsification table

- **D beats K and F; H provides little gain:** fixed μ already supports additional visual utility, and decoder optimization/access is a credible bottleneck. Do not conclude the encoder is optimal.
- **H improves F at bounded drift, with independent physical utility retained/improved:** shared head adaptation can supply more useful codes to the existing decoder. This is an encoder/readout repair, not a proof of missing information in raw Q or a new latent-learning principle.
- **HD alone improves:** the benefit may require co-adaptation; do not assign it uniquely to encoder information or decoder use. Inspect probe and drift results rather than inferring causality from the endpoint alone.
- **D and K improve equally:** the gain is explained by current-only decoder continuation at this budget; reject the claim that extra use of transition-specific z caused it.
- **Only future-swap sensitivity or drift increases:** reject the repair as unsupported. More future dependence is not the goal.
- **Visual MSE improves but measured effect utility deteriorates, or vice versa:** report a tradeoff. Do not label this a general improvement in action representation or world-model accuracy.
- **No arm improves:** reject this finite-budget local repair as a next training recipe. It does not prove that useful information is absent or that no larger optimization could work.

All64/96 episodes have already been observed repeatedly. Confidence intervals, if computed, are conditional descriptive comparisons; they do not repair adaptive reuse. This experiment chooses a mechanism to investigate, not an independently validated final architecture.

## Existing auxiliary and earlier proposal: exact boundaries

Production `lam_model.py:474` already calls `state_decoder(z_t=quantized, state_0=dec_states, embodiment_id=...)`. `lam_lightinng.py:679–717` already adds masked robot `delta_proprio` reconstruction; the final config uses lambda_aux1.0 and l2. Thus adding state-delta grounding or making it context-aware is not new. Its current-state input can itself explain labels, so its loss alone is not a test of z utility.

The present small-cache decoder/head comparison deliberately isolates visual MSE and is **not a full production-loss reproduction**. Production integration must retain and compare the original auxiliary with the same mask, dimensions, weights, and current-state inputs. An8D probe target is not a drop-in replacement for that production target. Do not claim improvement over a matched production objective before that separate experiment.

The earlier effect-conditioned-prior proposal remains a later, riskier hypothesis. This addendum does not combine it with the repair factorial. Learned conditional priors and training a prior branch on targets are already CVAE/GSNN constructions. Moreover, the real camera motion needed for Y can be penalized by a physical-effect-conditioned rate budget. If this local experiment identifies a decoder bottleneck, improving that use may be more justified than deleting future information from the IDM.

## Human videos, moving cameras, and deployment

The visual repair loss works without physical-effect labels, so its computational mechanism can be tested on human videos once appropriate cached pairs exist. The current cached512 experiment is robot-only; no human-transfer conclusion follows automatically.

Camera movement can be useful for forecasting the next view. Removing it without a separate accurate observer-state route may increase error. Future sensitivity and reconstruction cannot distinguish camera information from action information using generic observational pairs alone. Paired viewpoint interventions, verified physical targets, or other identifiable structure would be needed for the stronger claim. Optical flow or pose pseudolabels bring estimator and camera assumptions; they are not free ground truth.

All experiment arms infer μ from the true future. There is no verified command-conditioned code generator in these caches, and measured endpoint deltas are not commands. Deployment requires a separately evaluated current/action-conditioned code route and a decoder exposed to that route; neither reconstruction nor successful probes substitute for this test. Same-endpoint motions may differ in contact, trajectory, timing, and grasp success.

## Primary-method verification and novelty assessment

- [Lagging Inference Networks, ICLR2019](https://arxiv.org/html/1901.05534v2), §§3–4 and §5: inference lag and aggressive inference updates are established explanations/repairs; §5 distinguishes optimization lag from amortization-gap methods. Our H-only arm is a controlled shared-head repair, not their adaptive algorithm and not a novel principle. This source supports testing optimization allocation, not assuming action/camera identifiability.
- [JEVA](https://singhularadi.github.io/jeva/static/jeva.pdf), §§3.4–3.6, Eq.14–15: sparse action grounding, posterior-mean routing on labeled data, and coupled prediction/grounding objectives are already explicit. Our decision to retain the μ route controls an existing factor; it is not a new sampling proposal. The present8D measured-effect probes are not equivalent to its command-chunk decoder.
- [Sohn et al., NeurIPS2015](https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf), §4 Eq.4–5 and §4.2 Eq.8–9: conditional posterior/prior training and CVAE–GSNN hybrid objectives are established. The paper explicitly distinguishes recognition-path reconstruction from prior-path prediction. A conditional-prior extension must be compared as a known construction, not advertised as novel merely because both branches are trained.
- [Why Latent Actions Fail](https://arxiv.org/html/2605.20223v3), §4.2 Proposition4.2: under its linear-model assumptions reconstruction forces future exogenous information into the latent. This is a direct counterexample to treating stronger future dependence as better action semantics. Its robust-target and exogenous-swap analyses do not give our observational cache camera separation guarantees.

**Research value:** an experimentally reliable separation of “sensitive to future,” “accessible task information,” and “used by the actual predictor” could support a broader mechanism study with interventions and deployment evidence. This finite factorial, known trust-region penalty, and existing probes alone do not establish top-conference novelty. The justified deliverable now is a reproducible, falsifiable decision about the next training component to change.
