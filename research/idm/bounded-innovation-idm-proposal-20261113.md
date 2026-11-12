# A bounded innovation channel for IDM: proposed mechanism test

2026-11-13. Design only; no training or changes to running early-condition experiments. This is one upstream head intervention, not a claim that another variational bottleneck is novel.

## The unresolved question

Existing local repair shows that decoder-only adaptation improves reconstruction with **exactly unchanged codes**. H/HD changes violated their soft drift gate; HD's small physical-probe gain does not establish missing information. Raw queries and interaction probes change readout accessibility, not measured conditional mutual information. The newly measured frozen final-LayerNorm geometric floor (.002263 on observed96, approximately2.17% of prior MSE; parent audit) also leaves most error unexplained.

Test a narrower hypothesis: **the fixed query difference contains future-dependent predictive detail that the current32D code/readout combination does not use, and a small additional stochastic channel can transmit useful detail at a known information upper bound.** A positive result is incremental predictive utility under specified readouts, not proof that old z lacks the information in an information-theoretic sense.

## One intervention, with a hard sampled-channel budget

Keep the old encoder, its calibration, and reference42 `z0=mu0(C,Y)` fixed. Here C,Y denote raw observations (or observations sufficient to determine all query inputs), not automatically the decoder cached DINO features. The loss target is phi(Y), the cached future visual representation. Let `S=(Q(C,Y)-Q(C,C))/s_old` and `A=Q(C,C)` be the already cached1024D query inputs, with the frozen old scalar reference calibration s_old. Subsequent train-only coordinate normalization largely cancels that scalar except at its floor. Add an eight-dimensional head with a hard bound:

```
m_theta = sqrt(2*delta/8) * tanh(g_theta(A, z0, S))
W = m_theta + epsilon,       epsilon ~ N(0,I8)
delta = 1 nat               (fixed pilot, not selected on evaluation)
q_theta(W|C,Y) = N(m_theta,I8).
```

Then every example satisfies `KL(q_theta || N(0,I8)) = ||m_theta||²/2 <= delta`. Conditional on `B=(C,z0)`, the standard variational identity gives

```
E KL(q(W|C,Y) || N(0,I))
  = I(Y;W|B) + E_B KL(q(W|B) || N(0,I)) >= I(Y;W|B).
```

The distributional statement conditions on fixed learned weights (and frozen train-only calibration); it does not bound information carried by the learning algorithm or its training set. Thus the **additional** sampled-channel conditional information is at most1nat, including held-out inputs, provided the frozen query/old code are deterministic functions of C,Y and the decoder sees only W rather than its mean, random seed, or encoder activations. This is a distributional guarantee for fresh independent evaluation Gaussian noise conditional on learned weights and calibration, not a measurement of mutual information from four observed draws. The old deterministic z0 is not bounded by this argument. No learned variance or soft KL penalty can evade the additional bound. Eight is an implementation choice at fixed rate, not a claim that eight dimensions are optimal. One nat (about1.44bits) is a deliberately restrictive, interpretable information allowance and matches the numerical scale of the previous1nat scope gate, although these are different KLs with different meanings. No theorem makes1nat optimal or guarantees detectable MSE gain at512 examples. The purpose is to decide whether to investigate extra predictive information or finite readout, not to promise a large accuracy improvement.

Use the completed fixed400 direct prior-residual decoder42 as the prespecified starting decoder, not whichever running experiment later wins. Expand its code input32→40, copying the existing32 columns and initializing the new8 columns to zero. All arms share those weights. Continue the correction decoder, freeze the current-only prior, and train the new head. An implementable head is a2080→128→8 MLP (A1024,z032,S1024), GELU, ordinary nonzero seeded initialization in **all** head layers. Only the new decoder columns start at zero. Normalize input blocks using train-only statistics; do not refit the old query calibration or z0. Record head/input-gradient norms: With an additionally zero final head layer, initial new-column gradients would be noise-correlations with zero population expectation, so that symmetric stalled design is rejected. A nonzero head mean lets new columns receive signal-dependent gradients at initialization. Head gradients initially vanish because the new columns are zero and must be checked after updates, using an isolated copy rather than mutating fit initialization. Failure to open this path is an optimization failure, not evidence of no information.

The exact objective is solely

`L = E_train,epsilon mean_features ||D_omega(phi(C),z0,W)-phi(Y)||²`.

The hard architecture enforces the budget; no posterior-drift term, auxiliary state supervision, or negative current-only loss is added. Subtracting a fixed baseline risk would not change this gradient. This tests a progressive IDM code, not another unconstrained relocation of mu0.

## Matched controls and actual gain

Three predeclared arms use the same40D decoder input,2080→128→8 head allocation, initial decoder/head weights, training indices, independent noise stream shared across arms, and update count:

| Arm | Third1024D head-input block | Meaning |
| --- | --- | --- |
| Future innovation | Actual S | Additional access to the frozen future query |
| Recoding control | T z0, with fixed seeded orthonormal-column1024×32 T | Additional nonlinear readout capacity, but no future input beyond existing z0 |
| Noise control | Any placeholder, head output forcibly zero | Same decoder continuation with independent noise only |

All arms also receive A and z0. A may expose current information beyond the decoder features; the zero-extra-future-information argument for recoding conditions on raw C, not necessarily only those features. Under the recoding control, `W` is conditionally independent of Y given `(C,z0)` exactly (for independent sampling); a gain here can only reflect finite-readout/optimization effects, not extra conditional information. The future-minus-recoding contrast is the key. The same nominal network size does not make effective input complexity equal, so even this contrast is not an information-insufficiency proof. Record the original fixed-code D continuation as context, not as a perfectly matched noise/architecture control.

Prespecify400 updates,batch8,lr3e-4,AdamW and clip1, three optimization seeds42/43/44, on the existing512 pairs. Save train-only normalization and protocol before any fit. Evaluate the already observed64/96 at the endpoint. Primary risk is **single-sample prediction risk**: average losses over four independent draws for Monte Carlo precision, not MSE of the average prediction. Averaging predictions would give the predictor multiple channel uses (potentially up to4nat); decoding the posterior mean removes this sampled-channel bound entirely. Use both means only as explicitly unbounded secondary diagnostics, if desired, never as the primary claim.

For an ideal Bayes predictor of the visual target, adding W reduces square risk by

`E || E[phi(Y)|C,z0,W] - E[phi(Y)|C,z0] ||²`.

Actual fitted-decoder risk differences need not equal this quantity. Report paired episode differences and seed variability, plus train/evaluation KL distributions and mean saturation. Do not substitute swap sensitivity or an independently fitted physical-probe score for this visual gain.

## Inputs and limits

Posterior training/reconstruction uses C,Y to compute S and z0; the decoder receives only C,z0,W and the current-only prior. This needs no robot-state labels and can therefore be formulated for human videos. It does **not** remove camera motion: one nat might encode viewpoint rather than manipulation, and viewpoint can be necessary for image prediction. Existing physical probes and camera stress tests can diagnose what changes, but cannot turn this construction into action identification.

At real forecasting time Y is unavailable: neither z0 nor W can be extracted this way. A verified command-conditioned prior for both codes is required before claiming executable prediction. Drawing W from N(0,I) is a declared zero-extra-information baseline, not equivalent to the learned posterior. Current data has endpoint effects, not verified action commands; no deployment improvement is claimed. New held-out human/camera data and a proper command route are future requirements, not satisfied by these caches.

## Rejection and prior art

Reject the pilot's useful-additional-channel hypothesis if the future arm fails to beat **both** recoding and noise controls in prespecified visual risk, or if improvement exists only for deterministic means/multi-draw prediction averages. If it improves train risk alone, report overfitting. A failed finite noisy head does not prove sufficiency of z0: the1nat budget, query representation, optimization, or decoder may be limiting. No post-hoc rate sweep or extra training is authorized by this proposal. If future wins only over noise but not recoding, pursue decoder readout rather than new IDM information.

The relevant bound is established variational-information-bottleneck machinery: [Alemi et al., Deep VIB §3 Eq13–15](https://arxiv.org/html/1612.00410v7), verified in the primary text. Gaussian stochastic encoding and its deterministic-mean caveat are also explicit there. The conditional posterior/reconstruction structure overlaps [Sohn et al., CVAE §4 and §4.2](https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf), already reviewed in the [previous proposal](future-information-utilization-proposal.md). Progressive residual coding and bounded Gaussian parameterizations are not claimed novel. This is effectively an additional VIB latent with an architectural maximum rate and explicit recoding/noise controls. Free-bits objectives ordinarily reduce the optimization penalty below a chosen KL threshold; they do not enforce the pointwise upper bound used here. This distinction is an experimental constraint, not a new information-theoretic principle. Unlike the previous hard local-repair proposal, this bounds a genuinely sampled additional channel relative to independent noise; it does not bound a changed deterministic mean and then mistake that for an information guarantee.

As implemented here this would be a useful engineering/mechanism experiment. A conference-level contribution would need a defensible new result about conditional predictive sufficiency or an improved action-transfer algorithm, not this combination alone: replicated rate-versus-prediction behavior beyond strong recoding controls, genuinely held-out moving-camera/human tasks, useful downstream command-conditioned prediction, and evidence that gains are not merely appearance leakage. The present evidence does not yet establish those conditions.
