# Two diagnostics for the mature checkpoint and reference-centering mechanism

## Updated evidence and correction of interpretation

The parent verified identical current/future tensors and episodes at steps 2900 and 6501. At step6501, after warm-up5000, original-decoder MSE is 0.125485, global-swapped-code MSE 0.125495, and zero-code MSE about 0.126313. Validation mean-code variance rises from 9.11e-6 to 1.386e-4, but that alone does not show useful z conditioning. The strong current-conditioned predictor improves substantially while the measured mean-code swap effect remains tiny. These observations do not identify whether Q, the posterior head, or the decoder is primarily responsible.

**Numerical correction:** the earlier tiny mean-code variances and future-swap responses must not be treated as precision-independent estimates of information. They are properties of the specified evaluation computation. A new head-only precision audit on the identical saved step2900 Q shows:

| Production projection/VAE head on saved Q | Between-example mu variance | Fixed-current future-swap mu change energy |
|---|---:|---:|
| float32, medium matmul setting | 9.1108e-6 | 7.1117e-6 |
| float32, highest setting | 6.7527e-7 | 3.1368e-7 |
| float64 | 6.7516e-7 | 3.1358e-7 |

Medium reproduces the original cached mu exactly, but differs from float64 by up to .003288. Highest float32 differs by at most 9.94e-7. We do not assert which internal reduced-precision kernel PyTorch selected; the measured difference is between its documented settings. The saved Q itself still contains effects of its original encoder evaluation.

The adapted reference heads are much more stable on the same Q: medium/highest mu variance is 20.3883/20.3869, 16.8481/16.8465, and 19.7667/19.7608 across seeds42/43/44. Medium numerical-error energy is approximately 4e-6 of their float64 between-example variance. Global-centered heads are similarly stable. Adapted raw/RMS-only heads instead have tiny variance dominated by precision differences: medium error energy is 9.5–10.5 times their float64 variation. Thus large variation in centered/reference heads is not explained by this particular numerical artifact; this still does not validate action semantics.

[Precision audit](head-precision-audit-step2900.json) records PyTorch version, CPU/four threads, settings, batch sizes and query/calibration/head/script hashes. [Reproduction script](analyze_head_precision.py). That audit used saved step2900 tensors only. A subsequent step6501 eight-pair encoder/head measurement is reported below; it does not include decoder sensitivity.

## What explains the existing controls without overclaiming

At fixed X, `Q → Q−Q(X,X)` is an invertible translation. Therefore ideal conditional information `I(Y;Q|X)` is unchanged, as it is under subtracting a train mean and dividing by a nonzero fixed RMS. Reference subtraction does not create future information. Finite-capacity heads, initialization, normalization, optimization, numerical precision, and the removal of current-dependent offsets can change how accessible that information is.

In the existing three-seed comparison, “raw” already means Q divided by its train RMS: pure RMS scaling failed, global centering helped, and reference subtraction helped further. That rules out attributing the whole gain to one overall scale factor. It does not isolate the extra reference benefit: removing current-dependent offsets, changing feature geometry, numerical conditioning, or an OOD Q(X,X) reference are all viable explanations. Those heads and decoders were newly trained, so their improvement cannot be assigned solely to modifying the production VAE head.

A lightweight saved-Q geometric check further constrains the story. At step2900, the original W projection transmits the measured future-swap direction with 3.62 times the energy expected for an isotropic direction of the same norm. Thus these data do **not** support saying W preferentially removes future directions. Projected coordinates have standard deviation about231.8; much of the perturbation survives coordinate centering and radial removal, but LayerNorm scales it down. The LayerNorm Jacobian predicts its finite change with relative squared error2.56e-8. These are coordinate-dependent local transmission facts, not information measures or a proof that LN caused training failure. [Geometric result](saved-head-transmission-step2900.json), [script](analyze_saved_head_transmission.py).

Adding LN before projection is not a solution established by these observations: earlier algebra/real-weight checks showed that scalar rescaling is largely canceled by the existing downstream LN. Arbitrarily inserting centering into the production checkpoint would also be an untrained distribution change.

## Diagnostic 1: localize the future response through Q and the posterior head

Run a small fixed subset of eight identical observed validation pairs at both checkpoints; no optimization. First reproduce medium/B1 cached mu. Then measure the same frozen computation with highest float32. At a fixed current, use the real future and the prespecified cyclic replacement future strictly as a dependence probe. Q(X,X) may be included as an explicitly OOD reference, not a valid zero-action label. If local sensitivity is needed, use a small fixed interpolation fraction toward the replacement feature and label it a feature-space perturbation, not a physical counterfactual.

Capture Q before projection, WQ+b, coordinate-centered/tangent components at LayerNorm, normalized values, mu and logvar. Report absolute perturbation energy, same-scale isotropic-direction transmission, posterior symmetric Gaussian KL, and the numerical discrepancy between precision settings. Do not infer information from common-norm ratios alone. The saved-Q audit already supplies the head part at step2900; the missing work is especially step6501 Q and its posterior.

Interpretation:

- If Q's response is itself below numerical resolution at highest precision, the head cannot be blamed alone; inspect encoder access. This does not prove no response in every action direction.
- If Q changes measurably but posterior distributions barely separate, the bottleneck/head is the immediate attenuation point. Check whether projection, normalization or noise scale accounts for it before proposing an architectural repair.
- If posterior separation is substantial but decoder effects stay tiny, proceed to diagnostic2 rather than calling the head collapsed.

This is a single forward-path attribution diagnostic. It does not require new labels, a new baseline fit, or full query-cache extraction at step6501.

## Diagnostic 2: separate weak latent amplitude from low decoder response

With the corresponding original decoder frozen, evaluate eight pairs at their true mean and at two common-epsilon posterior samples. Use highest float32 for small-difference measurements and retain the original medium computation as a numerical reference. Exchange mu and logvar independently using observed posterior values, as in the earlier factor diagnostic, keeping current and epsilon fixed.

Add a local directional-gain measurement: compare the observed future-dependent latent direction with isotropic directions having the same posterior-Mahalanobis length. Evaluate the native direction and a fixed small radius (e.g.0.1 in total Mahalanobis norm), with symmetric positive/negative offsets. Report prediction-difference energy divided by latent-displacement energy, both absolute and relative to numerical resolution. A tiny predetermined finite-difference scale check can detect rounding-dominated estimates. This is a decoder Jacobian/gain probe, **not** an oracle optimization or a claim that rescaled codes improve prediction.

Interpretation:

- Small native output changes but appreciable matched-radius responses indicate that the decoder can respond locally, while inferred future distinctions are too small or poorly oriented. This supports a head-access/conditioning explanation.
- Small responses to both observed and matched-radius directions indicate local decoder insensitivity under the tested posterior region. It does not show that no distant z can change the decoder.
- Responses only in random directions indicate a possible mismatch between encoded directions and decoder-accessible directions; without measured actions it is not evidence of useful control.

The fixed-radius probe can leave the empirical mean-code cloud even though the Gaussian posterior has support there. Label it a local model probe and retain the native swaps as the observational measurement. Global future/code swaps remain potentially OOD; neither diagnostic certifies action use or camera invariance.

## Priority and limits

These are the only two proposed new diagnostics. Do diagnostic1 first, including the precision check, then diagnostic2 only where posterior response is numerically resolved. Preserve the strong step6501 current prediction rather than replacing it to make latent use look larger. The parent's frozen-current residual-correction experiment is compatible with this diagnosis but is a separate learning experiment. No new training was run. The initial note used saved-query/head arithmetic; the subsequent eight-pair step6501 encoder diagnostic below now adds direct measurements.


## Executed diagnostic 1: step6501, eight fixed validation pairs

Reproduction:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/diagnose_mature_transmission.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/summarize_mature_gaussian.py
```

Both scripts refuse to overwrite completed reports. [Full result](mature-transmission-step6501.json), [saved queries](mature-transmission-step6501.pt), [Gaussian separation](mature-gaussian-step6501.json). Reports contain script, weights, input-cache and saved-query hashes. Eight uniformly spaced existing validation rows (indices0,9,18,27,36,45,54,63), cyclic future replacement within those eight, CPU2, batch1. No vision extraction, fitting or decoder evaluation occurred. Medium reproduces cached posterior means exactly. Encoder was evaluated independently in highest and medium settings; head-only precision comparisons hold highest-encoder Q fixed.

| Location | Future-replacement response energy, highest encoder + float64 head | Response / matched isotropic response |
|---|---:|---:|
| Q | 74102.9568 | 1.00 |
| Projection | 103243.2521 | 110.86 |
| LayerNorm | 0.001325116 | 28.89 |
| mu | 0.000448750 | 42.05 |
| logvar | 0.000199620 | 13.12 |

The isotropic control uses64 seed713 directions per row, each with exactly the corresponding future-replacement query displacement norm. It is a coordinate-scale control, not an action or nuisance distribution. The highest head calculation agrees closely with float64: mu numerical-difference energy2.29e-12 versus future response4.49e-4. Full medium gives mu response4.34e-4, but its same-Q head error energy is5.16e-5; logvar is more affected (medium response2.88e-4 versus float642.00e-4). Thus the mature signal is resolved in highest and is not entirely a medium-precision artifact. This does not rehabilitate earlier step2900 tiny-response estimates.

Projected coordinate standard deviation is3345.57. Of coordinate-centered future-displacement energy103147.58,88127.90 (85.44%) is radial with respect to the centered projection, which LayerNorm largely removes. Tangent energy15019.68 survives that removal, but division by the large projected standard deviation makes its normalized response small. The LayerNorm Jacobian predicts the finite displacement with relative squared error0.000506. These measurements locate strong coordinate attenuation at normalization; different units alone do not establish information loss or a harmful training mechanism.

Posterior separation is also small relative to sampling noise: average symmetric Gaussian KL is0.00562488 nats; individual values range8.42e-6 to0.012764. Average posterior variance is0.0403789 and mean-difference squared Mahalanobis distance is0.0111504. These are the distributions for the tested same-current future replacements, not an estimate of mutual information across all futures. Variation between pairs is substantial, and n=8 does not support population conclusions.

**Updated hypothesis:** the original mature encoder has a measurable future-dependent response, and its projection preferentially transmits these tested directions compared with isotropic controls. Neither “Q ignores the future” nor “W rejects future directions” explains these observations. Future dependence arrives mainly as changes around a large shared projected component, with a large radial component and relatively small normalized/posterior separation. This makes a finite-head geometry and posterior-noise explanation more plausible than total absence of future access. It remains possible that this dependence codes camera or scene differences; the exchange is OOD and no action semantics are identified.

This is compatible with raw/RMS failing, global centering helping and reference centering helping further: a scalar RMS cannot remove the shared offset or radial component, whereas centered inputs change how a newly trained finite head sees conditional deviations. It does not prove that the additional benefit of Q(x,x) subtraction is causal, or that its reference is an in-distribution no-op. Nor does it justify inserting centering into the existing head without retraining. The next useful intervention remains a matched trained centered/reference head atop a preserved current-only prediction, with a constant-code control. Diagnostic2 is still needed to attribute weak decoder effects to latent amplitude versus decoder directional gain. The earlier2900 results used64 rows and a different cyclic future mapping, so these new energies are not a controlled checkpoint-to-checkpoint growth curve.

### Relation to a normalization head that retains mean and scale

A concrete untrained candidate is to give the Gaussian head the normalized projected direction plus two scalar channels: projected-coordinate mean and log standard deviation, standardized using training data only. For projected p in R^32, define m=mean(p), s=sqrt(mean((p−m)^2)+epsilon), and u=(p−m)/s. Then p=m+s*u exactly in real arithmetic. Supplying (u,m,log s) explicitly retains the mean and scale rather than requiring them to survive implicitly in normalized coordinates. Mean is exactly removed. Scale invariance is exact only for epsilon=0: with epsilon>0 and exact real arithmetic, v=mean(u^2)=var(p)/(var(p)+epsilon) implies var(p)=epsilon*v/(1-v), so scale is theoretically recoverable from u. Here var(p) is enormous relative to epsilon, making that recovery severely ill-conditioned and susceptible to rounding; the measured claim is scale attenuation and accessibility, not exact two-dimensional information loss for finite epsilon. If using learned affine LayerNorm output instead of u, inversion additionally requires nonzero affine gains; using pre-affine u makes that algebra explicit. Train-only affine standardization of m/log s preserves them provided nonzero scales are used; clipping or saturation would break exact recovery.

The measured 85.44% radial share makes retained log s worth testing: the tested future dependence includes substantial magnitude change that ordinary normalized direction largely suppresses. This is a hypothesis about accessible predictive dependence, not proof of useful operation information. The mean channel is included for algebraic completeness; this measurement does not show it is important. The 32-dimensional projection has already discarded a nullspace of the 1024-dimensional Q, so preserving **post-projection** mean/scale cannot restore information lost before projection. Preserving Q mean/scale before projection is a different intervention with a different conditioning and capacity comparison.

A matched-capacity comparison must hold current fixed when measuring future sensitivity, keep the strong current-only decoder branch, and use the same training-only calibration and constant-code controls. Otherwise shared scene/current scale could explain a gain. Camera motion, appearance and OOD future replacement can also affect these scalars; action meaning and camera invariance are not guaranteed. Reintroducing mean/scale is related to existing normalization and reversible-normalization design principles; no novelty is claimed. It may reintroduce large common components and worsen numerical conditioning unless calibration and the learned head handle them. No implementation, training or performance claim for this candidate is included in this diagnostic.

## Executed diagnostic 2: mature frozen decoder, same eight rows

[Result](mature-decoder-gain-step6501.json), [reproduction script](diagnose_mature_decoder_gain.py). Executed CPU2/highest float32, batch1,84.13 seconds, no fitting/GPU/vision extraction. Source cache, immutable weights and saved query hashes match diagnostic1; the same8 row identities and indices are verified. Posterior parameters are computed with float64 head arithmetic on saved highest Q, then cast to float32 for the decoder. No prediction error against the true future is measured.

Each example uses its native same-current future-replacement mu displacement. An independent seed1107 isotropic direction is drawn in posterior-whitened coordinates, then mapped back by the true posterior standard deviation. Both native and isotropic directions are tested at the native Mahalanobis radius and at radius0.1, with symmetric positive/negative offsets. One isotropic direction per example is a small diagnostic control, not a reliable spectrum estimate. The central output-response statistic is mean(((D(mu+d)−D(mu−d))/2)^2) across predicted feature coordinates.

| Probe | Average Mahalanobis radius | Mean central output response |
|---|---:|---:|
| Native future mu displacement | .0798454 | 2.80452e-7 |
| Isotropic, matched native radius | .0798454 | 1.10414e-7 |
| Native direction, fixed radius0.1 | .1000000 | 1.94861e-7 |
| Isotropic, fixed radius0.1 | .1000000 | 1.20483e-7 |

The native direction's fixed-radius output response is about1.62 times the isotropic average. Its per-example ratio exceeds1 in6/8 examples (range.722–4.119). Thus there is no evidence here that the decoder specifically suppresses the tested future-dependent directions relative to random directions. Both absolute local responses are small. This is a coordinate-local statement, not a demonstrated inability to use informative actions elsewhere.

Native radii range.00409–.15913. Some examples have much less latent displacement than others, so small native response can reflect amplitude as well as gain. At fixed radius the native directional response still varies across examples, but remains small in feature-MSE units. The radius0.05 native check yields response/(quarter of radius0.1 response) between.999962 and1.000068. The clean quadratic scaling argues against rounding dominating these finite differences at the tested radii, without establishing a universal numerical floor or global Lipschitz bound.

For2 common epsilon draws per example, replacing only posterior mu changes prediction by2.76478e-7 mean-square, replacing only logvar by4.25683e-10, and replacing both by2.70500e-7. Logvar replacement contributes very little in these pairs/draws; this does not establish posterior-wide collapse or irrelevance of sampling noise. Keeping epsilon fixed is necessary to avoid confusing latent sampling noise with posterior-parameter effects.

**Combined interpretation:** mature Q reacts to future replacement; its normalized posterior separation is small, and the original decoder responds only weakly to both native and matched random directions locally. Native directions are not unusually inaccessible to the decoder. A story attributing the entire issue to wrong directional alignment is therefore unsupported. Small amplitude contributes on some pairs, but there is also limited local decoder gain in the tested region. The evidence supports preserving the strong current-only predictor while jointly learning a head/correction path that can access conditional deviations; it does not establish that mean/logscale preservation will work. The prepared normalization comparison and constant posterior control are needed to distinguish that intervention from extra decoder fitting. Same-current future replacements remain potentially OOD, radius probes may leave the empirical mu cloud, and camera/action semantics remain unidentified.

Reproduction command (the script refuses to overwrite an existing report):

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/diagnose_mature_decoder_gain.py
```


## A restricted distinguishability bound for the sampled posterior

For a fixed current and the two tested futures, let P and Q be their diagonal Gaussian posteriors and let K=(KL(P||Q)+KL(Q||P))/2, in nats. In an artificial equal-prior binary experiment, the best accuracy after observing one z sample is (1+TV(P,Q))/2, with TV=0.5*L1. Pinsker gives TV<=sqrt(min(KL(P||Q),KL(Q||P))/2)<=sqrt(K/2). The factor convention and natural logarithm follow [Berend, Harremoës and Kontorovich, section1.1](https://arxiv.org/html/1206.6544v8#S1.SS1); that paper uses L1 itself as its variation convention. This auxiliary source is not counted as another fully reviewed survey paper.

Applying the bound separately to the eight measured pairs and then averaging yields **52.0051% maximum average binary accuracy**, versus chance50%, for this narrowly defined single-sample experiment. [Calculation and input hash](mature-posterior-binary-bound.json). The pair-specific bounds are50.10%–53.99%. This is an upper bound implied by the measured Gaussian parameters, not a measured classifier result. A downstream deterministic decoder cannot improve distinguishability of that same sample.

This does not bound a decoder supplied with deterministic posterior means: two distinct means can be distinguishable even when sampled distributions overlap strongly. It also does not apply to multiple independent posterior samples, extra future-dependent inputs, natural-data classification, all future transitions, or action identification. The replacement future may be OOD. Thus it quantifies one diagnosed training-time stochastic bottleneck and cannot be used to claim that our posterior-mean reconstruction MSE has the same limitation.
