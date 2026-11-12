# Frozen reference-query pilot and matched learning specification

This is a mechanism experiment, not a claim of a new action representation or camera invariance.

## Reproduce inputs

Run from the repository root:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python research/idm/cache_reference_queries.py --cache runs/frozen-idm-step2900 --output research/idm/reference-query-full
```

Omitting `--pairs` extracts every existing train128/validation64 cache row. The executed smoke experiment uses `--pairs 8` and output `research/idm/reference-query-pilot`. The parent extraction script `research/cache_stage1_features.py` already selected observed endpoints, deterministic preprocessing, frozen DINO features, and episode-disjoint splits. This script reuses those exact tensors instead of re-running the dataset loader. It verifies both tensor-file SHA256 values against the original manifest and verifies posterior means against every selected cached code. Batch size one, medium float32 matmul precision, and four CPU threads match the parent extraction. It does not allocate CUDA tensors.

Outputs per split contain `query=Q(x,y)`, `reference=Q(x,x)`, `future_swapped=Q(x,y_other)`, parent indices, and unmodified metadata rows. Q is the final 1024-dimensional query before projection to 32 dimensions. A transparent forward wrapper preserves the encoder's inference path. Future swaps are dependency probes only.

Train calibration stores one global 1024-dimensional query center and three scalar scales with a floor of 1e-3:

- raw: query / train RMS(query)
- centered: (query − train mean(query)) / train RMS(query − train mean)
- reference: (query − reference) / train RMS(query − reference)

No validation value determines the center, scale, floor, or hyperparameter. Scalar calibration preserves feature anisotropy and relative motion amplitude. Numerical variance after rescaling does not quantify information gain. Reference subtraction cancels only a common additive offset conditional on x; it has no guarantee against camera motion.

## Matched learning experiment (specified, not yet executed)

1. Extract full caches. Reject a split if parent episode identifiers overlap or any posterior reproduction error exceeds 1e-5. Preserve all 128/64 rows; do not tune settings on the validation subset.
2. For each representation, use the same new head `Linear(1024,256) → GELU → Linear(256,64)`, split into 32-dimensional mu and logvar. No per-example input LayerNorm. Identical initial head parameters for each seed. Clamp logvar consistently to the existing VAE bounds.
3. Use the same randomly initialized two-layer, width-128, four-head residual decoder supported by `research/train_cached_decoders.py::make_decoder('residual',...)`. Current features are identical. Initialize the decoder identically within each seed. This avoids measuring out-of-distribution codes through an unchanged pretrained decoder. Both head and decoder train; the original encoder stays frozen.
4. Compare raw, global-centered, reference-subtracted, and a constant-code control. The first three have exactly the same head capacity and optimizer. For the constant-code control keep the same modules but feed the train representation mean. This measures what current-only prediction achieves with the same optimization machinery.
5. Fix AdamW settings, batches, seeds 42/43/44, 2000 updates, and the same checkpoint step across variants before seeing results. Use the parent's decoder comparison learning rate and regularization; record resolved values in output. Use one posterior sample per pair, reconstruction loss and beta=5e-5 KL. Evaluate both posterior mean and samples. Do not change routing for only one representation. A later deterministic-head ablation is separate and must apply to all variants.
6. Keep a separate direct-DINO-difference baseline for prior-art comparison. Its dimensionality differs, so explicitly report adapter parameters and match them as closely as possible; do not present it as exactly head-capacity-matched.

## Separate measurements

**Posterior dependence:** fixed current, replace future, recompute posterior; record mean/logvar change and Gaussian KL. This is a dependence probe, not conditional mutual information or an action counterfactual.

**Decoder utilization:** with fixed current and common epsilon, swap observed posterior means/logvars separately; report prediction change and paired risk. Also evaluate constant-code controls. Increased sensitivity alone is not a success metric.

**Generalization:** held-out episode feature MSE, paired per-episode intervals, and train/validation gap. The cache is small; success is only a reason to run larger training.

**Action relevance:** the feature cache has no robot action/state labels. Load labels only through the original row indices and exact parent dataset/config if this stage is added, and verify endpoint alignment. Compare current-only versus current-plus-z predictions of actual robot increments using equal capacity and held-out episodes. Do not infer joint labels from latent dimensions.

**Camera confounding:** there are no camera-pose labels in this cache. Global future swaps combine action, appearance, and camera changes. Camera invariance remains unmeasured. A correlation with global feature change would be only a nuisance warning, not a camera test; do not label it as camera motion. A later paired camera transform or pose-labeled evaluation must independently vary camera and interaction.

## Stop conditions and novelty limits

Reject reference subtraction as the favored intervention if it does not beat global centering and raw difference, improves only training reconstruction, increases future dependence without useful decoder conditioning, or fails incremental action prediction. Its extra encoder call must be counted for end-to-end experiments. ACT-LAM and RLA-WM already use transition residuals; JEVA already grounds latent actions and addresses sampling collapse. A subtraction location and amplified variance alone are not a top-conference contribution.

## Executed eight-pair-per-split result

The pilot completed with parent-cache hash matches, episode-disjoint rows, and exactly zero posterior-mean reproduction error in both splits. Training scalar scales are raw 41.03484, centered 0.0390652, and reference 0.0297535. Both subtraction forms expose small fluctuations previously dominated by a large shared offset.

This does **not** favor reference subtraction yet. At fixed x,

`[Q(x,y_other)-Q(x,x)] - [Q(x,y)-Q(x,x)] = Q(x,y_other)-Q(x,y)`.

Thus its absolute future-swap response is exactly unchanged before scale calibration. Its larger reported normalized response is partly the smaller scale, not newly recovered information. Global centering also exposes variation with one encoder call. A learned, capacity-matched comparison is required to justify the reference branch's extra cost. No decoder was trained or evaluated in this query pilot, and no camera claim follows. The saved manifest explicitly records these limits.

Artifacts: [pilot manifest](reference-query-pilot/manifest.json), [extraction script](cache_reference_queries.py). At the pilot stage the full-cache and head-training runs were unexecuted; the authorized run below is now complete.

## Authorized full run

Full extraction and matched training were subsequently authorized. `run_reference_experiment.py` waits for the verified full query manifest, runs seed 42 for 400 updates per variant, validates common initial parameter hashes, sampled batch hashes, Gaussian-noise hashes and a zero swap response for the constant baseline, then runs seeds 42/43/44 for 2000 updates per variant. The runner fails closed on extraction timeout, subprocess errors, or matching-control failures. Outputs are `reference-smoke400/` and `reference-comparison2000/`.

Implementation: [trainer](train_reference_heads.py), [runner](run_reference_experiment.py), [focused tests](test_reference_heads.py). Three tests passed for matched initialization, reference/constant representation logic, and a finite differentiable decoder/head path. Training checks loss and gradient finiteness on every update. Heads use diagonal Gaussian sampling, logvar clamp [-10,10], beta 5e-5, MSE reconstruction, and AdamW lr 3e-4 with the parent pilot's weight-decay split. The production reconstruction uses SmoothL1; matching the beta coefficient does not make these objectives quantitatively equivalent. One fixed epsilon is shared for paired decoder-swap evaluation; training noise is identical across variants for each seed. Fixed-current future swaps separately measure Gaussian posterior dependence.

The encoder is frozen and the decoder is a fresh width-128, two-layer residual model. Results must be presented as a small-cache representation experiment, not evidence that the production decoder or full Stage1 training has improved. The constant control still samples a common learned Gaussian, so it measures current-only prediction with matched stochastic regularization, not a deterministic current-only network.

## Export completed heads for decoder comparisons

`export_reference_codes.py` accepts only a checkpoint whose seed, variant and final step have a completed entry in the supplied comparison report. Example for the completed smoke reference head:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python research/idm/export_reference_codes.py --cache runs/frozen-idm-step2900 --queries research/idm/reference-query-full --comparison research/idm/reference-smoke400/comparison.json --checkpoint research/idm/reference-smoke400/reference-42.pt --output research/idm/export-smoke-reference42
```

Change `reference` to `raw` or `centered` for the other smoke heads. Export preserves current/future tensors and metadata rows exactly and changes only `codes`. It reloads the saved export to verify preservation. The new cache manifest records parent cache hashes, checkpoint hash, comparison/query manifest/calibration hashes and posterior-future use. These caches are suitable for matched Stage1 reconstruction/decoder comparisons. A head consuming Q(x,y) does not provide an executable current-only action prior and must not be evaluated as one. The seed42/2000step reference export was subsequently executed after all primary comparisons completed; see the result section below.

## Completed primary result

All train128/validation64 queries were extracted with zero posterior reproduction error. The 400-step smoke gate passed, and all 12 conditions (four variants × seeds 42/43/44) completed 2000 updates. Initial-weight, minibatch, and posterior-noise hashes agree within every seed, parameter counts agree, and constant-code swap effects are exactly zero. All per-step loss/gradient checks passed. [Full report](reference-comparison2000/comparison.json), [summary](reference-comparison2000/summary.json), and [source/checkpoint provenance](reference-comparison2000/provenance.json).

| Representation | Validation mean-code MSE, seed mean ± SD | Sample-code swap risk increase |
|---|---:|---:|
| Raw query | 0.178606 ± 0.001299 | -0.0000028 |
| Global center | 0.170286 ± 0.002487 | +0.010436 |
| Current reference | 0.160962 ± 0.000721 | +0.026050 |
| Constant | 0.179063 ± 0.000788 | 0 |

Reference subtraction reduces mean-code MSE by 5.48% relative to global centering and 9.88% relative to raw query in this fixed-encoder/small-decoder experiment. The reference-minus-centered episode-mean difference is -0.009324, with a paired episode bootstrap interval [-0.011916,-0.006827]. This interval conditions on these three trained seeds; seed variability is reported separately. Training MSE is similar (centered 0.080646, reference 0.081049), so the measured validation advantage is not simply a lower final training loss.

This establishes a reproducible reconstruction advantage in this small-cache comparison, not action identifiability, camera invariance, or improved production Stage1 quality. All posterior codes use the real future. Global swaps are dependency stress tests and may be out of distribution. See [conditional-utility audit](conditional-utility-audit.md) for the missing semantic and conditional-swap checks.

The completed reference seed42 head was exported to `research/idm/reference-codes-seed42/`: train128 and validation64 current/future tensors and rows are preserved exactly after reloading. Only codes were replaced. [Export manifest](reference-codes-seed42/cache_manifest.json) records source split hashes, encoder and head tensor hashes, checkpoint/calibration/query hashes, and the explicit posterior conditioning meaning. This is ready for the parent's matched decoder/flow comparison.
