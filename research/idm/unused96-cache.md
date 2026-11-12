# Fixed research-unused episode cache

The selection was fixed from metadata before image decoding or prediction error evaluation. All 96 validation episodes are absent from the research train128/validation64 caches. They may have appeared in original Stage1 validation aggregates; this is not an external test set. Do not use these outcomes to change model design or select checkpoints.

## Reproduction

From repository root, on CPU only:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python research/idm/cache_unused_episodes.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python research/idm/align_unused_decoder_precision.py --cache runs/research-unused-96-prior6501
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python research/idm/verify_unused_cache.py > research/idm/unused96-cache-verification.json
```

Extraction and alignment refuse to overwrite completed outputs or realign an already aligned cache. Source caches, fixed head, calibration and selection are not modified. No training or recalibration occurs. The output is `runs/research-unused-96-prior6501/validation.pt`, with `cache_manifest.json`; there is no train split.

## Tensor and score definitions

- `current`, `future`: frozen visual patch features. Both checkpoint visual state hashes match, and features computed independently through both original model paths agree bitwise on all 96 pairs.
- `codes`: fixed seed42 reference head posterior means, using step2900 Q(x,y)−Q(x,x), training calibration and the completed step2000 head. These use the true future and are not execution-time priors.
- `original_codes`: actual step6501 original encoder/VAE posterior means, computed by the original model path in medium precision, batch1.
- `baseline`: step6501 original decoder(current, zero code), recomputed at highest precision, batch2 to match the correction training cache.
- `rows.original6501_mse`: original decoder with `original_codes`, highest precision, batch2, matching the original cached-reference evaluation.
- `rows.original6501_mse_medium_b1`, `rows.baseline_mse_medium_b1`: original extraction-path scores for numerical reproduction. Compatibility fields `reference_mse` and `reference_zero_z_mse` retain those medium/batch1 values; use the explicitly named aligned fields for comparisons.

Adapted head inference uses medium precision with batches64/32. The source export used validation batch64. The independent head precision audit documents stability of the adapted reference head; this does not assert that raw production head variation is precision independent.

The manifest records selection, source cache/model/head/calibration hashes, precision alignment diagnostics, and preserved tensors. The verification script checks final hashes, selection identity, episode disjointness, tensor shapes and finiteness. Parent evaluation must retain each model's preselected final400step checkpoint, and use the constant model's training-cache constant on these examples.
