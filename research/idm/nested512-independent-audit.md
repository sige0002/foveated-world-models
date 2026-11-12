# Independent nested512 cache audit: passed

The completed `runs/reference-idm42-nested512-prior10000` cache passed the read-only CPU2 audit in `verify_nested512_cache.py`. The audit did not restart extraction, modify shared scripts/caches, decode images, rerun the visual/query encoder, or run training.

Verified:

- All retained128 tensors and rows are exactly equal to the parent cache.
- Added384 episode/index/endpoint/dt mappings match the fixed selection and NPZ metadata; all endpoints are originally observed with zero fill age.
- All24 chunk hashes match; chunk current/future/baseline slices exactly match the expanded train tensors, and query/reference slices exactly match `additional_queries.pt`.
- The fixed completed seed42/step2000 head and original train calibration match their hashes. Recomputing in medium precision, batch128, CPU2 reproduces every retained and added code exactly: both maximum absolute errors are **0.0**.
- Validation64 has the same byte hash as the parent. Train512 has512 unique episodes and no overlap with the160 unique episodes in dev64 plus observed96.
- Source/checkpoint/calibration hashes, encoder identity and old/new visual-state identity pass. All loaded tensor artifacts are finite and CPU-resident.

The extraction's live process was independently observed before completion with `CUDA_VISIBLE_DEVICES=''` and `OMP_NUM_THREADS=4`; limited runtime evidence is saved in `nested512-extraction-preflight.json`. The audit itself used CPU2 with CUDA disabled. Saved CPU tensor placement alone is not treated as retrospective proof of all hardware operations.

**Metadata interpretation:** inherited `export.verification.train.pairs=128` and `frozen_head_training_pairs=128` describe the original head export/training. They do not claim verification or head/calibration refitting on512. The original IDM head/calibration stay fixed; this independent audit verifies the expanded512 cache and its additional codes.

Audit result: `nested512-independent-verification.json`. Completed cache-manifest SHA256: `998ed490c2ab76209c21b76c2fcd3fa83c95d8ad100def6aad536ce069fa8471`.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/verify_nested512_cache.py --output research/idm/new-audit-result.json
```

The script refuses to run before the final cache manifest exists and refuses to overwrite an existing audit result. Image-feature correctness beyond the preserved source and saved-chunk checks is outside this independent audit; original extraction recorded its own exact two-example smoke test.
