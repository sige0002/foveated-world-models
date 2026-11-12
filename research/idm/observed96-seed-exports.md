# Observed96 reference-IDM seed replication caches

The cached current/future features from `runs/research-unused-96-prior6501` were passed through the immutable step2900 encoder to regenerate Q(x,y) and Q(x,x), using float32 medium, batch1, CPU4. The original source directory retains its historical name; these episodes are now research-observed and are not an independent final test. No video decoding, GPU, training or recalibration was used.

Reusable small query artifacts are in `research/idm/observed96-reference-queries2900/validation.pt`, with source/encoder/calibration/script hashes and row identities in its manifest. Applying the fixed seed42 completed2000-step reference head using original B64/B32 batching reproduced the existing96 codes with **maximum absolute error0.0**. The script required error≤1e-5 before exporting other seeds.

Completed seed43/44 head checkpoints were verified against their existing export manifests, including encoder identity, head checkpoint/state hashes, calibration hash, seed and2000-step metadata. Their means use the same frozen training calibration and true future. Outputs:

- `runs/research-observed96-idm43-prior6501/validation.pt`
- `runs/research-observed96-idm44-prior6501/validation.pt`

Each output includes `cache_manifest.json` and has no train split. Every original field except `codes` is bitwise/value-identical to the source, including `current`, `future`, `baseline`, `original_codes` and `rows`. Thus the prior6501 baseline and original6501 comparison values retain their previous highest/B2 precision path. No baseline or original posterior is recomputed. Code generation is medium/B64+B32, matching the original42 export. These are posterior codes conditioned on the real future, not executable current-only prior predictions.

The nested `export` metadata records the preexisting completed head's provenance; `observed96_export` records this new96-row export, parent source hash, query hashes, fixed calibration, reproduction test and field-preservation checks. An independent audit checks output/input/query hashes, head identity, finite codes, exact rows and all noncode fields; see `observed96-seed-export-verification.json`.

Reproduction commands (export requires new output directories):

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python research/idm/export_observed96_reference_seeds.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/verify_observed96_seed_exports.py
```

All original caches/checkpoints/calibration files remain unchanged. Decoder evaluation and interpretation of seed reproducibility are separate downstream tasks.
