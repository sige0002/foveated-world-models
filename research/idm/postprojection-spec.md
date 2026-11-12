# Mature encoder normalization controls

These are prepared representations and fresh heads, not trained models. Extract exact train128/validation64 current/future pairs from `runs/reference-idm-with-prior6501` using the immutable step6501 encoder, CPU2, float32 highest, batch1. No visual encoder is rerun. `research/idm/mature-postprojection-cache` stores Q (1024D), p (32D, before VAE LayerNorm), row identities, representations and training calibration. Source features must agree exactly with the step6501 source cache, and episode/index/dt must agree in order. Input/model hashes are checked before/after extraction. Train and validation episodes must be disjoint.

Let m=mean(p), s=sqrt(mean((p−m)^2)+1e-5), u=(p−m)/s. Every representation has34 coordinates:

| Variant | Input to new head |
|---|---|
| direction | [u,0,0] |
| direction_statistics | [u,standardize_train(m),standardize_train(log s)] |
| calibrated_raw | [standardize_train_per_coordinate(p),0,0] |

Calibration uses population standard deviations with floor1e-3 and exclusively train128. It is frozen for validation and any future research evaluation. No learned original LayerNorm affine is used. All groups must train new heads. `postprojection_heads.shared_initial_heads` returns independent34→256→64 Gaussian heads with exactly equal initial weights and parameter count; the final64 values represent32 mu and32 logvar coordinates when connected to the training harness. No training harness has been launched. Future comparisons must also hold decoder, initialization, batch schedule, posterior noise, objective and update count fixed.

The decomposition reconstructs p=m+exp(log s)*u in exact arithmetic. Finite epsilon makes u weakly scale dependent; direction-only does not mathematically erase every possible scale clue. Raw calibration is affine and invertible when stored scales are positive. Reintroducing statistics cannot recover Q components lost in projection. Neither this algebra nor future sensitivity identifies action or removes camera information. Matching parameter count does not make the information content of padded inputs equal: this is the intended experimental distinction.

Reproduce extraction (new output required):

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/cache_mature_postprojection.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m unittest research.idm.test_postprojection_representations -v
```

Meaningful tests cover mathematical reconstruction, constant vectors, finite-epsilon scale dependence, frozen train-only calibration under drastically changed validation inputs, raw-calibration inversion,34D shapes, and identical independent head initialization. The test log is `postprojection-tests.txt`. No semantic or performance result follows from these tests.

## Completed extraction

All128 train and64 validation examples were extracted in507.76 seconds using CPU2. Source/model hashes remained unchanged, train/validation episodes were disjoint, and source feature/row identity checks passed. `verify_mature_postprojection.py` independently reproduced training calibration and all saved representations exactly. Float32 reconstruction maximum absolute errors were0.003418(train) and0.002930(validation), within the magnitude-scaled rounding bound for the large projected coordinates; the float64 mathematical reconstruction test passed at1e-13 tolerance. See `mature-postprojection-verification.json` and `mature-postprojection-cache/manifest.json`. All7 tests passed. Predictive head training remains unstarted.
