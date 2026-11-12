# Same-current future response of the four matched400-step heads

All four completed seed42 Gaussian heads were evaluated on diagnostic1's exact8 highest-precision6501 Q pairs. The frozen projection reproduced saved pre-LN p with maximum error0. Each group used the same frozen train128 calibration used during its training, its corresponding34D representation, and logvar clamp[-10,10]. Constant inputs were zero34. Projection/representation/head arithmetic used highest float32, batch1; Gaussian divergences used float64. CPU2; no decoder, fitting, checkpoint changes or new data extraction.

| Head | Mean symmetric Gaussian KL | Mean squared mu Mahalanobis distance | Mean single-sample binary-accuracy upper bound |
|---|---:|---:|---:|
| direction |0.0000580|0.0000923|0.50203|
| direction + statistics |12.7995|25.4095|0.85286|
| calibrated raw |28.9374|55.6555|0.89549|
| constant |0|0|0.50000|

| Head | Future-swap mu change energy | Future-swap logvar change energy | Between-pair mu variance | Between-pair logvar variance |
|---|---:|---:|---:|---:|
| direction |0.0000661|0.0000475|0.0000243|0.0000175|
| direction + statistics |13.2635|0.341505|4.83148|0.124452|
| calibrated raw |26.1010|2.93168|9.46999|1.08246|
| constant |0|0|0|0|

Energies sum over32 coordinates and average over the8 pairs; variance uses population variance on the8 true-future posteriors. Squared Mahalanobis distance uses the true-future diagonal posterior variance. Symmetric KL is half the sum of both directed KLs, in nats. Full per-pair metrics and hashes are in `postprojection-posterior-sensitivity400.json`.

Within this matched training experiment, direction-only learned extremely little distributional dependence on the tested future replacements, whereas retaining calibrated mean/logscale or calibrated raw p yielded large separation. Constant control has exactly equal mu/logvar before and after future swaps and zero between-example parameter variance. This is evidence of a difference in accessible future dependence under this architecture/training protocol, not identification of what that dependence represents. Camera, scene and OOD future-image changes remain possible explanations; the extra scalars could also improve optimization rather than exclusively carry missing semantic content.

For each fixed current, consider an artificial balanced binary choice between its actual future and the cyclic-replacement future, with one sampled z from the corresponding Gaussian and no other future-dependent input. Pinsker gives accuracy*≤.5+.5 min(1,sqrt(symmetric_KL/2)), computed pairwise before averaging. For direction-only this upper bound stays near chance. For statistics/raw it reaches1 on5/8 pairs: those bounds are uninformative on those pairs and **do not demonstrate high achievable classification accuracy**. The bound does not apply to observing deterministic mu, multiple samples, or a decoder that additionally receives the future. It is not an action-accuracy measurement or an information estimate. The convention follows `mature-posterior-binary-bound.json`.

Original6501 symmetric KL0.0056249 is a historical comparison only. Its head, optimization history, decoder and loss differ from these new400-step heads; this is not a controlled causal before/after estimate. The four newly trained groups are the primary matched comparison. Large posterior response likewise does not itself establish good reconstruction or useful real-operation prediction.

Provenance: the script verifies immutable6501 weights, saved diagnostic queries,8 row identities, saved head checkpoint hashes, completed400-step/seed42 metadata, train-only calibration and shared initial weights/batch/noise hashes. Existing input artifacts were not changed. Reproduce with `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/diagnose_postprojection_posteriors.py`; it refuses to overwrite the completed report.
