# Fixed posterior means add linear measured-displacement utility

The prespecified ridge probe has completed, with no world-model training. All settings and exact5episode folds were written to `displacement-ridge-probe/protocol.json` before fitting. Hyperparameters were selected from train128 only; neither validation64 nor observed96 was used for alpha/scaler fitting. No settings were changed after inspecting results.

Each input has820 coordinates: current DINO patch mean768, current prepared state20, and32 code coordinates. The three code choices are zeros, original step6501 posterior mu, and the existing reference seed42 posterior mu. All use identical folds, feature dimensionality, ridge implementation and alpha grid. Within each training fold, input and8D target scalers are fit afresh; feature std floor1e-6 and target floors .001m/ .01 gripper-source units are fixed. Fulltrain scalers/models are then fit once and applied to both evaluation sets. Alpha penalizes coefficients in the sum-squared-error objective; it is not normalized by sample count. The unpenalized intercept is implemented through centering.

| Code | Train-CV selected alpha | Train normalized MSE | Validation64 normalized MSE | Observed96 normalized MSE |
|---|---:|---:|---:|---:|
| zero |1000|.394938|.943928|.866747|
| original6501 mu |1000|.380046|.908416|.839108|
| reference42 mu |100|.048161|.732079|.603342|

Reference mu reduces normalized MSE relative to zero code by22.44% on validation64 and30.39% on observed96. The physical coordinate MSEs improve on7/8 coordinates in both sets; left gripper slightly worsens (.110397→.110550 on validation; .115953→.117834 on observed96). Thus the aggregate result is not uniform improvement across measured components. Full8D physical-coordinate errors (m² for xyz, source-unit² for gripper), predictions and targets for every row are saved in `displacement-ridge-probe/results.json`. No mixed-unit unnormalized scalar error is presented.

This establishes **linear predictive utility of these fixed future-conditioned posterior means beyond the specified pooled-current and current-state inputs**, within this dataset/protocol. It is not conditional mutual information, action identification, camera invariance or deployable future prediction. The probe sees a compressed current image, and z may encode nonlinear current information missing from that linear baseline. A spatial/current nonlinear baseline could explain some gain. The posterior itself consumes true future visual information; this is intended for representation auditing but must not be described as an execution-time forecast.

The mean probe is different from decoding a single sample from a Gaussian posterior. Original6501 means also have documented numerical precision dependence; per-coordinate feature standardization can amplify small signals and numerical variation. The probe does not settle whether the original world-model decoder can use that information. Nominal dimension/capacity is matched, but zero padding intentionally removes sample-dependent code features and lowers effective design rank. Training error is much lower for reference, so overfitting remains relevant even though both fixed evaluation sets improve.

Observed96 has already been inspected during visual research and is **not an independent final test**. Validation64 also participated in earlier representation design. A new evaluation split and stronger current-only controls would be needed for a confirmatory claim; none was silently added here. Alpha1000 is the upper grid boundary for zero/original: the grid was not extended after seeing that fact, and those baseline hyperparameters are optimal only within the prespecified grid.

Provenance and validation: source cache/label/manifest hashes and original row fields agree, every target/current-state coordinate is observed, all three groups share the exact folds, and no data was imputed. Four synthetic tests pass: dual/primal ridge equality, intercept centering, zero-input/target-floor handling, grouped folds, and scaler immutability (the grouped/intercept checks are combined within four test cases). `verify_displacement_probe.py` reloads fitted models without refitting and reproduces predictions/metrics, source hashes and protocol hashes. See `displacement-probe-verification.json` and `displacement-ridge-tests.txt`.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/probe_displacement_ridge.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/verify_displacement_probe.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m unittest research.idm.test_displacement_ridge -v
```

The fitting script refuses to overwrite the completed output directory. New world-model training, new auxiliary objectives and causal claims are outside this experiment.
