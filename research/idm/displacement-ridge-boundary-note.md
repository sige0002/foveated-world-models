# One-time train-CV ridge boundary audit

The original train-only CV selected alpha1000, the original upper boundary, for zero and original6501 codes. This training-set fact motivated one prespecified common-grid extension for **all three variants** to {.01,.1,1,10,100,1000,10000,100000,1000000}. The new protocol was saved before fitting. All train-CV selections were saved to `train_selection.json` before any validation64/observed96 predictions. Inputs, exact episode folds, fold-specific scalers, loss definition and feature/target floors were unchanged. No further grid extension was performed.

| Alpha | Zero CV MSE | Original6501 CV MSE | Reference42 CV MSE |
|---:|---:|---:|---:|
| .01 | .980209 | .963555 | .748733 |
| .1 | .979962 | .963322 | .748615 |
| 1 | .977533 | .961030 | .747450 |
| 10 | .957108 | .941502 | .737783 |
| 100 | .883179 | .867228 | **.712052** |
| 1000 | **.868332** | **.856524** | .788403 |
| 10000 | .989493 | .989274 | .971844 |
| 100000 | 1.075262 | 1.075545 | 1.072918 |
| 1000000 | 1.093243 | 1.093273 | 1.092997 |

All selections remain unchanged: zero1000, original6501 1000, reference42 100. They are interior points of the extended grid. Thus the original boundary did not conceal a better baseline among these additional candidate penalties. This remains a finite-grid CV statement, not proof of a globally optimal regularizer or current representation.

| Code | Original val64 | Extended-grid val64 | Original observed96 | Extended-grid observed96 |
|---|---:|---:|---:|---:|
| zero | .943928 | .943928 | .866747 | .866747 |
| original6501 | .908416 | .908416 | .839108 | .839108 |
| reference42 | .732079 | .732079 | .603342 | .603342 |

The original six alpha scores reproduce within1e-13. Final model tensors are bitwise identical to the original selected models, and every saved prediction/error is unchanged. Original output files were not modified. New protocol, selections, models and complete CV/per-example results are under `displacement-ridge-boundary-audit/`; integrity findings are in `displacement-boundary-verification.json`.

All interpretation limits remain: reference mu has linear predictive utility for measured displacement beyond the specified pooled-current/state inputs, but pooled current is not a strong spatial/nonlinear baseline; z could supply missing current information. This is not conditional MI, action identification, camera invariance or executable future prediction. Both evaluation sets are research-observed, and96 is not an independent final test. A stronger spatial/nonlinear current baseline remains necessary.

Reproduction script: `probe_displacement_ridge_boundary.py` (CPU2; refuses existing output directory). The only fitting performed was ridge regression.
