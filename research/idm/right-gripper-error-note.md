# Right-gripper error is concentrated in underpredicted large positive changes

This is a descriptive analysis of already saved, research-observed predictions from the train-CV-selected spatial-RBF probes. No model, predictions, threshold or hyperparameter was changed. The target is the **measured right-gripper endpoint value difference**, in its original source units. It is not a command, velocity, joint angle or validated aperture measurement; positive/negative is not equated with opening/closing.

Before examining evaluation-bin results, static was defined as |delta|≤.01, using the existing gripper target-scaling floor. The small/large boundary is the median nonstatic absolute delta in train128 only: **.7268722**. Small is(.01,.7268722], large is above that boundary. This is an operational binning rule, not a calibrated sensor-noise threshold.

## Main decomposition

| Set | Overall right-gripper MSE increase, reference−zero | Static contribution | Small-change contribution | Large-change contribution |
|---|---:|---:|---:|---:|
| dev64 | +.0272158 | −.0001888 | +.0009870 | **+.0264176** |
| observed96 | +.0174599 | +.0006937 | −.0000321 | **+.0167982** |

Contributions are sums over the corresponding subset divided by the full set size, and reconcile exactly to the total increase. Large changes explain about97%/96% of the net worsening. Extra prediction on static examples is therefore **not the main MSE cause**, although it is present.

The positive-target subgroup is especially problematic. On observed96, its9 examples have mean absolute target.9594; mean absolute prediction shrinks from.2323 for current-only to.0907 with reference code. Its MSE rises .5369→.7663. Negative changes instead improve .3056→.2676. Dev positive changes likewise worsen .4923→.6805. Thus the main pattern is severe **magnitude underestimation of large positive source-value changes**, rather than pervasive reversal of movement direction.

At the fixed .01 detection threshold:

- Dev active14: current-only detects all14 with correct sign; reference has12 correct,1 below threshold,1 wrong sign.
- Observed96 active21: both detect all21 with correct sign. The deterioration persists despite correct signs because predicted magnitudes are too small.
- Static false-movement counts rise39→46 of50 on dev and55→66 of75 on observed96. These are thresholded regression diagnostics, not actual issued control events. On dev, static squared error slightly improves despite more threshold crossings, so count alone would misdiagnose the loss.

No predictions exceed the possible source-delta range[-1,1]. Values were not clipped.

## Concentration and training support

Top error-increase episodes account for:

| Set | Top1 share of net increase | Top3 | Top5 | Remaining mean increase after descriptive top5 removal |
|---|---:|---:|---:|---:|
| dev64 |21.7%|52.4%|77.7%|+.006582|
| observed96 |24.2%|61.9%|93.0%|+.001293|

All top5 episodes in each set have large positive targets near.974–.982. For example observed96 episode1471 has target.97797, current-only prediction.32310 and reference prediction.06485. This is a recurring subgroup, not one obviously invalid outlier. Removing error-ranked examples is shown only to describe concentration and is **not a permissible improved-performance estimate**.

Training128 contains110 operationally static examples and only18 active examples (9 positive,9 negative). Of these, just3 are large positive versus6 large negative. Dev has7 large-positive examples and observed96 has9. This sparse training support is consistent with poor generalization on the affected subgroup, but does not prove causation. The code-augmented kernel also changes similarity geometry and uses alpha.01 instead of current-only1e-8, chosen to optimize aggregate8D error. That can trade gripper performance for xyz improvements. Saved predictions cannot isolate insufficient data, latent content, kernel mixing and regularization as causal mechanisms. In particular, this failure is not proof that reference z contains no gripper information.

The already planned larger-training-set study can retain these **unchanged train128-derived bins** as a diagnostic, so changes are not hidden by redefining the hard subgroup. This note adds no new fitting or sweep.

## Integrity and artifacts

`right-gripper-error-analysis/protocol.json` fixes source hashes, bins and descriptive top1/3/5 counts; `results.json` contains all rows, signs, bin errors, biases, detection counts and concentration. `verify_right_gripper_analysis.py` checks every target against the original measured-label artifact, exact episode/index/dt, valid masks, training-derived boundary and decomposition sums. All checks pass; see `right-gripper-analysis-verification.json`. Scripts used CPU≤2, no GPU, no image decoding and no model training. Existing artifacts were not modified.
