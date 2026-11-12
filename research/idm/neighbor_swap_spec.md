# Prespecified current-neighbor swap diagnostic

Before inspecting decoder effects, use the entire current DINO patch tensor flattened into a vector, L2-normalized per example, and Euclidean distance between those normalized vectors. This preserves patch positions; it is sensitive to camera/scene differences and is not a physical-state metric. No future, latent, decoder prediction, action label, or outcome selects the distance.

Set the allowed-neighbor radius to the 95th percentile of nearest **different-episode** current distances in the training split. On validation, require a one-to-one permutation, no self-assignment, different episodes, and all assigned distances within that fixed radius. Find the minimum-total-distance matching under those constraints. If infeasible, explicitly report infeasibility and evaluate an unconstrained minimum-distance derangement only as a minimum-distance sensitivity control, not a successful neighborhood match.

Controls: the original cyclic permutation; a maximum-total-distance derangement; and the trained constant-code head. All mappings are selected once using current features and held fixed across all completed seeds and head variants. Report distance quantiles, mean/max distance, radius pass rate, and feasibility before reporting prediction effects.

Evaluate mean-code risk and shared-epsilon sampled-code risk separately, using the identical decoder and destination current/target in each pair. This is an exploratory sensitivity diagnostic. An exact conditional test requires valid conditional resampling assumptions that nearest matching does not supply. Neither smaller feature distance nor a nonzero swap effect establishes action semantics. Train radius selection prevents tuning on held-out decoder outcomes but does not ensure physical-state overlap.

## Completed result

[Full result](neighbor-swaps-step2900.json), [summary](neighbor-swaps-summary.json), [script](diagnose_neighbor_swaps.py). All three completed seeds and four variants were evaluated. Original baseline mean/sample MSE reproduced within 1e-8 for all 12 conditions. The pure-Python Hungarian solver was checked against exhaustive permutations at sizes 2–6. No source cache was changed.

The train-prespecified radius is **0.520962**. A 64-way radius-constrained derangement is **infeasible**: three validation rows have no eligible different-episode neighbor at all. Therefore no result is labeled a successful within-radius current-neighbor swap.

| Fixed bijective mapping | Mean distance | Maximum distance | Fraction within radius |
|---|---:|---:|---:|
| Original global cycle | 0.483516 | 0.708051 | 56.25% |
| Unconstrained minimum distance | 0.376853 | 0.588184 | 84.375% |
| Maximum distance | 0.666577 | 0.732118 | 0% |

Shared-epsilon sampled-code risk increase, averaged across the three seeds:

| Representation | Global cycle | Minimum distance | Maximum distance |
|---|---:|---:|---:|
| Raw | -0.0000028 | -0.0000042 | -0.0000033 |
| Global center | +0.010436 | +0.008174 | +0.006387 |
| Reference | +0.026050 | +0.017180 | +0.021016 |
| Constant | 0 | 0 | 0 |

The reference gap decreases about 34% under the minimum-distance mapping relative to the global cycle. This demonstrates sensitivity to swap construction and weakens interpreting the original global gap as a direct action-utility measure. The remaining gap does not prove action semantics: the mapping still violates the prescribed radius for ten rows, and visual proximity is not complete physical-state matching. The farthest mapping produces a smaller gap than the global cycle, so swap effects are not monotonic in this current-feature distance; different latent/content changes also matter. Constant controls behave correctly. The diagnostic supports latent dependence but cannot separate action from future scene/camera information without the additional audit in [conditional-utility-audit.md](conditional-utility-audit.md).
