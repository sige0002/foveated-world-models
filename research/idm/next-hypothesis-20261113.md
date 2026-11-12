# Next hypothesis: retain measured-motion utility, penalize known observer perturbations

**Recommendation:** test a small, target-aware linear projection of the existing IDM features before another large encoder redesign. Select directions that add prediction of measured8D displacement beyond current information, while penalizing response to prescribed image-coordinate perturbations. Compare it against the *same supervised projection without the nuisance penalty*. This is a falsifiable combination of established regression/invariance ideas, not a novelty claim.

## What remains unresolved

Reference codes improved the specified displacement ridge probe beyond pooled-current/state inputs; extending the alpha grid did not remove the gain. However, pooled current is weak, codes can contain nonlinear current information, and neither this result nor a code-swap gap establishes action identity. Postprojection statistics/raw heads transmit much more future dependence than direction-only, but their Gaussian KL can equally increase because of nuisance. Pinsker only limits an artificial single-sample discrimination problem; it does not certify useful content.

The parent reports final10000-prior corrections across three IDM/decoder seeds at mean MSE.1036242 on dev64 and.0906967 on observed96, versus original10000 .1186541/.1036306. These are encouraging posterior reconstruction results on already observed research data. They do not resolve whether the added information is robot motion, appearance, or camera. The next intervention should address that missing distinction, not maximize future KL.

## A concrete projection with an interpretable objective

Let f be frozen reference-query features, C the current visual features plus current20D state, and A the measured8D endpoint displacement. On training folds only, fit current-only predictors g(C) for f and m(C) for A. Define e=f−g(C), r=A−m(C), with training-only target units/scales. Use out-of-fold residuals to estimate covariance; independently evaluate the complete fitted pipeline on heldout episodes.

For a scalar projected code u=wᵀe, its *population linear* improvement in squared prediction error for centered r is exactly

`G(w) = wᵀ B w / (wᵀ Σee w),  B = Σer Σre`.

This is a regression identity, not a causal claim. If m(C)=E[A|C], a code depending only on C has zero covariance with r. With estimated/underfit m, that protection fails; cross-fitting prevents same-example fitting leakage but does not make the residual orthogonal to every current-dependent function.

Create paired, independently prescribed small image-coordinate transforms at the two endpoints, with A unchanged. Let e_g be the entire residual pipeline evaluated on that transformed pair, including its transformed-current g(C_g). Estimate `N=E[(e_g−e)(e_g−e)ᵀ]`. Select at most8 directions by the generalized eigenproblem

`B w = λ (Σee + κ N + ε I) w`.

The numerator rewards measured-displacement covariance; N penalizes actual code response to the known perturbation, rather than all future sensitivity. The supervised matrix has rank≤8 because this experiment has8 targets; pad an8D output to32 if needed for an identical downstream head. This is not a claim that all useful actions have8 dimensions. The added denominator terms regularize the regression criterion; the eigenvalue is not itself the unpenalized prediction gain. Use a training-data low-rank basis/dual computation to avoid an expensive1024D covariance solve.

**Single primary comparison:** κ=0 supervised projection versus κ>0 nuisance-penalized projection, equal output dimension, calibration, training episodes, downstream model and optimization budget. A constant code remains the no-extra-information control. Select κ/regularization with train-only folds, then freeze them. Do not choose directions from validation future labels. Do not replace ordinary full-displacement grounding with only the residual objective: retain ordinary supervised projection as the essential control.

## Feasibility and decisive rejection conditions

Existing paired state labels, query extraction and probe infrastructure are reusable. New work is a *small offline paired augmentation cache*: decode selected endpoints, apply predeclared modest translations/rotations/crops, and rerun frozen DINO/IDM. **Warping cached DINO patches is not a justified substitute** for rerunning the visual encoder. Common visible regions must retain both grippers; transformed borders/crop loss must be audited. No such extraction or training is performed in this proposal.

The primary endpoint is additional heldout measured-displacement prediction gain from C+z over a matched C-only predictor, including a stronger spatial/nonlinear current baseline; code-swap gaps are secondary. Report both full displacement and near-static subsets with training-defined bins. Simultaneously report paired augmentation sensitivity and posterior reconstruction versus the frozen final10000 prior. Reject if nuisance sensitivity falls only by collapsing physical prediction; if κ=0 matches the tradeoff; if a stronger current predictor removes the gain; or if ordinary grounding is as effective. Larger KL, code variance or swap gaps alone are not success.

Image-coordinate perturbations cover only the prescribed2D transformation family. They do not reproduce real head translation, depth-dependent parallax, SE(3) motion, occlusion/disocclusion or view-dependent appearance. A cropped-out interaction invalidates the assumed label-preserving intervention. Actual camera motion may correlate with physical action; suppressing that signal could hurt. Consequently, reduced N supports robustness only to these interventions. Real moving-head human validation and multiview/depth evidence are separate requirements. Human clips lack A: this projection is initially robot-supervised; transfer to unlabeled humans is unproven.

## Primary-source overlap checked on2026-11-13

- [WING](https://arxiv.org/html/2610.03607v1), §3.2 and appendix C.1.3, already uses camera interventions and consistency of interaction representations; camera-invariance consistency alone is not a contribution. Its motion-routed teacher also separates observer and interaction motion. Our proposed measured-target generalized-eigen projection is a simpler controlled experiment, not a claimed superior replacement.
- [Why Latent Actions Fail, and How to Prevent It](https://arxiv.org/html/2605.20223v3), §§4.2,4.4–4.5, shows future-exogenous leakage, relates exogenous-swap reconstruction to CCA under assumptions, and analyzes supervision with nuisance-robust targets. The covariance/eigen mathematics above belongs to that established family; real exogenous independence is not supplied by our observations.
- [JEVA](https://singhularadi.github.io/jeva/static/jeva.pdf), §§3.4–3.5, already combines action grounding with Gaussian-latent training and variance-reduced routing. Merely adding physical targets or using posterior means is not new. Here measured endpoint displacement is weaker than a commanded action and misses orientation/contact/trajectory information.

The most defensible potential contribution is an audited *physical-utility versus nuisance-sensitivity tradeoff*, with a mechanism linking normalization to accessible conditional signal and failures ruled out by matched controls. Whether that yields a top-conference contribution remains unestablished. Current scope: a small, testable next hypothesis; no new implementation, fitting or GPU use.
