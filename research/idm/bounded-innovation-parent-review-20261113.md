# Bounded innovation: parent review before fitting

2026-11-13. Read-only model review; no fit launched by this review. The three large early-condition experiments are separate and remain unchanged.

## Mathematical scope

Let C and Y denote the observations sufficient to determine all frozen query inputs; the prediction target is phi(Y). Do not silently identify these observations with a particular cached feature tensor. Let B=(C,z0), and condition on fixed learned weights and calibration statistics. The auxiliary mean is m(C,Y), and W=m+epsilon with independent standard Gaussian epsilon.

For a fixed independent Gaussian reference r, adding and subtracting log q(W|B) gives

`E KL(q(W|C,Y) || r) = I(Y;W|B) + E_B KL(q(W|B) || r)`.

The hard bound `||m||²/2 <= 1` therefore bounds additional conditional information by one nat. This uses established variational information machinery, rather than a new theorem; [Deep VIB, section 3, equations 13–15](https://arxiv.org/html/1612.00410v7) supplies the underlying marginal-reference KL bound. The conditioning specialization above follows by applying that identity at each B. The old deterministic z0 has no such information bound.

The claim concerns the distribution under fresh independent sampling with fixed weights. Four fixed evaluation draws estimate the single-sample risk; they do not measure mutual information. Average four losses, not four predictions. Jointly decoding four channel samples can convey more information, and decoding m without noise has no finite information guarantee from its norm alone. For example, arbitrarily many distinct scalar means can fit inside a bounded interval while remaining distinguishable without noise.

## Initial gradient correction

Zero head output together with zero added decoder columns leaves the expected initial new-column gradient zero under independent zero-mean noise; the head gradient is also zero. Finite stochastic batches can open this path accidentally, but relying on that would confound a negative information-access result with an avoidable optimization problem.

Use a common nonzero head initialization and zero added decoder columns. Initial predictions then remain the old decoder's predictions for every auxiliary code. The new columns can receive signal from the input-dependent head mean; after those columns update, gradients can reach the head. Verify these claims with a noise-free diagnostic on a disposable model copy. The actual training still uses independent Gaussian samples and its declared untouched initialization.

## Implementation review findings

The draft `InnovationModel` sends only `[z0, m+epsilon]` to its decoder. Raw future queries, m, and epsilon are not separate decoder inputs. The original 32-dimensional affine projection is evaluated separately from the extra eight columns, preserving the original floating-point GEMM at initialization. Input normalization is fit on training rows only. The recoding block is a fixed linear embedding of z0 and thus adds no future information conditional on B, even though it can change finite-model optimization.

Requested before launch:

- Verify all old decoder tensors and initial predictions against the fixed400 reference checkpoint, using real inputs and all evaluation rows.
- Preserve cached z0 exactly and verify query/cache row and episode identities.
- Use independent common training-index and noise generators across arms. Evaluation noise must be independent of training noise.
- Match the existing continuation optimizer's matrix weight decay .01 and zero decay for bias/norm vectors; record these groups.
- Store protocol and checkpoint hashes. Distinguish active head parameters from allocated parameters in the noise control.
- Declare that seeds change head initialization, training batches, and noise; all seeds start from the same pretrained decoder.
- Keep the future/recoding/noise contrasts as finite-predictor utility tests. None proves that z0 lacks information or that the extra channel represents physical actions.

The implementation is being prepared in an isolated worktree. This review does not certify the still-pending final preflight or any accuracy improvement. No additional rate, update-count, or evaluation-dependent model selection is introduced here.
