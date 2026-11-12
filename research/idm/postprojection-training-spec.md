# Prepared joint-head comparison: not launched

`train_postprojection_heads.py` jointly trains a new34→256→64 Gaussian head and the unmodified `prior_residual` factory (dim128, layers2). Each representation receives independent copies of exactly equal initial head/decoder weights. Encoder Q/p, train-only calibration and current-only step6501 baseline remain fixed. Source and representation hashes and full row identity are checked before execution; train128/validation64 counts are enforced. The96-episode cache is not accessed by this harness.

Fixed first comparison:400 updates, batch8, AdamW lr3e-4, matrix weight decay.01 (bias/norm0), global gradient clipping1, highest float32 CPU. Loss is sampled-posterior reconstruction MSE plus5e-5 times mean(sum_code_dimension KL(q||N(0,I))). Outputs are32mu and32logvar, clamped to[-10,10]. This differs from the production head path, production SmoothL1 objective and original decoder; it cannot establish same-condition improvement of the production head.

For each seed, all three variants use the same index/noise schedules and exact initial weight hashes. Final checkpoints, final weight hashes, complete train128 and validation64 mean-code MSE and fixed-epsilon sample MSE, baseline MSE, losses and provenance are saved. Only the fixed final checkpoint is evaluated; no validation selection or calibration refitting occurs. The mean-code and sample metrics answer different questions and remain separate.

The factory initializes output and AdaLN modulation to zero. Initial predictions therefore equal baseline and MSE initially cannot train the head; KL can. This is deliberately retained identically across variants. At updates1,10,100,400 the harness records head MSE gradient norm, raw KL gradient norm and weighted KL gradient norm using `autograd.grad(... retain_graph=True)`. It does not touch `.grad` or RNG state. Synthetic tests verify exact same optimizer update with and without the diagnostic. The subsequent normal backward/update remains unchanged. Gradient clipping rejects nonfinite norms before optimizer updates or checkpoint saves. Missing early head MSE gradients must not be interpreted as a failed representation.

Prepared invocation, **not executed**:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/train_postprojection_heads.py --output research/idm/postprojection-joint400 --threads 2 --seeds 42 --steps 400 --batch-size 8 --lr 3e-4 --beta 5e-5
```

Checks actually executed:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m unittest research.idm.test_postprojection_training -v
```

All8 synthetic checks passed: initial baseline equality, KL reduction/frozen inputs, gradient path after zero gates open, equal initial weights/schedules, clamp, per-example deterministic validation logging, diagnostic RNG/gradient/update neutrality, and rejection of nonfinite gradients. They use tiny generated tensors and never load research training examples. A separate read-only call to `load_inputs` verified actual hashes/rows and34D train/validation shapes; record: `postprojection-harness-input-check.json`. No real-data training or96-episode evaluation was run.

## Optional constant posterior control

Add `--include-constant-control` to append a fourth `constant` variant using all-zero34D inputs for every train/validation example. Existing three representations and saved caches remain unchanged. It uses exactly the same head/decoder parameter count, initial weights, index sequence and posterior epsilon sequence. Posterior mu/logvar are therefore identical across examples, while sampled z can differ because epsilon differs. The current-conditioned decoder remains free to learn; this tests whether additional decoder training with a jointly learned but example-independent posterior can explain an apparent gain. It is not equivalent to forcing z=0. The additional synthetic test verifies zero34 inputs, identical posterior parameters across rows, equal initial weights/capacity and nonmutation of original feature mappings. All9 harness tests pass. Real-data training remains unstarted.
