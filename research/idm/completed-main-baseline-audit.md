# Completed-main original-model baseline audit

The main run has completed10000 updates. `run_summary.json` records global_step10000 and `final.ckpt`; process inspection found no matching active training process. No process was stopped or modified. The initial assumption that9802 was the final checkpoint was incorrect: it is the named checkpoint selected by the main ModelCheckpoint callback, while10000 is the final checkpoint.

The callback embedded in final10000 explicitly records `monitor=val_loss`, `mode=min`, `best_model_path=epoch=2-step=9802.ckpt`, `best_model_score=0.1622703671`. This is the main training callback's selection, not selection using the fixed research-feature MSE below. Its monitored val_loss is not the same metric as these original-posterior feature MSEs.

Both original checkpoint files were loaded with `weights_only=True` from an open file descriptor, and hashed through that same descriptor before release. File size/modification time was checked during reading. Optimizer states were omitted from the new snapshots. New immutable outputs are `runs/frozen-model-step9802` and `runs/frozen-model-step10000`, each with weights snapshot, resolved configuration, manifest and `train.pt`/`validation.pt`/`observed96.pt`.

## Fixed-data results

| Checkpoint | Dev64 original posterior | Dev64 zero code | Observed96 original posterior | Observed96 zero code |
|---|---:|---:|---:|---:|
|6501 previous baseline|.12548481|.12631256|.10944115|.10957642|
|9802 main-selected best|.11700046|.11795689|.10648947|.10724805|
|10000 final|.11865409|.11916550|.10363062|.10421857|

For the optional train128 cache,9802 original/zero MSE is .10291090/.10343318;10000 is .10215132/.10290534. The checkpoints reverse rank between dev64 and observed96. Do not select a separate best checkpoint on each evaluation set and present that as a prespecified model result. Report the main-selected9802 and final10000 separately. Existing corrections based on6501 must now be compared against these stronger original-model baselines; beating6501 alone is insufficient.

“Original posterior” means the original encoder/VAE's true-future-conditioned mu followed by its original decoder. Zero code means its decoder(current,zero). A difference between these predictions is not proof of example-specific action use: a common latent bias could explain some difference, and zero may leave the empirical latent distribution. No global or conditional code-swap claim is made by this audit.

## Numerical and data fidelity

Model config and expected-vision config match6501. The complete `lam.vision_encoder.*` state hash, including buffers, equals6501 and the recorded observed96 visual-state hash. Feature normalization remains enabled, type`ln`, with the same latent layer setting. Frozen cached h/y were reused; no images were decoded and no visual model was run. Encoder/VAE inference is medium float32, batch1; decoder inference is highest float32, batch2; CPU4. This matches the earlier baseline evaluation paths. All tensors are CPU; no GPU or training is used.

Each output preserves source h/y and rows exactly; codes and baseline now refer to its own checkpoint. Rows retain historical source metadata such as `reference_mse`, so those inherited fields must **not** be interpreted as9802/10000 scores. Correct per-row new errors are in `research/idm/model-step9802-baseline-audit.json` and `model-step10000-baseline-audit.json`.9802 uses keys`original9802_mse`/`zero9802_mse`; the general parameterized audit uses`original_model_mse`/`zero_model_mse`. Both also record the same cached6501 zero baseline error.

Independent read-only verification checks snapshot/source/output hashes, all source h/y/rows, dev/train feature equality to the original6501 cache, observed96 visual hash, finite posterior/baseline tensors, and recomputation of every zero-code MSE from saved baseline. Both pass. The audit records source code/config/weights hashes and main run summary; the final audit additionally preserves callback metadata. Original files remain unchanged. Observed96 has already been used for research and is not an independent final test.

## Reproduction

The exact initial9802 script is retained as `audit_model_step9802.py`. `audit_model_checkpoint.py` is the parameterized version used for final10000 and records its own source hash. Existing output directories are not overwritten.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python research/idm/audit_model_checkpoint.py --checkpoint /media/sadasue/SSD1/foveated-world-models/runs/stage1-gb10-main-20261112/checkpoints/final.ckpt --expected-step 10000 --output runs/frozen-model-step10000 --result research/idm/model-step10000-baseline-audit.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/verify_model_checkpoint_audit.py --step 9802
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/verify_model_checkpoint_audit.py --step 10000
```

Verification records: `model-step9802-verification.json`, `model-step10000-verification.json`.
