# Official Stage-1 fidelity audit

2026-11-13; read-only code/config audit. No training, GPU work, or running-experiment changes. Robot motion and video/pair correctness are outside this sub-audit and are being checked separately.

**Conclusion:** the baseline is a locally trained model using most of the pinned official architecture/loss code, not official released LaWAM weights or a faithful reproduction of the official training recipe. Existing research results cannot establish that the official trained model ignores actions or that official human-video training fails.

## File verification

Re-fetched all12 files from `https://raw.githubusercontent.com/RLinf/LaWAM/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/…`. Every fetched SHA256 matches `UPSTREAM.json`. Ten local files still exactly match, including `lam_decoder.py`, `lam_encoder.py`, `modules.py`, `vq.py`, `video_aug.py`, `collate.py` and the official YAML. The two changed local files, `lam_model.py` and `lam_lightinng.py`, exactly match their `RESEARCH_PATCHES.json` patched hashes. All file URLs and hashes are in [hashes.json](hashes.json); local metadata hashes are in [local-metadata-hashes.json](local-metadata-hashes.json).

`PROVENANCE.md:7–8` says all12 file contents are unchanged. That statement is now false/stale and conflicts with the explicit patch manifest. This is a documentation discrepancy, not evidence of an unrecorded third modification. No production document was edited in this audit.

The complete two-file diffs are [model](lam_model.py.diff) and [Lightning](lam_lightinng.py.diff). Changes introduce optional flow/refinement decoders, a flow-loss path and extra validation metrics. The default `decoder_type='direct'` constructs the original LAMDecoder_v2 with the same arguments and executes its original forward; model10000 resolved config does not override this default. Its direct training loss remains the original path. No active IDM/direct-decoder algorithm change was found in these diffs. Extra direct validation MSE logging changes reports, not the training objective. This is a static path audit, not a new end-to-end bitwise official reproduction.

## Training recipe is substantially different

Sources: [official pinned YAML](../../../vendor/lawam/latent_action_model/config/dino_large_vae.yaml) and [actual final10000 resolved config](../../../runs/frozen-model-step10000/resolved_config.yaml).

| Item | Official pinned config | Actual local main10000 |
| --- | --- | --- |
| Data integration | StarVLA/LeRobot mixture `lam_plus_human` | Custom manifest/timestamp RoboMIND FR3-dual adapter |
| Human data | Configured mixed human data | No human dataset in the declared run |
| Robot/human horizon | Robot1.6s; human.4s | Robot1.2s; shortened end-of-episode pairs allowed |
| Batch |64 per configured process; devices auto/DDP |2, single GPU, accumulation1 |
| Duration |10 epochs of mixture |10000 updates; dataset differs, so epochs not comparable |
| LR warmup |10000 updates |5000 updates |
| Validation |every10000, up to50 batches |every100, limit1 batch |
| Architecture |DINOv3 ViT-B16 path (despite YAML filename),1024D,24 encoder/12 decoder layers,z32 |Same configured model dimensions and backbone family |
| Loss |smooth-L1 beta.1, state L2, auxiliary weight1, VAE KL |Same main recipe; beta5e-5 explicitly supplied locally, equal to official code default |
| Augmentation flags |image_aug=true, dual_view_aug=true |Same flags and identical augmentation source |

Optimizer lr3e-4/wd.01, bias/norm exclusion, bf16-mixed and clipping1 also match. Identical backbone family/path name does not by itself independently establish official vision-weight file identity. The local adapter and upstream dataset integration differ; matching augmentation code/flags alone does not prove identical decoded/resized images or sampling distributions. See `train_stage1.py:60–68,110,124–136` and `vendor/dataset/robomind.py:96–122`.

The official YAML has much larger batches and mixed data. A10000-update local run is therefore not evidence that the official model was trained comparably. The config comparison cannot quantify official convergence or exact official sample exposure without its actual training record.

## Quantization, reconstruction, and checkpoint identity

Official `vq.py` is byte-identical. This config selects **continuous VAE**, not discrete VQ: `forward` samples `mu+sigma*epsilon` and adds beta-weighted KL; `inference(sample=False)` uses mu (lines67–109). Thus using production mu for posterior evaluation is consistent with official inference, although sampled training and mean evaluation are distinct distributions. The learned reference-difference Gaussian head and subsequent decoder pilots are additional research implementations, not the official native head.

`research/cache_stage1_features.py:73,99` freezes/evaluates the model with augmentation disabled. That is an evaluation cache, not evidence main training had augmentation disabled. Cached correction pilots also optimize feature MSE rather than the full official smooth-L1+physical+KL objective. They use selected frozen pairs, not the full official mixed data stream.

`train_stage1.py:110` instantiates VJEPA_LAM, and line136 resumes only the explicitly supplied checkpoint if any. The baseline artifact manifest identifies the locally produced `runs/stage1-gb10-main-20261112/checkpoints/final.ckpt`, step10000, SHA256 `2ac16fb97347e06342da9c7b986114edc32f974b0b9702503423c712015305e5`; it is not an official distributed LAM checkpoint. Frozen snapshot SHA256 is `1e68b8eb97c529646a7bbc9c6e02b7cf759c97b82a485241b85669bf7613a1d6`. The manifest also records a local best-checkpoint callback pointing to9802; final10000 and named9802 should remain distinct. This audit does not newly reconstruct the entire resume history.

## Data-contract issue requiring the motion/data audit

The local adapter records `original_valid` and `fill_age_seconds`, but `robomind_collate` only attaches those as extra fields. The byte-identical official `lam_collate` sets `state_mask=True` for every supplied state dimension. The local wrapper does not intersect that mask with original validity. `_compute_step` consumes `state_mask` and `delta_proprio`, not `original_valid`. Therefore any prepared filled state endpoints can contribute to the physical auxiliary loss as valid targets. Actual missingness frequency and impact require the separate data audit; this code path alone does not prove that such examples are prevalent or that video frames do not move.

`dt` is retained by the local wrapper but is not passed to the original model in `_compute_step`. End-of-episode shortened pairs consequently share the original unconditioned decoder path. Again, whether this materially affects the selected clips is a data question, not settled by the source match.

**Safe description:** “local10000-step Stage-1 baseline using the pinned official direct architecture, with custom RoboMIND data integration and a reduced training recipe.” Avoid “official pretrained baseline,” “official human-video result,” or “faithful official reproduction.”
