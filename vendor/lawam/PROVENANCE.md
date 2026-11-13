# Official Stage-1 source snapshot

Source: https://github.com/RLinf/LaWAM

Pinned commit: `c12168d9077edea67fe7c1b70d08b80b8e65a9c8`

Only the 12 files listed in `UPSTREAM.json` were originally fetched. Ten still
match the pinned source byte for byte. `lam_model.py` and `lam_lightinng.py` have
the research patches recorded in `RESEARCH_PATCHES.json`; `UPSTREAM.json` retains
their original hashes. SHA-256 digests allow both versions to be verified.
Four `__init__.py` package
markers were added locally. The earlier proposed tree was described as 13 files,
but its actual file count is 12.

The upstream root at this revision did not list a LICENSE file. No license is
assigned or changed by this extraction; preserve upstream notices and attribution.

The project-specific launcher, manifest dataset and configs live outside this
directory. They replace the upstream LightningCLI/StarVLA data integration while
reusing the official model, loss, augmentation and collator implementations.

No Stage-2 code, Git history, datasets or model weights are stored here.

The local `stage1-gb10-main-20261112` checkpoint is a locally trained RoboMIND
baseline, not the official released LaWAM weights. Its data mixture, time horizon,
batch size and training schedule differ from the pinned official configuration.
See the [source/config audit](../../research/idm/official-source-audit-20261113/report.md).
