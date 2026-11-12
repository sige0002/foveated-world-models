# Measured displacement labels for fixed research pairs

`observed-displacement-labels/` contains `train.pt` (128), `validation.pt` (64), `observed96.pt` (96), and `manifest.json`. Extraction reads prepared NPZ state/timestamp/validity arrays and reproduces the original dataset pair index. It never calls the video-decoding dataset accessor. No probe fitting, scale calibration or predictive training was performed.

The8D target is endpoint difference at prepared-state indices[0,1,2,9,10,11,12,19]: left xyz displacement, left gripper change, right xyz displacement, right gripper change. It is not divided by dt and therefore is not velocity. It is neither an action command nor a joint-angle target. Source `meta/info.json` identifies observation end-effector xyz units as meters; each arm retains its original source coordinate frame. The metadata does not establish a shared world frame or axis transforms. Gripper values come from observation joint_position indices7/15; preprocessing checks the range[0,1]. Metadata calls them gripper_position without a physical unit, so we retain source units and do not assert aperture in meters or a particular normalized interpretation.

`current_state20` and `endpoint_states20` use two10D arm blocks: xyz, first two rotation-matrix columns (six dimensionless values from Rz(yaw)Ry(pitch)Rx(roll)), gripper. These are prepared measured states, not original Euler/joint angles. `timestamps`, original row fields, exact endpoint indices and source array paths are retained.

Masks are explicitly separate:

- `structural_state_mask32`: the production collate dimension-presence mask, first20 coordinates present and12 padding coordinates absent. This mask does not certify observations.
- `original_valid`: original endpoint validity for video plus14 raw state fields.
- `endpoint_state_observed_mask` and `current_state_observed_mask`: observed validity mapped into the prepared20D state; rotation columns require all three corresponding Euler inputs.
- `delta8_mask`: validity at both endpoints for each selected coordinate.

Missing entries would be NaN with masks retained; extraction performs no imputation. In these fixed caches, every selected endpoint is originally observed, all selected labels are finite, and fill ages are zero. The validity mask confirms rather than silently assumes this condition. Camera validity is saved separately and is not repurposed as state-coordinate validity.

The extraction verifies source cache/manifest hashes, episode/index/dt and (where already stored) endpoint indices, source state schema, all endpoint observations, zero fill age, and episode disjointness across all three sets. It records individual NPZ hashes and confirms inputs remain unchanged. An independent verification checks every original row field, exact endpoint differences, current states, shapes and masks. Synthetic tests cover index mapping, missing values/rotation dependencies, structural versus observation masks, and camera/state separation.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/extract_observed_displacements.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/idm/verify_displacement_labels.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m unittest research.idm.test_displacement_labels -v
```

Extraction refuses to overwrite its output. Three synthetic tests passed; see `displacement-label-tests.txt` and `displacement-label-verification.json`. `observed96` has already been used for research evaluation and must not be presented as untouched or external test data. Whether reference-difference z adds measured displacement information beyond current inputs remains to be tested with a matched probe and suitable controls.
