# DNN Inference Specification

This specification was extracted directly from `DNN_Inference.ipynb`.

## 1. File Discovery
- **Perfusion Volumes:** Located by searching for files containing `perfusion` (case-insensitive) and ending with `.nii.gz` within the subject directory. They are sorted alphabetically.
- **M0 Volumes:** Located by searching for files containing `m0` (case-insensitive) and ending with `.nii.gz`. Sorted alphabetically.

## 2. M0 Processing
- The 4 loaded M0 volumes are averaged across the 4th dimension (time/delay axis) to create a single mean M0 map:
  `M0_mean = np.mean(M0_vols, axis=-1)`

## 3. Masking & Valid Voxels
- Valid voxels are determined by the intersection of the provided anatomical mask (`mask` in filename) and a threshold of `M0_mean > 100`.

## 4. Feature Extraction & Scaling
- For each valid voxel, the 4-dimensional perfusion vector (`dm`) is scaled by `100.0 * m0` (where `m0` is the voxel's value in the `M0_mean` map).
- The raw feature vector is: `features = dm / (100.0 * m0)`

## 5. Input Normalization
- The raw features are Z-score normalized using pre-computed statistics loaded from disk (`X_mean (1).npy` and `X_std (1).npy`):
  `features_n = (features - X_mean) / X_std`
- These normalized features are converted to PyTorch tensors and passed to the neural networks.

## 6. Network Architectures & Bounds
- **CBF Network:** `BoundedNet(input_dim=4, n_hidden_layers=4, n_neurons=64, y_min=0.0, y_max=100.0)`
- **ATT Network:** `BoundedNet(input_dim=4, n_hidden_layers=4, n_neurons=64, y_min=0.5, y_max=3.0)`
- *Note:* The `BoundedNet` uses a sigmoid output layer scaled to `[y_min, y_max]` to strictly enforce physical parameter boundaries.

## 7. Outputs & Post-processing
- The model outputs are directly mapped back to the 3D voxel coordinates `CBF_map` and `ATT_map`.
- Voxels outside the valid mask are left as `np.nan`.
- The outputs are saved as `CBF_pred.nii.gz` and `ATT_pred.nii.gz` using the affine from the original anatomical mask.

---
**Conclusion:** The DNN maps 4 alphabetical perfusion signals (normalized by `100 * mean(M0)`) directly to CBF and ATT without an explicit physical forward model containing PLD or LD values.

