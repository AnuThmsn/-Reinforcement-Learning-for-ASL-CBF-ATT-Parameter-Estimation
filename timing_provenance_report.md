# Timing Provenance Report

## 1. RL Timing Provenance
The SAC model uses the following fixed delay vectors (4-PLD variant):
- `PLD_SEC = [0.700, 2.033, 3.366, 0.700]`
- `LD_SEC  = [1.333, 1.333, 1.333, 4.000]`

**Source Trace:**
These exact values are hardcoded as executable PyTorch tensors in multiple project files:
- `rl_asl.py` (lines 74-77, includes a 5th bolus: PLD 3.000, LD 4.000)
- `rl_asl_4pld.py` (lines 74-77)
- `rl-asl (6).ipynb` (lines 146-149)
- `rl-asl-4pld.ipynb` (lines 140-143)

**Origin:**
The surrounding code explicitly comments:
```python
# Multi-delay bolus timings from the base paper (Ishida et al. 2023),
# the "combined approach": (PLD [sec], LD [sec]) for boluses 1-5.
```
This confirms that the RL timing is a **legitimate, intentional simulation** of a published sequence ("combined approach", Ishida et al., JMRI 2023), rather than an arbitrary mathematical assumption.

## 2. Clinical Filename Timing Provenance
The values `1525`, `2025`, `2525`, and `3025` were traced across the entire repository.

**Source Trace:**
- These values appear **exclusively** in the raw filenames of the uploaded `test data/` folder (e.g., `perf_1525.nii`, `M0_2025.nii`).
- They also appear in the diagnostic logs and reports generated *after* the data was uploaded (e.g., `Diagnostic_Report.md`, `protocol_verification_report.md`).
- They **do not** appear anywhere in the project's source code, DNN inference scripts, documentation, or configuration files prior to the clinical inference diagnostics.

**Origin:**
The values originate solely from the user's uploaded filenames. No internal metadata, JSON sidecar, or DICOM header confirms what these numbers represent (e.g., whether they are PLD in ms, or Total Delay (PLD+LD) in ms).

## 3. DNN Timing Provenance
The existing DNN inference pipeline (`DNN_Inference.ipynb`) was inspected to determine its timing assumptions.

**Findings:**
- **DNN training PLD = Unknown.** The DNN training script is not present in the repository, only the pre-trained weights (`CBF_best.pth`, `ATT_best.pth`) are loaded.
- **DNN training LD = Unknown.** 
- **DNN clinical interpretation = None.** The DNN architecture (`BoundedNet`) is a simple Multi-Layer Perceptron (MLP) taking a 4-dimensional input. It maps the 4 normalized signal intensities directly to CBF and ATT. It **does not explicitly use PLD/LD values** or a physical forward model (Buxton) during inference. 
- **DNN signal ordering = Alphabetical.** The DNN script sorts the clinical files alphabetically (`perf_1525.nii`, `perf_2025.nii`, etc.) and feeds them directly into the 4 input nodes of the network.

## 4. Dataset / Protocol Provenance
The repository contains the following references to the data and protocols:
- **Source Paper:** "Ishida et al., JMRI 2023"
- **Sequence Name:** "combined approach"
- **Scanner Information (from clinical NIfTI headers):** GE 3T MRI (`3T 3D RM ... SO=EDR_GEMS\SPIRAL_GEMS`), indicating a 3D spiral readout.

No documentation, URL, DOI, or README exists in the project describing the newly uploaded clinical dataset in `test data/`.

## 5. Evidence Table

| Quantity | Value | Evidence | Confidence |
|----------|-------|----------|------------|
| RL PLD 1 | 0.700 | Hardcoded tensor in `rl_asl_4pld.py` line 76 | HIGH |
| RL PLD 2 | 2.033 | Hardcoded tensor in `rl_asl_4pld.py` line 76 | HIGH |
| RL PLD 3 | 3.366 | Hardcoded tensor in `rl_asl_4pld.py` line 76 | HIGH |
| RL PLD 4 | 0.700 | Hardcoded tensor in `rl_asl_4pld.py` line 76 | HIGH |
| RL LD 1 | 1.333 | Hardcoded tensor in `rl_asl_4pld.py` line 77 | HIGH |
| RL LD 2 | 1.333 | Hardcoded tensor in `rl_asl_4pld.py` line 77 | HIGH |
| RL LD 3 | 1.333 | Hardcoded tensor in `rl_asl_4pld.py` line 77 | HIGH |
| RL LD 4 | 4.000 | Hardcoded tensor in `rl_asl_4pld.py` line 77 | HIGH |
| Clinical timing 1 | 1525 | Extracted from filename `perf_1525.nii` | LOW |
| Clinical timing 2 | 2025 | Extracted from filename `perf_2025.nii` | LOW |
| Clinical timing 3 | 2525 | Extracted from filename `perf_2525.nii` | LOW |
| Clinical timing 4 | 3025 | Extracted from filename `perf_3025.nii` | LOW |

*(Note: The LOW confidence clinical timings cannot be safely assumed to be PLDs without further clarification.)*

## 6. Confirmed Facts
1. The RL agent's timings are explicitly designed to mirror the published "combined approach" protocol (Ishida et al. 2023).
2. The DNN bypasses physical modeling during inference and does not mathematically evaluate PLD/LD values at runtime.
3. The clinical filenames (`1525`, `2025`, etc.) do not correspond to the RL agent's training protocol timings (`2033`, `3366`, `4699`, `4700` total delay).

## 7. Unresolved Facts
1. We do not know if `1525` represents the Post-Labeling Delay (PLD) or the Total Delay (PLD + LD).
2. We do not know the Labeling Duration (LD) used by the clinical scanner.
3. We do not know the exact timing configuration the DNN was trained on, making it impossible to guarantee the DNN is returning physically accurate estimates for this specific clinical data.

## 8. What information is still required before retraining?

Before we can construct a scientifically valid, matched 4-PLD SAC training experiment, we must obtain the following exact information from the user or the MRI technician:

1. **Are the filename numbers (1525, 2025, 2525, 3025) exactly equal to the PLD (in milliseconds)?**
2. **What is the exact Labeling Duration (LD) used for these scans (e.g., 1000ms, 1500ms)?**

Once these two facts are definitively answered, we can overwrite `PLD_SEC` and `LD_SEC` in the training notebook and train a new SAC model perfectly matched to the clinical scanner's protocol.

