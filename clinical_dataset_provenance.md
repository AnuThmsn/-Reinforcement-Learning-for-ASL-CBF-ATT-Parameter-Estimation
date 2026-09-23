# Clinical Dataset Provenance Report

## 1. Test-Data Location
The clinical test dataset is located in the local directory:
`D:\reinforcement learning\asl-github-repo\test data`

## 2. File Inventory
The directory contains data for a single patient across 13 NIfTI files:
- **Perfusion (ΔM):** `perf_1525.nii`, `perf_2025.nii`, `perf_2525.nii`, `perf_3025.nii`
- **Calibration (M0):** `M0_1525.nii`, `M0_2025.nii`, `M0_2525.nii`, `M0_3025.nii`
- **Masks & Structural:** `brain_mask.nii`, `struc.nii`
- **Pre-computed Maps:** `ATTC.nii`, `ATTmapfinal.nii`, `CBFfinal.nii`

## 3. Metadata Available
NIfTI header inspection (`inspect_nifti.py`) reveals the following scanner metadata:

**Dimensions & Resolution:**
- Perfusion/M0: `(128, 128, 44)`, Voxel size: `[1.7188, 1.7188, 4.0] mm`
- Structural: `(512, 512, 296)`, Voxel size: `[0.4883, 0.4883, 0.6000] mm`

**Scanner & Sequence (from `descrip` field):**
All files show a GE 3T 3D spiral sequence acquired on `07-Nov-2024`.
- `perf_1525.nii`: `3T 3D RM TR=4751ms/TE=10.728ms/FA=111deg/SO=EDR_GEMS\SPIRAL_GEMS`
- `perf_2025.nii`: `3T 3D RM TR=4963ms/TE=10.728ms/FA=111deg/SO=EDR_GEMS\SPIRAL_GEMS`
- `perf_2525.nii`: `3T 3D RM TR=5446ms/TE=10.728ms/FA=111deg/SO=EDR_GEMS\SPIRAL_GEMS`
- `perf_3025.nii`: `3T 3D RM TR=5681ms/TE=10.728ms/FA=111deg/SO=EDR_GEMS\SPIRAL_GEMS`

*(Note: TR increases as the filename numerical value increases. No PLD or LD fields are present in the headers).*

## 4. References Found
**None.** 
A comprehensive search of the repository for keywords (`dataset`, `DOI`, `patient`, `clinical`, URLs) found **zero** references documenting the origin of this specific dataset. 

The only references found in the codebase (e.g., "Ishida et al., JMRI 2023") describe the *synthetic training protocol*, not the clinical test data.

## 5. Dataset Origin
**C. Supplied by a researcher**

The dataset was uploaded locally by the user. It is not downloaded from a public repository, Zenodo, OpenNeuro, or a documented GitHub release, and it does not originate from the Ishida et al. paper used to construct the simulation environment.

## 6. Acquisition Information Found
- **Scanner:** GE 3T
- **Readout:** 3D Spiral (`SPIRAL_GEMS`)
- **TR/TE/FA:** TR varies (4751ms - 5681ms), TE = 10.728ms, FA = 111°
- **Date:** 07-Nov-2024
- **PLD / LD:** **Unknown.** The numerical values (1525, 2025, 2525, 3025) appear only in filenames. 

## 7. Evidence Supporting Each Claim
- **NIfTI inspection:** Output of `inspect_nifti.py` confirms the headers contain only GE scanner `descrip` strings without specific ASL timing parameters.
- **Code search:** A regex search for URLs, DOIs, and keywords across all `.py`, `.ipynb`, and `.md` files confirmed the total absence of documentation for `test data/`.
- **Ishida et al. mismatch:** The repository cites Ishida et al., JMRI 2023 as the source for the training protocol. The Ishida protocol explicitly uses 5 PLDs (`0.700, 2.033, 3.366, 0.700, 3.000`) and LDs (`1.333, 1.333, 1.333, 4.000, 4.000`). This completely contradicts the 4 volumes and the `1525/2025/2525/3025` pattern seen in the clinical test data.

## 8. Missing Information
The repository lacks:
1. DICOM files or JSON sidecars that contain the sequence ground-truth parameters.
2. A README or protocol document describing the patient scan.
3. The exact PLD (Post-Labeling Delay) array.
4. The exact LD (Labeling Duration).

---

### FINAL QUESTION: What exact document, metadata, or source do we need from the dataset provider/source to determine the true PLD and LD?

**Repository provenance exhausted; external acquisition documentation is required.**

To determine the true PLD and LD, we must obtain one of the following directly from the MRI technician or the clinical researcher who acquired the data:
1. **The original DICOM files** (which contain the private GE ASL tags for labeling duration and post-labeling delay).
2. **A BIDS-compliant JSON sidecar** (generated during dcm2niix conversion, which extracts `PostLabelingDelay` and `LabelingDuration`).
3. **The official MRI Protocol PDF** from the scanner console for this specific sequence.

