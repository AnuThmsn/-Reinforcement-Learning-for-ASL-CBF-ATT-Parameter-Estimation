# Protocol Verification Report

## Task 1 — Verify Actual Clinical PLD/LD

A comprehensive inspection of all available sources for the clinical ASL dataset was performed.

### Findings by Source:
*   **NIfTI Headers (`descrip` field):** The headers contain sequence parameters but **do not** contain PLD or LD values.
    *   `perf_1525.nii`: `3T 3D RM TR=4751ms/TE=10.728ms/FA=111deg/SO=EDR_GEMS\SPIRAL_GEMS`
    *   `perf_2025.nii`: `3T 3D RM TR=4963ms/TE=10.728ms/FA=111deg/SO=EDR_GEMS\SPIRAL_GEMS`
    *   `perf_2525.nii`: `3T 3D RM TR=5446ms/TE=10.728ms/FA=111deg/SO=EDR_GEMS\SPIRAL_GEMS`
    *   `perf_3025.nii`: `3T 3D RM TR=5681ms/TE=10.728ms/FA=111deg/SO=EDR_GEMS\SPIRAL_GEMS`
*   **Filenames:** The filenames contain the numerical values `1525`, `2025`, `2525`, and `3025`. While this strongly suggests a timing parameter (either PLD or PLD+LD in milliseconds), the filenames alone do not definitively distinguish between PLD and Total Delay, nor do they specify the Labeling Duration (LD).
*   **JSON Sidecars / DICOM Metadata:** No JSON sidecars or DICOM files were present in the `test data/` directory or the repository.
*   **Acquisition Documentation / README:** The project README and `ASL_RL_Project_Report.md` document the *synthetic* training protocol (Ishida et al.), but contain no documentation regarding the acquisition protocol of the newly uploaded clinical `test data`.
*   **Existing DNN Notebooks:** The `DNN_Inference.ipynb` notebook relies on a pre-trained model located in a Google Drive folder (`MODEL_DIR`). The notebook expects 4 perfusion files but never explicitly references PLD or LD variables in the code.

### Exact Mapping:
Based strictly on available metadata without assumptions:
*   Clinical perfusion volume 1 → PLD = **Unknown** (Filename suggests 1525)
*   Clinical perfusion volume 2 → PLD = **Unknown** (Filename suggests 2025)
*   Clinical perfusion volume 3 → PLD = **Unknown** (Filename suggests 2525)
*   Clinical perfusion volume 4 → PLD = **Unknown** (Filename suggests 3025)

*   LD 1 = **Unknown**
*   LD 2 = **Unknown**
*   LD 3 = **Unknown**
*   LD 4 = **Unknown**

---

## Task 2 — Verify the Current SAC Timing

The `rl_asl_4pld.py` and `rl-asl-4pld.ipynb` executable code explicitly define the constants used by the Buxton forward model during SAC training. 

Exact values (in seconds):
*   `PLD_SEC = torch.tensor([0.700, 2.033, 3.366, 0.700])`
*   `LD_SEC = torch.tensor([1.333, 1.333, 1.333, 4.000])`

---

## Task 3 — Compare

| Parameter | SAC training | Clinical acquisition |
|-----------|--------------|----------------------|
| PLD 1 | 0.700 s | Unknown (Filename: 1525?) |
| PLD 2 | 2.033 s | Unknown (Filename: 2025?) |
| PLD 3 | 3.366 s | Unknown (Filename: 2525?) |
| PLD 4 | 0.700 s | Unknown (Filename: 3025?) |
| LD 1 | 1.333 s | Unknown |
| LD 2 | 1.333 s | Unknown |
| LD 3 | 1.333 s | Unknown |
| LD 4 | 4.000 s | Unknown |

---

## Task 4 — Check Existing DNN

The `DNN_Inference.ipynb` script loads a pre-trained 4-layer BoundedNet (`CBFnet` and `ATTnet`) from external weight files (`CBF_best.pth`, `ATT_best.pth`). Because neural networks are model-free estimators once trained, the inference code merely passes the 4 normalized signal features through the network without needing a physical forward model or PLD/LD definitions.

Furthermore, the DNN training notebook is **not present in the repository**. Therefore, the exact PLD/LD configuration the DNN was trained on cannot be determined from the available files.

*   DNN training PLDs = **Unknown (Training code not in repo)**
*   DNN clinical PLDs = **Unknown (Implicitly matched to training)**
*   RL training PLDs = **[0.700, 2.033, 3.366, 0.700]**
*   RL clinical PLDs = **Unknown (Filenames: 1525, 2025, 2525, 3025)**

---

## Task 5 — Do Not Modify Any Model

No models, architectures, checkpoints, clinical maps, parameters, or physics models were modified during this verification.

---

## Task 6 — Final Decision

**C. Timing cannot be established reliably from available metadata.**

While the filenames (`1525`, `2025`, `2525`, `3025`) strongly imply a standard multi-delay sequence, without DICOM headers, JSON sidecars, or a protocol document, we cannot definitively state whether these numbers represent PLD alone, or PLD + LD. More importantly, the LD (Labeling Duration) is entirely missing from the metadata.

Because the true clinical acquisition parameters cannot be verified, **the current clinical SAC maps must be treated only as a diagnostic demonstration of protocol mismatch, and NOT as scientifically valid quantitative estimates.** The SAC model was trained on a highly specific and complex multi-delay protocol (Ishida et al.), which clearly differs from whatever standard clinical protocol generated the test data.

Before the SAC model can be scientifically applied to this clinical data, the exact clinical PLD and LD values must be obtained from the scanner operator or original DICOM headers, and the SAC model must be retrained using those specific values.

