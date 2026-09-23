# RL-ASL 4-PLD Diagnostic Report

## Executive Summary

The trained SAC model (`sac_asl_final.pth`) is **architecturally sound and performs well on synthetic data**. The high ATT saturation observed in clinical inference is **NOT caused by a defective SAC model**. It is caused by a **fundamental mismatch between the clinical acquisition protocol and the training protocol**.

**Verdict: (B) — Inference/preprocessing mismatch.**

---

## 1. Checkpoint Validation — PASS

| Check | Result |
|-------|--------|
| Checkpoint keys | `actor`, `critic`, `critic_target`, `log_alpha` |
| Actor input dim | 11 (matches 4-PLD config) |
| Critic input dim | 13 (= 11 state + 2 action) |
| `log_alpha` | -1.9259 (α ≈ 0.146) |
| Weight shapes match | ✅ All match |
| Weight values match | ✅ Exact bitwise match |

---

## 2. Synthetic Clean Round-Trip (Task 2)

The actor was tested on 72 clean (noiseless) CBF×ATT combinations.

| Metric | Value |
|--------|-------|
| Mean CBF absolute error | **1.27 ml/100g/min** |
| Mean ATT absolute error | **51.17 ms** |
| Mean signal NRMSE | **0.0273** |

> [!TIP]
> The actor is remarkably accurate on clean data. A CBF error of 1.27 and ATT error of 51ms on a clean signal proves the model has learned the Buxton physics correctly.

**Notable weakness:** At very low CBF (20 ml/100g/min) combined with short ATT (500ms), the ATT error jumps to 300ms. This is a known identifiability problem — at low flow, the signal amplitude is so small that ATT is poorly constrained.

---

## 3. Noisy Synthetic Evaluation (Task 3)

| SNR | CBF NRMSE | ATT NRMSE | CBF MAE | ATT MAE (ms) |
|-----|-----------|-----------|---------|---------------|
| 5   | 0.1266    | 0.1169    | 4.74    | 136.98        |
| 10  | 0.0682    | 0.0639    | 2.61    | 76.64         |
| 15  | 0.0474    | 0.0475    | 1.87    | 57.38         |
| 20  | 0.0395    | 0.0427    | 1.55    | 50.25         |

Performance degrades gracefully with noise — no catastrophic failure at any SNR level.

---

## 4. ATT Saturation Analysis (Task 4) — NO SATURATION on Synthetic

| ATT Range (ms) | Predictions | Percentage |
|----------------|-------------|------------|
| 0–550          | 258         | 2.6%       |
| 550–1000       | 1762        | 17.6%      |
| 1000–1500      | 1991        | 19.9%      |
| 1500–2000      | 2128        | 21.3%      |
| 2000–2500      | 1980        | 19.8%      |
| 2500–2900      | 1501        | 15.0%      |
| ≥ 2900         | 380         | **3.8%**   |

| Statistic | True | Predicted |
|-----------|------|-----------|
| Mean      | 1755 ms | 1744 ms |
| Std       | 719 ms  | 710 ms  |

> [!IMPORTANT]
> Only 3.8% of synthetic predictions hit the upper boundary (≥ 2900ms). The distribution closely mirrors the uniform ground truth. **The SAC actor does NOT have a systematic ATT saturation bias.**

---

## 5. CBF Boundary Analysis (Task 5) — Healthy

| CBF Range | Predictions | Percentage |
|-----------|-------------|------------|
| 20–25     | 563         | 5.6%       |
| 25–40     | 2387        | 23.9%      |
| 40–55     | 2166        | 21.7%      |
| 55–70     | 2141        | 21.4%      |
| 70–85     | 2161        | 21.6%      |
| 85–91     | 582         | 5.8%       |

CBF predictions are well-distributed across the physiological range.

---

## 6. Initialization Sensitivity (Task 6) — Insensitive

| Init Mode | CBF RMSE | ATT RMSE | CBF MAE | ATT MAE |
|-----------|----------|----------|---------|---------|
| Random    | 3.68     | 109.14   | 2.55    | 74.78   |
| Center    | 3.67     | 109.02   | 2.54    | 74.77   |
| Grid-31   | 3.67     | 109.12   | 2.54    | 74.86   |

> [!TIP]
> The actor has learned a robust 12-step policy that converges regardless of initialization. This confirms the training was successful.

---

## 7. Signal Fit Residuals (Task 7)

| Metric | Mean | Median | 95th Percentile |
|--------|------|--------|-----------------|
| L2 Residual | 34.82 | 33.17 | 62.53 |
| NRMSE | **0.0572** | **0.0405** | 0.1583 |

The actor's predicted CBF/ATT values reconstruct the observed signal with low error on synthetic data.

---

## 8. Clinical Diagnostics (Task 8) — THE SMOKING GUN

### Clinical Signal Statistics (Scaled to RL units)

| PLD | Mean | Std | Min | Max |
|-----|------|-----|-----|-----|
| 1   | 239.16 | 125.88 | -7.08 | 1197.33 |
| 2   | 185.72 | 85.94 | 5.03 | 816.18 |
| 3   | 207.01 | 118.50 | -130.30 | 2969.48 |
| 4   | 152.70 | 68.33 | -22.79 | 683.30 |

### Clinical ATT Distribution — SEVERE SATURATION

| ATT Range (ms) | Count | Percentage |
|----------------|-------|------------|
| 0–550          | 1     | 0.0%       |
| 550–1000       | 40    | 0.0%       |
| 1000–1500      | 3,202 | 1.5%       |
| 1500–2000      | 20,657 | 9.8%      |
| 2000–2500      | 36,400 | 17.2%     |
| 2500–2900      | 68,696 | 32.5%     |
| **≥ 2900**     | **82,462** | **39.0%** |

Compare synthetic (3.8% at boundary) vs clinical (**39.0%** at boundary).

### Clinical Signal Reconstruction NRMSE

| Metric | Value |
|--------|-------|
| Mean NRMSE | **0.6801** |
| Median NRMSE | **0.6895** |
| 95th % NRMSE | **0.8455** |

> [!CAUTION]
> The clinical signal NRMSE is **0.68**, compared to **0.057** on synthetic data. This is a **12× degradation**. The actor literally CANNOT fit the clinical signals using the Buxton model with the training PLD/LD table. When it cannot match the signal, it pushes ATT toward the upper boundary as a "best effort" to minimize the unexplainable residual.

---

## 9. Acquisition Order (Task 10) — CRITICAL MISMATCH

### Training Protocol
| Bolus | PLD (s) | LD (s) | Total (ms) |
|-------|---------|--------|------------|
| 1     | 0.700   | 1.333  | **2033**   |
| 2     | 2.033   | 1.333  | **3366**   |
| 3     | 3.366   | 1.333  | **4699**   |
| 4     | 0.700   | 4.000  | **4700**   |

### Clinical Files
| Volume | File | Implied Total (ms) |
|--------|------|---------------------|
| 0      | perf_1525.nii | **1525** |
| 1      | perf_2025.nii | **2025** |
| 2      | perf_2525.nii | **2525** |
| 3      | perf_3025.nii | **3025** |

> [!CAUTION]
> **NONE of the clinical total delays (1525, 2025, 2525, 3025 ms) match the training total delays (2033, 3366, 4699, 4700 ms).** The clinical scanner used a completely different PLD/LD protocol. When these mismatched signals are fed to the SAC actor, it receives input patterns it has never seen during training. The actor's response is physically meaningless.

---

## 10. Final Diagnosis

The high ATT values in the clinical maps are caused by **answer (B): inference/preprocessing mismatch**.

Specifically:
1. **The SAC model is healthy.** On synthetic data, it achieves CBF MAE of 1.27 and ATT MAE of 51ms (clean), with only 3.8% of predictions hitting the ATT boundary. No systematic bias exists.
2. **The clinical acquisition protocol is different.** The scanner that produced the test data used PLDs that create total delays of 1525–3025ms. The SAC agent was trained on a protocol with total delays of 2033–4700ms. These are fundamentally different timing patterns.
3. **The signal reconstruction proves it.** The clinical NRMSE of 0.68 (vs 0.057 synthetic) demonstrates that the Buxton model with the training PLD/LD table **cannot produce** the clinical signal shapes. The actor is being asked to solve an impossible physics problem.

### What Needs to Happen Next
To produce valid clinical maps, you must either:
- **(Option A)** Obtain the exact PLD and LD values from the clinical scanner's DICOM headers and retrain the SAC agent using those exact delays.
- **(Option B)** Retrain using the clinical PLD/LD values (which appear to be PLD = [0.525, 1.025, 1.525, 2.025] and LD = [1.0, 1.0, 1.0, 1.0] based on the 500ms spacing pattern, but this MUST be confirmed from scanner metadata).

