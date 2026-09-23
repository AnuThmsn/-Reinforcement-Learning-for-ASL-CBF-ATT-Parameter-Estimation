# RL-ASL Experiments Summary & Architecture Evolution

This document outlines all the tracked experiments run in this repository (found in the `experiments/` directory), their results, and the scientific justification behind the architectural changes made during the project.

---

## 1. `EXP_00_baseline` (Pure Non-RL Numerical Solver)

### Overview
This experiment bypasses the Reinforcement Learning agent entirely. It utilizes a **31x31 Coarse Grid Search** followed by a differentiable **Levenberg-Marquardt (LM)** solver.

### Architecture
- **State/Action Space:** None (No RL).
- **Initialization:** Brute-force evaluates a 31x31 grid of possible CBF and ATT combinations to find the lowest error.
- **Refinement:** The LM solver takes the best grid guess and mathematically optimizes it using `torch.func.jacrev` (Jacobians).

### Results
- **Final Mean Error:** ~0.121 (Normalized distance)
- **Conclusion:** This purely numerical approach is extremely accurate (achieving CBF RMSE ~3.5 at SNR 10) but is computationally expensive at inference time due to the exhaustive grid search. It serves as the "Gold Standard" baseline.

---

## 2. `rl_sac_trained` (The 5-PLD SAC Agent - Original DPAC)

### Overview
This is the core Reinforcement Learning experiment utilizing the **Soft Actor-Critic (SAC)** algorithm in a DPAC (Deep Policy and Actor-Critic) setup.

### Architecture
- **Acquisition Protocol:** 5 Post-Labeling Delays (PLDs). The 5th bolus features a very long delay (`w=3.0s, tau=4.0s`).
- **State Space:** 13-Dimensional (5 observed signals + 5 predicted signals + current CBF guess + current ATT guess + time).
- **Handoff Mechanism:** The SAC agent acts for the first 80% of the episode (`lm_handoff_frac=0.8`). It takes large, decaying steps to quickly approximate the physical parameters. At the end of the episode, it hands its guess over to the LM solver for final numerical refinement.

### Results
- **Final Mean Error:** ~0.1238 (at iteration 1000)
- **Conclusion:** The SAC agent successfully learns to map noisy 5-PLD MRI signals to physical parameters. By replacing the 31x31 grid search with the RL agent, the pipeline becomes significantly faster at inference time while maintaining accuracy close to the numerical baseline.

---

## 3. `rl_sac_4pld_trained` (The 4-PLD SAC Architecture Shift)

### Overview
We deliberately handicapped the acquisition protocol by dropping the 5th, long-delay bolus to see if the RL agent could still accurately estimate blood flow.

### Architecture Changes
- **Acquisition Protocol:** Reduced to 4 PLDs.
- **State Space:** Reduced from 13-D to 11-D (4 observed + 4 predicted + 3 state variables).
- **Why We Made This Change:** In a clinical setting, spending less time in the MRI scanner (by acquiring 4 PLDs instead of 5) reduces patient discomfort and motion artifacts. We needed to scientifically quantify exactly how much accuracy we sacrifice by speeding up the scan. 

### Results
- **Final Mean Error:** ~0.1377 (at iteration 1000)
- **Conclusion:** While Cerebral Blood Flow (CBF) estimation remained relatively stable, **Arterial Transit Time (ATT) estimation suffered massive degradation**, especially under high noise (SNR=10). Because the 4-PLD protocol lacks the crucial late-arrival information provided by the 5th bolus, the agent struggles to distinguish between low CBF and very slow transit times. 
- **Final Verdict:** This architecture change proved that the 5-PLD protocol (Ishida et al. 2023) is biologically necessary for robust parameter mapping.

---

## 4. `best_model`
*(Reserved directory for storing final production weights; currently unused as individual experiment checkpoints serve as the source of truth).*

---

## Summary of the RL Architecture Pipeline
Across the evolution of these experiments, our architecture matured into the following pipeline:
1. **Procedural Generation:** The environment (`ASLEnvironment`) simulates millions of theoretical human brains using the Buxton kinetic model, injecting Rician noise at runtime. This makes data-overfitting impossible.
2. **SAC Agent (Phase 1):** A Twin-Delayed Actor-Critic network reads the noisy signal and rapidly steps the CBF/ATT estimates toward the true value.
3. **LM Refinement (Phase 2):** A differentiable physics solver takes over for the final few steps to hone the precision mathematically, resulting in the final NIfTI output maps.

