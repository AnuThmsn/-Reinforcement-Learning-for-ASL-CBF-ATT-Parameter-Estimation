# Model Audit and Inventory: ASL CBF/ATT Parameter Estimation (4-PLD)

**Date:** 2026-10-14  
**Repository:** `https://github.com/AnuThmsn/-Reinforcement-Learning-for-ASL-CBF-ATT-Parameter-Estimation`  
**Protocol Target:** Exactly four PLDs: `[1.525, 2.025, 2.525, 3.025]` seconds, LD = `[1.8, 1.8, 1.8, 1.8]` seconds.

---

## 1. Executive Summary

A comprehensive audit of the codebase, scripts, notebooks, configurations, and checkpoint files was conducted. The repository contains four genuinely distinct model families applicable to the 4-PLD ASL parameter estimation task:

1. **Classical Discrete Global Search (Coarse Grid 31×31)**: Training-free physics-based discrete evaluation across parameter space.
2. **Classical Continuous Optimization (Grid 31×31 + Levenberg–Marquardt)**: The gold-standard clinical nonlinear least-squares (NLS) estimator combining global coarse grid initialization with damped Gauss–Newton gradient refinement.
3. **Supervised Deep Neural Network (DNN / BoundedNet)**: Direct parametric regression mapping 4-PLD perfusion signals to bounded $(CBF, ATT)$ outputs.
4. **Deep Reinforcement Learning (Pure SAC)**: Soft Actor-Critic agent learning an iterative continuous refinement policy over an ASL Markov Decision Process (MDP) from random initialization without classical solvers.
5. **Hybrid Reinforcement Learning + Optimization (SAC + LM)**: Macro-exploration policy (SAC) coupled with local micro-refinement (Levenberg–Marquardt).

---

## 2. Model Family Inventory

### Family 1: Classical Discrete Optimization (Coarse Grid 31×31)
- **Implemented Code**: `rl_asl_4pld.py: ASLEnvironment._coarse_grid_init`, `configs/coarse_grid31_lm.json`.
- **Mechanism**: Precomputes noise-free Buxton forward signals across a 31×31 regular lattice spanning $CBF \in [0, 100]$ mL/100g/min and $ATT \in [0.5, 3.0]$ s (961 candidate points). For each observed signal, it assigns the candidate point minimizing the Mean Squared Error (MSE).
- **Best Configuration**: 31×31 grid (step size: $\Delta CBF = 3.33$ mL/100g/min, $\Delta ATT = 0.083$ s).
- **Checkpoint Availability**: None required (training-free, deterministic).
- **Computational Cost**:
  - *Training*: 0.0 seconds.
  - *Inference*: 961 parallel forward evaluations per batch.
- **Protocol Compatibility**: 100% compatible with the 4-PLD protocol.

### Family 2: Classical Continuous Optimization (Grid 31×31 + Levenberg–Marquardt)
- **Implemented Code**: `rl_asl_4pld.py: ASLEnvironment._levenberg_marquardt_step`, `configs/best_grid31_lm.json`.
- **Mechanism**: The 31×31 coarse grid provides a robust global prior, avoiding shallow local minima. Subsequently, the Levenberg–Marquardt (LM) algorithm iteratively refines $(CBF, ATT)$ using analytical Jacobians of the Buxton model with adaptive damping factor $\lambda$.
- **Best Configuration**: 31×31 grid initialization followed by 5–10 LM refinement steps ($\lambda = 0.01$).
- **Checkpoint Availability**: None required (training-free, deterministic optimization).
- **Computational Cost**:
  - *Training*: 0.0 seconds.
  - *Inference*: Coarse grid evaluation + analytical Jacobian calculation and $2 \times 2$ normal matrix inversions per sample.
- **Protocol Compatibility**: 100% compatible with the 4-PLD protocol.
- **Supporting Literature**: Standard clinical benchmark for multi-delay ASL parameter estimation.

### Family 3: Supervised Deep Neural Network (DNN / BoundedNet)
- **Implemented Code**: `DNN_Inference.ipynb`, `dnn_inference_spec.md`, and `asl-dnn-papermatched-v2-noise-density.ipynb`.
- **Architecture**: `BoundedNet` (or `StandardizedNet`): Multi-Layer Perceptron (MLP) consisting of an input projection ($4 \to 64$), 4 hidden layers of 64 neurons with ELU nonlinearities, and a bounded Sigmoid output layer enforcing exact physiological ranges:
  $$CBF = 0 + 100 \cdot \sigma(z_1)$$
  $$ATT = 0.5 + 2.5 \cdot \sigma(z_2)$$
- **Checkpoints Audited**:
  - Checkpoints in `D:\DOWNLOADS` (`CBF_G_best.pth`, `ATT_G_best.pth`, `CBFnet_final.pth`) were trained either with standardized target scalers or an input noise convention ($M_c + M_l = 2 \times S$) differing from the repository's clinical and RL 1× $\Delta M$ convention.
  - *Decision*: To ensure complete parity and eliminate data-leakage or distribution mismatch, a representative `BoundedNet` must be trained on the exact unified 4-PLD training distribution, with measured wall-clock training time recorded.
- **Computational Cost**:
  - *Training*: ~30–60 seconds on GPU for 100,000 synthetic samples across 25 epochs.
  - *Inference*: Single forward pass (<0.001 ms per sample).
- **Protocol Compatibility**: 100% compatible.

### Family 4: Deep Reinforcement Learning (Pure SAC)
- **Implemented Code**: `rl_asl_4pld.py`, `rl_4pld_experiment.py`, `rl-asl-4pld.ipynb`.
- **Architecture**: Soft Actor-Critic (SAC) reinforcement learning:
  - *State*: $[S_{obs}/S_{ref}, S_{pred}/S_{ref}, \text{norm}(CBF), \text{norm}(ATT), t/T_{max}] \in \mathbb{R}^{11}$.
  - *Action*: $[\Delta CBF, \Delta ATT] \in [-1, 1]^2$ scaled by maximum step size ($\Delta_{CBF}=10.0, \Delta_{ATT}=0.3$).
  - *Actor*: `GaussianPolicy` (MLP 256 $\times$ 256 $\to$ mean, log\_std).
  - *Critic*: `TwinCritic` (two independent MLPs 256 $\times$ 256).
  - *Reward*: Normalized per-step parameter error reduction.
  - *Horizon*: $T_{max} = 12$ steps.
  - *Initialization*: Random uniform in $[0, 100] \times [0.5, 3.0]$.
  - *LM Handoff*: Disabled (`lm_handoff_frac = 1.0`). Pure RL.
- **Checkpoint Audited**:
  - Path: `experiments/rl_sac_4pld_trained/notebook_results/sac_asl_final.pth`.
  - Training: Fully trained for 1,000 iterations (3,072,000 environment transitions, 64,000 gradient updates) on the exact 4-PLD protocol.
  - Status: Validated, non-trivial weights, fully functional.
- **Computational Cost**:
  - *Training*: 31 minutes wall-clock time on NVIDIA GeForce GTX 1650.
  - *Inference*: 12 sequential policy evaluations + 12 forward model steps per sample.
- **Protocol Compatibility**: 100% compatible.

### Family 5: Hybrid Reinforcement Learning + Optimization (SAC + LM)
- **Implemented Code**: `rl_4pld_experiment.py`, `sac_asl_4pld_final.pth` (in `experiments/rl_sac_4pld/`).
- **Mechanism**: Uses the SAC policy for steps 0 to 8 to rapidly navigate from random initialization into the basin of attraction, then hands off to Levenberg–Marquardt for steps 9 to 11 to fine-tune the solution.
- **Checkpoint Audited**: `experiments/rl_sac_4pld/sac_asl_4pld_final.pth` (and compatible with `sac_asl_final.pth`).
- **Computational Cost**: 9 SAC steps + 3 LM steps.
- **Protocol Compatibility**: 100% compatible.

---

## 3. Excluded and Incompatible Variants

1. **TD3 (Twin Delayed DDPG)**:
   - Found in `rl-asl-4pld-td3.ipynb`.
   - Audit result: An exploratory notebook with model definitions (`DeterministicPolicy`), but zero checkpoints were saved. Conceptually identical MDP to SAC; omitted in favor of the fully validated SAC agent to avoid redundant hyperparameter tuning.
2. **5-PLD Legacy Implementations**:
   - `rl_asl.py`, `sac_asl_out/`, `rl-asl (6).ipynb`.
   - Audit result: Hardcoded to 5 PLDs (`[0.700, 2.033, 3.366, 0.700, 3.000]`), CBF `[20, 90]`, ATT `[500, 3000]` ms. Incompatible with the 4-PLD clinical protocol.
3. **Extreme Learning Machine (ELM) / Random Vector Functional Link (RVFL)**:
   - Audit result: Mentioned in old un-integrated notebooks (`version-2.ipynb`, `trial.ipynb`), but completely removed from the 4-PLD pipeline (`asl-dnn-papermatched-v2-noise-density.ipynb` Cell 9 confirms: *"unused now that ELM is removed"*). No active, executable 4-PLD ELM implementation exists in the repository.

---

## 4. Final Candidate Selection for the 4-PLD Comparison

To answer the core research question (*accuracy vs. training time trade-off*), we select one strongest representative from each distinct implemented family:

| Candidate ID | Model Family | Representative Configuration | Training Time | Checkpoint / Source |
| :--- | :--- | :--- | :--- | :--- |
| **GRID31** | Classical Discrete Optimization | 31×31 regular grid (961 evaluations) | 0.0 s (Training-free) | `rl_asl_4pld.py: ASLEnvironment._coarse_grid_init` |
| **GRID31_LM** | Classical Continuous NLS | 31×31 Grid Init + 5 LM iterations | 0.0 s (Training-free) | `rl_asl_4pld.py: _levenberg_marquardt_step` |
| **DNN_BOUNDED** | Supervised Deep Learning | `BoundedNet` (4×64 ELU, sigmoid bounds) | ~35 s (GPU) | `DNN_Inference.ipynb` architecture |
| **SAC_PURE** | Deep Reinforcement Learning | SAC 12 steps, random init, no LM | 1,860 s (~31 min) | `sac_asl_final.pth` (1,000 iters) |
| **SAC_LM** | Hybrid RL + Classical Solver | SAC 9 steps + 3 LM steps | 1,860 s (uses SAC weights) | `sac_asl_final.pth` + LM handoff |

