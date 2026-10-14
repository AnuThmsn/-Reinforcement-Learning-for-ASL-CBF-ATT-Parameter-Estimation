# 4-PLD SAC Baseline Experiment Report

## 1. Research Objective
The objective was to implement and test a controlled 4-PLD SAC parameter-estimation model that exactly matches the physics and parameter distributions of the established DNN reference model. This establishes a scientifically valid baseline before pursuing any targeted architectural improvements.

## 2. Why 4 PLDs
The specific `[1.525, 2.025, 2.525, 3.025]` protocol represents the controlled acquisition setup corresponding to the new parameter estimation task. Matching this configuration precisely ensures our findings isolate the performance differences between the learning methodologies (DNN vs. SAC) without conflating them with different acquisition timing or physics models.

## 3. Exact PLD/LD Configuration
- **PLD:** `[1.525, 2.025, 2.525, 3.025]` seconds
- **LD:** `[1.8, 1.8, 1.8, 1.8]` seconds

## 4. DNN Reference Pipeline
The DNN model (established via `DNN_Inference.ipynb`) directly maps normalized signal intensities to parameters using `BoundedNet` (4 hidden layers, 64 neurons). Crucially, the DNN uses hard bounds for physiological realism: `CBF [0, 100]` and `ATT [0.5, 3.0]`. The DNN does not use the forward physical model during inference.

## 5. SAC Formulation
The inverse parameter-estimation problem is framed as a sequential decision process. The agent observes the current residual between its predicted signal and the observed signal and takes a bounded step in parameter space to refine its estimate.

## 6. State Representation
The state is 11-dimensional:
- 4 observed signal values
- 4 currently predicted signal values
- 1 normalized CBF estimate
- 1 normalized ATT estimate
- 1 time/step fraction scalar

## 7. Action Representation
The action is a 2-dimensional continuous space `[ΔCBF, ΔATT]` passed through a `tanh` squash, which bounds the maximum update size per refinement step.

## 8. Reward
The dense reward is the Negative Normalized Root Mean Square Error (`-NRMSE`) of the reconstructed synthetic signal compared to the observed target signal. 

## 9. Initialization
**Method:** `coarse_grid`. The environment evaluates a fixed coarse parameter grid through the forward Buxton model and selects the configuration that minimizes the residual. This requires *no* ground-truth data, adhering to strict inference separation.

## 10. Physics Model
We implemented the strict Buxton kinetic forward model for continuous ASL:
- α (labeling efficiency) = 0.85
- β (background suppression) = 0.75
- λ (blood-brain partition coefficient) = 0.9
- T1 tissue = 1.2 s
- T1 blood = 1.66 s
- Signal scale = 100,000.0

## 11. Noise Model
A Rician noise distribution (magnitude of complex Gaussian noise) is applied to simulate true MRI acquisition noise, scaled by the `rsnr` (relative Signal-to-Noise Ratio). 

## 12. Training Data
1000 batches of synthetic pairs dynamically sampled from continuous uniform distributions:
- CBF: `[0.0, 100.0]`
- ATT: `[500.0, 3000.0]` ms

## 13. Test Data
Held-out synthetic independent batches generated identically to the training data. The ground truth was strictly withheld from the actor's state space during evaluation.

## 14. SAC Architecture
- Standard off-policy SAC
- Twin Q-value critics (Hidden=256)
- Gaussian policy actor (Hidden=256)
- Automatic entropy temperature tuning (alpha)

## 15. Training Procedure
- Iterations: 1000
- Batch size: 256
- Max Steps per episode: 12
- Replay Buffer Capacity: 1,000,000
- The model trained steadily, minimizing actor/critic losses while tuning entropy.

## 16. Evaluation Methodology
An independent test set of 2000 pairs per condition was used. Inference was executed greedily (deterministic policy) for 12 steps. 

## 17. Baseline Results
*(Results calculated automatically by script. See `evaluation_results.csv` for high-SNR clean data).*

## 18. SNR Robustness
The model's robustness was systematically evaluated across a decreasing SNR gradient (`20, 15, 10, 5, 2`).
*(See `snr_results.csv` for exact performance drops under high noise).*

## 19. Parameter-Grid Results
A deterministic grid (11x11, CBF: 0-100, ATT: 0.5-3.0) was used to measure boundary saturation and systematic bias.
*(See `parameter_grid_results.csv` for explicit reconstruction data).*

## 20. Checkpoint Validation
The `sac_asl_4pld_final.pth` checkpoint was loaded into a fresh process. Evaluation metrics matched the post-training metrics identically (`Match: True`), confirming the model saved cleanly and is reproducible.

## 21. DNN vs SAC Comparison
See the final user message for the concise tabular comparison between the DNN inference specification and this SAC baseline.

## 22. Limitations
The sequential SAC approach, while physically anchored, still requires multiple forward evaluations during inference. Extremely high noise (SNR 2) still causes some divergence because the residual signal space becomes highly non-convex.

## 23. Clinical Inference Limitations
Exploratory clinical inference (if performed) using this model carries a strict disclaimer: the acquisition timing is simulated, but clinical NIfTI header provenance remains unverified. 

## 24. Future Experiments
Based on the results, we recommend targeted architecture refinements:
1. **Residual-based state features:** Replacing absolute signal values with `(Observed - Predicted)` to explicitly guide the Q-network gradients.
2. **Reward shaping:** Adding a regularizing penalty for parameter updates that exceed physiological bounds early in the rollout.

