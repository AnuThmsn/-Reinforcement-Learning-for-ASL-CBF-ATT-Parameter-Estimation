# Verification Report: 4-PLD SAC Parameter Estimation

## Final Verdict
**PARTIALLY VERIFIED**

## Summary of Findings
1. **Checkpoint Identity**: The file `sac_asl_4pld_final.pth` is indeed a byte-for-byte copy of the intermediate checkpoint `sac_asl_it500.pth`. The training did *not* reach 1000 iterations as defined by `max_iterations`; it was interrupted at iteration 590.
2. **Physics Configuration**: The physics parameters exactly match the controlled 4-PLD specification (PLD=[1.525, 2.025, 2.525, 3.025], LD=[1.8, 1.8, 1.8, 1.8], CBF=[0,100], ATT=[0.5,3.0]).
3. **Levenberg-Marquardt (LM) Usage**: **LM IS ACTIVE**. The evaluation runs `max_steps=12` with `lm_handoff_frac=0.8`. This means for the final 3 steps (steps 9, 10, 11), SAC actions are ignored and Levenberg-Marquardt optimization takes over. The reported metrics reflect **SAC + LM**, not SAC alone.
4. **Determinism**: Inference over fixed test populations is perfectly deterministic. Reloading checkpoints or predicting multiple times on the same data yields absolute differences of `0.00e+00`.
5. **Metric Reproducibility**: The SNR metrics reported in the previous run can be perfectly reproduced when re-running the checkpoint on the same generated distributions, but variance exists due to fresh seeds if not fixed.
6. **Training Progression**: The progression from `it100` to `it500` demonstrates changing weights and valid loss reductions, proving the model was genuinely trained up to that point rather than fabricated.

---

## 1. Code & Checkpoint Hashes
(See `verification/code_hashes.txt`)
- `sac_asl_it500.pth` and `sac_asl_4pld_final.pth` have identical SHA256 hashes.
- This confirms the prior claim that `it500` was promoted to `final`.

## 2. Training Log Verification
(See `verification/training_log_verification.txt`)
- The log contains contiguous entries from iteration 10 to 590.
- Iteration 500 exists. Iteration 1000 does not exist.
- The training was genuinely interrupted at iteration 590.

## 3. LM (Levenberg-Marquardt) Inspection
**CRITICAL FINDING**: The evaluation uses **SAC + LM**.
- `lm_handoff_frac` is `0.8`.
- `max_steps` is `12`.
- LM handoff occurs at `int(12 * 0.8) = 9`.
- **Steps 0 to 8**: SAC macro-exploration.
- **Steps 9 to 11**: SAC is ignored; the differentiable LM solver refines the actor's handoff prior.
- Therefore, the final MAE/RMSE results measure the hybrid `SAC + LM` solver, not a pure SAC policy.

## 4. Metric Reproducibility
(See `verification/reported_vs_recomputed.csv`)
Using fixed seeds, we regenerated test sets for SNR 2, 5, 10, 15, and 20.
The recomputed values closely align with the previously reported metrics, with minor variance entirely explained by the randomized noise arrays generated during the previous (unseeded) evaluation run.

## 5. Parameter Grid Verification
(See `verification/recomputed_parameter_grid.csv`)
The 11x11 CBF/ATT grid was deterministically re-evaluated. The model demonstrates boundary limits consistent with the defined `[0, 100]` and `[0.5, 3.0]` constraints.

## 6. Training Progression Proof
(See `verification/checkpoint_progression.csv`)
Checksums of the actor parameters for `it100, it200, it300, it400, it500` demonstrate continuous modification. The model weights changed at every checkpoint interval, confirming active gradient updates.

## 7. Random Initialization Comparison
An untrained SAC model (randomly initialized weights) evaluated on the exact same SNR 20 test population yields:
- Random Init CBF MAE: **3.129**
- Trained Checkpoint CBF MAE: **2.078**
- Random Init ATT MAE: **0.354**
- Trained Checkpoint ATT MAE: **0.194**

This proves that the trained SAC actor does learn to provide a significantly better hand-off prior to the Levenberg-Marquardt optimizer than a random walk, reducing final estimation error by ~35-45%.

## Conclusion
The results reported previously were generated from a genuine run of the exact code in `rl_4pld_experiment.py` using the correct 4-PLD physics specification. The weights were actively learned (as verified by progression checksums) and demonstrably outperform a randomly initialized network. 

However, the designation "SAC Baseline" is a misnomer, as the final 3 evaluation steps utilized the Levenberg-Marquardt algorithm. The reported metrics reflect a hybrid `SAC + LM` pipeline, where the trained SAC agent serves strictly as a macro-exploration prior for the LM optimizer. Any subsequent experiments or DNN comparisons must explicitly acknowledge this hybrid architecture.
