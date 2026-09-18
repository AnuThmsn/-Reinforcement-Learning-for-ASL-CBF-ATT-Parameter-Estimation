# Experiment 004 — random-initialization SAC-only ablation

Objective: isolate the saved SAC actor by disabling LM. Change: `lm_handoff_frac=1.0`; random initialization retained. Result: SNR-10 CBF/ATT RMSE 5.9004 / 0.1670. Decision: REJECTED. SAC-only is materially worse than SAC+LM and coarse-grid+LM.
