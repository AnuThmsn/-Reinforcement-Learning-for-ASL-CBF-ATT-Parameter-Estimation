# Experiment 003 — random-initialization LM-only ablation

Objective: isolate the contribution of LM without SAC or improved initialization. Change: only `lm_handoff_frac=0.0`; initialization remains random. Result: SNR-10 CBF/ATT RMSE 10.3308 / 0.4722. Decision: REJECTED. This verifies that local LM needs a good initializer.
