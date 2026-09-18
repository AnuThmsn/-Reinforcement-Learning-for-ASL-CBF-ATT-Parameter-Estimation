# Experiment 002 — observable coarse-grid initialization + LM

Hypothesis: LM fails from random starts because it is local; a forward-model grid evaluated against `obs_signal` can select a valid basin without truth leakage. Change: `init_mode=coarse_grid` (21x21) and `lm_handoff_frac=0.0`. Result at seed 1234: SNR-10 CBF/ATT RMSE 2.0814 / 0.0781. Decision: PROMISING; followed by a grid-resolution and multi-seed study. Commands/configurations are stored in `configs/coarse_grid_lm*.json`.
