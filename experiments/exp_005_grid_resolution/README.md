# Experiment 005 — coarse-grid resolution

Objective: test whether initializer resolution, rather than a broad redesign, controls final LM quality. Compared 15x15 and 31x31 grids with all other settings fixed at seed 1234. Results: 15x15 produced SNR-10 CBF/ATT RMSE 2.0961 / 0.0820; 31x31 produced 2.0928 / 0.0777. Decision: 31x31 selected for validation because it improves ATT without a CBF trade-off.
