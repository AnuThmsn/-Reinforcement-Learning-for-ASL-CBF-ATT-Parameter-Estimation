# Reinforcement Learning for ASL CBF and ATT Estimation

This repository contains a Soft Actor-Critic (SAC) reinforcement-learning implementation for estimating cerebral blood flow (CBF) and arterial transit time (ATT) from noisy multi-delay arterial spin labeling (ASL) signals.

## Contents

- `rl-asl (6).ipynb`: complete physics-informed SAC notebook.
- `ASL_RL_Project_Report.md`: architecture, implementation details, results, errors, corrections, and reproducibility notes.
- `sac_asl_out/`: training log, SNR sweep plot, final model checkpoint, and periodic checkpoints.

## Run

1. Open `rl-asl (6).ipynb` in VS Code or Jupyter.
2. Select a Python kernel with PyTorch installed. CUDA is recommended.
3. Run the notebook cells in order.
4. Wait for the training cell to finish.
5. Run the SNR sweep and detailed metrics cells.

The notebook writes outputs to `sac_asl_out/` relative to the current working directory.

## Current benchmark output

The latest trained run reports benchmark-compatible metrics using CBF in mL/100 g/min, ATT in seconds, and Lin's CCC:

| SNR | CBF RMSE | CBF MAE | CBF CCC | ATT RMSE | ATT MAE | ATT CCC |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2.4421 | 1.8081 | 0.9927 | 0.1129 | 0.0712 | 0.9886 |
| 15 | 1.8707 | 1.3109 | 0.9958 | 0.1005 | 0.0561 | 0.9909 |
| 20 | 1.7343 | 1.0997 | 0.9963 | 0.1028 | 0.0534 | 0.9904 |

See `ASL_RL_Project_Report.md` for the full technical explanation and known limitations.
