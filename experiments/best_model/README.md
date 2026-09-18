# Best validated configuration: coarse-grid initialization + LM

## Objective and hypothesis

The original SAC policy was intended to place the estimate in a useful basin for local LM refinement. A coarse grid search over the observable five-delay signal can provide that basin directly, without ground-truth leakage. The grid compares only forward-model predictions with `obs_signal`.

## Configuration

Use [best_grid31_lm.json](../../configs/best_grid31_lm.json): a 31x31 CBF/ATT grid, followed by the existing 3-iteration normalized-space LM solver at every one of 12 steps. `lm_handoff_frac=0.0` means SAC actions are ignored; no checkpoint is required for inference.

## Reproduction

```powershell
python rl_asl.py --config configs/best_grid31_lm.json --evaluate-only
```

The stored seed-1234 result is in `results/benchmark.json`; seeds 1235 and 1236 are documented under `exp_006_grid31_multiseed`.

## Decision

PROMOTED. Across three independently seeded evaluation simulations, SNR-10 ATT RMSE was 0.0777, 0.0783, and 0.0792 seconds. This meets the stated approximately 0.08 target while improving CBF metrics relative to the saved SAC+LM baseline.

## Scope limitation

This is a physics-based estimator, not evidence that SAC helped and not a PLD-selection policy. All five fixed signals remain available at reset.
