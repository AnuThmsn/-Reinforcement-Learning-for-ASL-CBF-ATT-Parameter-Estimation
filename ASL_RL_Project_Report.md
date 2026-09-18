# SAC Reinforcement Learning for Synthetic ASL CBF/ATT Estimation

## Executive Summary

This project studies estimation of cerebral blood flow (CBF) and arterial transit time (ATT) from five noisy, fixed ASL measurements. The current notebook contains a genuine Soft Actor-Critic (SAC) training pipeline. The final verified run trained an actor, twin critics, and entropy temperature through replay-buffer updates for 1,000 iterations, saved a checkpoint, and evaluated the deterministic trained policy on a fresh synthetic test population.

The final RL estimator is deliberately separated from the earlier physics-only estimator. In the final RL configuration, each episode starts from a random CBF/ATT estimate, the SAC actor outputs continuous CBF/ATT increments, and LM refinement is disabled by setting the handoff step to the 12-step horizon. The current final RL result is therefore not `ZeroActionAgent`, not grid search, and not LM.

The verified final independent test output was:

| SNR | CBF RMSE (mL/100 g/min) | CBF MAE | CBF CCC | ATT RMSE (s) | ATT MAE (s) | ATT CCC |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2.4249 | 1.8639 | 0.9925 | 0.0881 | 0.0609 | 0.9923 |
| 15 | 1.8894 | 1.4287 | 0.9957 | 0.0710 | 0.0480 | 0.9952 |
| 20 | 1.4706 | 1.1054 | 0.9973 | 0.0590 | 0.0382 | 0.9965 |

These values were produced by the final report cell in the executed notebook using `TEST_SEED = 20260919` and 2,000 fresh samples per SNR. They are not claimed to be superior to the non-RL baseline. The notebook also reports a separate observable 31x31 grid-initializer ablation. Historical grid-plus-LM values recorded in older documentation are not treated as current notebook RL evidence.

The main scope limitation is important: all five PLD/LD measurements are available before the first action. The agent estimates CBF and ATT iteratively, but it does not choose which PLD to acquire, skip measurements, stop early, or minimize acquisition time. This is SAC-based iterative parameter estimation, not adaptive PLD reduction.

## 1. Problem Definition

Arterial spin labeling (ASL) is a non-invasive MRI technique that labels inflowing blood magnetically and measures the resulting signal change. The two target parameters in this project are:

- **CBF**: cerebral blood flow, reported in mL/100 g/min.
- **ATT**: arterial transit time, the time required for labeled blood to reach tissue, represented internally in milliseconds and reported in seconds in the benchmark tables.

The simulated observation contains five ASL measurements made with different post-labeling delays (PLD) and labeling durations (LD). The task is an inverse problem:

```text
CBF + ATT
    |
    v
Buxton kinetic model
    |
    v
five ASL signals
```

The implemented task reverses that relationship:

```text
five noisy ASL signals
    |
    v
estimated CBF + ATT
```

The inverse problem is difficult because the Buxton relationship is nonlinear, the two parameters affect the signal jointly, the signal is noisy, and different parameter pairs can produce similar signals. The multiple timing pairs provide different views of the delivery and decay process, which helps separate flow magnitude from arrival time. At low SNR, however, the inverse mapping remains uncertain and local optimization can depend strongly on initialization.

## 2. Simulated Data Preparation

The current project uses fully simulated data. It does not train or test on clinical scanner data. Each environment reset creates a new batch of synthetic examples:

1. CBF is sampled uniformly from 20 to 90 mL/100 g/min.
2. ATT is sampled uniformly from 500 to 3000 ms.
3. The Buxton model converts each sampled CBF/ATT pair into five clean signals using the fixed PLD/LD table.
4. An SNR is sampled uniformly from 2 to 20 during training.
5. Ten percent of training samples are assigned zero added noise through `clean_fraction = 0.10`.
6. For the remaining samples, independent Gaussian real and imaginary noise is generated using the calibrated reference-signal scale.
7. The magnitude operation produces the observed five-value noisy signal.
8. The environment creates an independent random CBF/ATT starting estimate for the SAC episode.
9. The observed and predicted signals are normalized by the reference signal before entering the policy state, while CBF and ATT are normalized to `[-1, 1]` for state representation and error calculation.

The simulator keeps the sampled true CBF and ATT internally so it can calculate reward and evaluate the estimate. Those true values are not included in the policy state and are not supplied to the actor as inputs. The policy sees the five observed signals, the five signals predicted by its current estimate, the current normalized estimate, and normalized time progress.

Training and testing are generated procedurally rather than stored as a fixed supervised dataset. During training, each of the 1,000 iterations resets 256 parallel environments, producing new synthetic batches. The replay buffer stores the resulting transitions, not a precomputed label table. Periodic evaluations use fresh batches with `clean_fraction = 0.0`.

The final independent test uses `TEST_SEED = 20260919`, separate from the training seed `SEED = 20260918`, and evaluates 2,000 newly generated samples at each of SNR 10, 15, and 20. The test population is therefore generated by the same physical simulator but is not the training batch or replay-buffer data. This is a synthetic held-out simulation test, not an external clinical validation set. Multiple training seeds and real-data validation would still be required to establish generalization beyond this simulator.

The simulated-data pipeline is:

```text
sample true CBF and ATT
          |
          v
Buxton forward model with five fixed PLD/LD pairs
          |
          v
clean five-signal ASL observation
          |
          v
Rician-like magnitude noise at sampled SNR
          |
          v
observed signal and hidden simulator ground truth
          |
          +--> policy state uses observed signal, not hidden truth
          |
          +--> reward and final metrics use hidden truth inside the simulator
```

This preparation supports controlled experiments because the simulator can vary SNR and the true physiological parameters while keeping the forward model and timing protocol fixed. Its limitation is that any mismatch between this Buxton/Rician-like simulator and real scanner physics will not be learned automatically by SAC.

## 3. Physics Model

### 3.1 Timing table

The notebook uses five fixed timing pairs:

| Measurement | PLD (s) | LD (s) |
|---:|---:|---:|
| 1 | 0.700 | 1.333 |
| 2 | 2.033 | 1.333 |
| 3 | 3.366 | 1.333 |
| 4 | 0.700 | 4.000 |
| 5 | 3.000 | 4.000 |

PLD is the post-labeling delay between labeling and readout. LD is the duration of the labeling pulse. Different PLD/LD combinations change how much labeled blood has arrived and how much longitudinal relaxation has occurred. A single timing pair generally does not contain enough independent information to robustly identify both CBF and ATT, which is why the simulation uses five measurements.

### 3.2 Buxton forward model

The notebook implements a vectorized single-compartment Buxton signal model. In simplified notation, the signal is proportional to

$$
\Delta M = \frac{2\alpha\beta T_{1b}}{\lambda} f e^{-ATT/T_{1b}}
\left[e^{-\max(PLD-ATT,0)/T_{1t}}
-e^{-\max(LD+PLD-ATT,0)/T_{1t}}\right].
$$

Here:

- $f$ is CBF converted from mL/100 g/min to mL/g/s;
- $ATT$ is converted from milliseconds to seconds;
- $T_{1b}$ is blood longitudinal relaxation time;
- $T_{1t}$ is tissue longitudinal relaxation time;
- $\alpha$ is labeling efficiency;
- $\beta$ is background-suppression efficiency;
- $\lambda$ is the blood-tissue partition coefficient.

The notebook multiplies the model output by `SIGNAL_SCALE = 100000` for numerical convenience. That scale changes signal magnitude, not the underlying parameter relationship.

### 3.3 Physics constants and assumptions

The following values are simulation assumptions used by the notebook. The report does not claim that they are universally optimal or scanner-specific calibrations.

| Constant | Value | Meaning and role |
|---|---:|---|
| `ALPHA` | 0.85 | Labeling efficiency. It scales how much arterial magnetization is inverted. Lowering it reduces signal amplitude and makes estimation less informative at the same noise level. |
| `BETA` | 0.75 | Background-suppression efficiency. It scales suppression of static tissue background. Changing it changes the signal amplitude and therefore the effective inverse-problem conditioning. |
| `LAMBDA_BLOOD` | 0.9 | Blood-tissue partition coefficient. It appears in the signal denominator and converts the labeled blood contribution to the modeled tissue signal. |
| `T1_TISSUE` | 1.2 s | Tissue longitudinal relaxation time. It controls decay during the PLD/LD timing terms. A different value changes the timing dependence and can shift the inferred ATT. |
| `T1_BLOOD` | 1.66 s | Blood longitudinal relaxation time. It controls blood decay after arrival and appears in the scaling coefficient. A substantially different value changes both signal magnitude and arrival-time sensitivity. |
| `SIGNAL_SCALE` | 100000 | Numerical scale used for the simulated signal. It improves numerical convenience but is not a physiological parameter. |
| CBF range | 20 to 90 | Simulation range in mL/100 g/min. It defines the support from which synthetic ground truth and initial estimates are sampled. |
| ATT range | 500 to 3000 | Simulation range in ms. It defines the supported arrival-time population. |

The ranges are project simulation choices based on the intended physiological operating region. The notebook does not contain a sensitivity study proving that these exact endpoints are optimal. If the ranges were changed, the training distribution, action scale, normalization, and reported difficulty would all change.

## 4. Noise Model

MRI magnitude data are nonnegative and are commonly modeled by taking the magnitude of two noisy quadrature channels. The notebook implements:

```text
clean signal + Gaussian real noise
clean signal + Gaussian imaginary noise
                    |
                    v
       square root of summed squares
                    |
                    v
          Rician-like magnitude signal
```

For each sample, the clean five-bolus signal is generated first. Independent Gaussian noise is then added to real and imaginary channels, and the magnitude is computed as:

$$
y = \sqrt{(s+n_r)^2+n_i^2}.
$$

The reference signal is computed at the project reference condition: PLD 2.0 s, LD 1.8 s, CBF 50, and ATT 1600 ms. The current calibration is:

```python
RICIAN_NOISE_FACTOR = 2.0
sigma = ref_signal / (RICIAN_NOISE_FACTOR * rsnr)
```

The earlier convention used `ref_signal / rsnr` for each independent channel. That treated the per-channel standard deviation as though it were already the intended total complex-noise scale. Because the magnitude combines two independent channels, that convention produced excessive effective magnitude noise relative to the intended SNR definition. Dividing by two is the implemented calibration correction. The current notebook uses the corrected convention; the report does not claim that the factor is a universal MRI calibration.

## 5. Why an Iterative RL Formulation?

The current formulation does not ask a network to output the final CBF and ATT in one shot. Instead, an episode begins with a random estimate and repeatedly modifies it. This makes the problem an iterative continuous-control problem:

```text
current estimate
      |
      v
SAC action: [CBF increment, ATT increment]
      |
      v
updated estimate
      |
      v
new predicted signal and new error
      |
      v
next state and reward
```

This formulation is technically compatible with SAC because SAC is designed for continuous actions. A supervised DNN could be a reasonable alternative for one-shot regression, but it would solve a different formulation. Here the policy learns an update strategy over a finite horizon.

The actor does **not** select a PLD. It also does not learn the Buxton equation itself. The physics model defines the environment transition and predicted signal; SAC learns how to change the current parameter estimate to improve future reward.

## 6. State Representation

The notebook defines `state_dim = 13`. The state is:

1. Five observed signal values, normalized by the reference signal.
2. Five predicted signal values from the current CBF/ATT estimate, normalized by the same reference signal.
3. The current CBF estimate normalized to the range `[-1, 1]`.
4. The current ATT estimate normalized to `[-1, 1]`.
5. The normalized step count, from zero toward one.

The observed signals tell the policy what measurement it is trying to explain. The predicted signals tell it what the current estimate would produce. Their difference is therefore available implicitly through the pair, without requiring a separately constructed residual feature. The current parameter estimate tells the policy its location in parameter space, which is necessary because the same signal mismatch can require different corrections at different locations. The time feature tells the policy how much update budget remains.

All of these values are available to the simulator at inference time. The state does not contain the true CBF, true ATT, or any hidden ground-truth error. The true parameters are used by the environment only to generate rewards and evaluation metrics.

## 7. Action and Environment Transition

The action is a two-dimensional bounded vector:

$$
a_t = [a_{CBF}, a_{ATT}], \qquad a_t \in [-1,1]^2.
$$

The environment scales the components by:

- `STEP_CBF = 10.0` mL/100 g/min;
- `STEP_ATT = 300.0` ms.

The estimate is updated and clipped to the physiological range. The action is therefore an increment, not a direct final prediction. This makes the actor responsible for a sequence of corrections and allows reward feedback after each correction.

The action does not select a PLD, does not request a new measurement, and does not change the five timing pairs. All five observed signals exist before the first action.

## 8. SAC Implementation

### 8.1 Actor

The actor is a Gaussian policy with two hidden layers of width 256. For each state it outputs a mean and log standard deviation for each action dimension. A stochastic Gaussian policy is useful during training because the agent must explore different CBF/ATT corrections. At evaluation time the notebook calls the actor deterministically, using the squashed mean action.

The sampled action is transformed with `tanh`, which guarantees the action is in `[-1, 1]`. This is important because the environment uses the action as a bounded parameter increment.

### 8.2 Critics

The twin critic contains two independent Q-functions. Each receives the concatenated state and action and estimates expected discounted future return. SAC uses the minimum of the two target Q-values:

$$
y = r + (1-d)\gamma\left(\min(Q_1^{target},Q_2^{target})-\alpha\log\pi(a'|s')\right).
$$

The minimum reduces optimistic overestimation from using one learned Q-function. The critic loss is the sum of the two mean-squared Bellman errors.

### 8.3 Replay buffer and target critics

Each transition stores state, action, reward, next state, and the absorption flag. Batches of transitions are reused after collection rather than learning only from the most recent step. This improves sample reuse and decorrelates updates.

The target critics are slowly updated with:

$$
\theta_{target} \leftarrow (1-\tau)\theta_{target}+\tau\theta.
$$

The slow update reduces instability caused by using a rapidly changing critic to construct its own targets.

### 8.4 Entropy and temperature

The actor objective includes entropy:

$$
L_{actor}=\mathbb{E}[\alpha\log\pi(a|s)-\min(Q_1,Q_2)].
$$

The Q term encourages high-return actions. The entropy term encourages exploration. The temperature `alpha` is learned using a target entropy of `-action_dim`, with the implementation clamped between `0.05` and `1.0`. It is not fixed because the useful exploration level can change during training.

### 8.5 Discount factor

`gamma = 0.95` discounts future reward. In this problem, a correction is valuable not only for its immediate reduction in error but also because it places the estimate in a better position for later corrections. The selected value is an engineering choice for the 12-step horizon, not a proven optimum.

### 8.6 Historical SAC-to-LM path

Earlier notebook and runner versions experimented with a two-stage path:

```text
SAC = learned macro/global parameter updates
LM  = deterministic local physics refinement
```

The idea was that SAC could move a random estimate into a useful basin, after which a local optimizer could reduce the remaining forward-model residual. For normalized parameters, the LM update was based on:

$$
(J^T J + \lambda\,\operatorname{diag}(J^T J))\delta = J^T(y-\hat{y}).
$$

Here $y$ is the observed five-signal vector, $\hat{y}$ is the signal predicted by the current estimate, and $y-\hat{y}$ is the residual. $J$ is the Jacobian of the forward-model signal with respect to normalized CBF and ATT. $\delta$ is the local parameter correction. $\lambda$ is a damping coefficient; multiplying the diagonal of $J^TJ$ by $\lambda$ stabilizes the normal-equation solve when the two parameters are poorly conditioned or the local signal surface is nearly flat.

LM is local: it linearizes the forward model around the current estimate, solves a damped least-squares step, updates the estimate, and repeats. It does not learn a policy, use a replay buffer, or update actor/critic weights. Therefore an SAC-to-LM result cannot be reported as SAC-only accuracy unless the final estimate is produced entirely by SAC actions. The final executed RL configuration disables LM by setting `env.lm_handoff_step = max_steps`; the equation above documents the historical ablation, not the final RL model.

## 9. Reward Design

The current environment computes:

$$
r_t = 10(e_{t-1}-e_t)+2\,CCC-0.02.
$$

The first term rewards improvement rather than merely rewarding a low absolute error. This gives the policy feedback about whether its latest action helped. The factor 10 controls the scale of that learning signal.

CCC is Lin's concordance correlation coefficient. It combines correlation, scale agreement, and mean agreement. The factor 2 adds a batch-level agreement term to the per-sample error-improvement term.

The `-0.02` step cost discourages unnecessary movement and makes shorter successful correction sequences preferable when accuracy is otherwise comparable.

The notebook's CCC is computed across the parallel batch, not independently for each sample. Therefore it is a population-level structural bonus shared by the batch. This is a limitation: it can couple the reward for different synthetic samples and is not a pure per-sample agreement measure.

The implementation should not be described as using potential-based shaping in the strict theoretical sense. It uses an error-improvement term plus a batch-level CCC term and a step cost.

## 10. Training Configuration Actually Executed

The executed training cell used:

| Parameter | Value | Role |
|---|---:|---|
| Iterations | 1000 | Number of batched episodes/iterations. |
| Parallel environments | 256 | Number of synthetic samples advanced together per episode. |
| Horizon | 12 steps | Number of continuous parameter updates per episode. |
| Training SNR | Uniform range 2 to 20 | Noise difficulty distribution during training. |
| Clean fraction | 0.10 | Fraction of training samples with zero added noise. |
| Initialization | Random CBF/ATT | Prevents the final RL result from using the grid initializer. |
| LM handoff | Disabled | `env.lm_handoff_step = max_steps`; SAC acts on all 12 steps. |
| Hidden width | 256 | Width of actor and critic hidden layers. |
| Learning rate | 0.0003 | Adam learning rate for actor, critics, and alpha. |
| Discount | 0.95 | Future-return weighting. |
| Target update | 0.005 | Polyak target-critic update rate. |
| Replay capacity | 1,000,000 | Maximum stored transitions. |
| Training batch | 256 | Transitions sampled per update. |
| Updates/iteration | 64 | SAC gradient updates after a training iteration is eligible. |
| Warm-up | 25 iterations | Random actions before policy actions and updates. |
| Minimum buffer | 5,000 | Minimum transitions before SAC updates. |
| Evaluation pairs | 2,000 | Fresh samples used for periodic evaluation. |
| Log interval | 10 iterations | Training-log frequency. |
| Checkpoint interval | 100 iterations | Intermediate checkpoint frequency. |

These values are engineering choices. The notebook does not establish that they are globally optimal. The requested phrase “10 SAC steps and 2 LM steps” does not describe the final executed configuration: the final notebook uses 12 SAC-controlled steps and no LM refinement.

## 11. Verified Training Evidence

The notebook executed the training cell and printed logs from the live run. During iterations 1 through 25, the replay buffer was warming up, so actor and critic losses were not yet updated. After the buffer threshold was reached, the losses became numeric and the policy updates ran.

Representative executed logs were:

| Iteration | Mean reward | Final parameter error | Critic loss | Actor loss | Alpha |
|---:|---:|---:|---:|---:|---:|
| 10 | -1.446 | 1.0304 | not updated | not updated | 1.0000 |
| 50 | 25.472 | 0.2233 | 1.4171 | -10.5233 | 0.6377 |
| 100 | 26.972 | 0.1600 | 1.1852 | -2.6189 | 0.2697 |
| 500 | 29.365 | 0.1028 | 244.0925 | 65.8758 | 0.0859 |
| 1000 | 29.274 | 0.0955 | 84.2887 | 36.6766 | 0.2400 |

These logs demonstrate that the training loop collected transitions, began replay updates after warm-up, produced actor and critic losses, and changed the entropy temperature. They are evidence of SAC optimization, not proof that every update improved every metric monotonically. The critic loss is noisy, which is expected in off-policy value learning.

The checkpoint and CSV log were written to:

```text
experiments/rl_sac_trained/notebook_results/sac_asl_final.pth
experiments/rl_sac_trained/notebook_results/train_log.csv
```

The saved checkpoint contains actor, critic, target-critic, and `log_alpha` state dictionaries. The final report cell used the in-memory trained `SACAgent` after training; it did not use `ZeroActionAgent` or load an old checkpoint.

## 12. Independent Test Protocol and Verified RL Results

The final report cell reset the random generators to `TEST_SEED = 20260919`, generated a new environment population with zero clean-fraction, and called the trained actor deterministically. It used 2,000 samples for each of SNR 10, 15, and 20. The true CBF/ATT values were hidden from the policy and used only after rollout for metrics.

The verified final SAC output was:

| SNR | CBF RMSE | CBF MAE | CBF CCC | ATT RMSE (s) | ATT MAE (s) | ATT CCC |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2.4249 | 1.8639 | 0.9925 | 0.0881 | 0.0609 | 0.9923 |
| 15 | 1.8894 | 1.4287 | 0.9957 | 0.0710 | 0.0480 | 0.9952 |
| 20 | 1.4706 | 1.1054 | 0.9973 | 0.0590 | 0.0382 | 0.9965 |

RMSE is the square root of mean squared error. MAE is mean absolute error. CCC measures agreement, including bias and scale, in addition to association. ATT is converted from the notebook's internal milliseconds to seconds before reporting.

The executed deterministic SNR sweep also generated a fresh plot and reported:

| SNR | SAC CBF NRMSE | SAC ATT NRMSE |
|---:|---:|---:|
| 2 | 0.2377 | 0.2423 |
| 5 | 0.0853 | 0.1022 |
| 8 | 0.0547 | 0.0688 |
| 12 | 0.0390 | 0.0460 |
| 16 | 0.0330 | 0.0356 |
| 20 | 0.0279 | 0.0357 |

The plot was saved as `experiments/rl_sac_trained/notebook_results/sac_nrmse_vs_snr.png`.

## 13. Baseline and Invalid-Experiment Audit

### Verified non-RL baseline/ablation

The current notebook contains a separate `evaluate_grid_baseline` function. It uses the five observed signals and a 31x31 forward-model grid to select an initial CBF/ATT candidate. It is not used by `train`, `evaluate`, the SAC SNR sweep, or the final SAC report. The current final report cell prints it separately:

| SNR | Grid-only CBF RMSE | Grid-only ATT RMSE (s) |
|---:|---:|---:|
| 10 | 2.2766 | 0.0793 |
| 15 | 1.7360 | 0.0625 |
| 20 | 1.4669 | 0.0530 |

This baseline is useful for comparison, but it is not RL and must not be presented as evidence that SAC achieved those values.

### Historical or unverified values

Older report text and the `experiments/` documentation contain SAC+LM, random-start LM, random SAC-only, and grid-plus-LM values. Those values were generated by earlier code paths or external runner configurations and are not all reproducible from the current notebook. They are therefore historical/unverified for this report. They are not used as current SAC evidence.

In particular:

- `ZeroActionAgent` was used in an earlier notebook state and is not a trained RL agent.
- Grid-search plus LM is a physics-based non-RL estimator, not an RL result.
- The old SAC+LM metrics with ATT RMSE around 0.11 are not the current final SAC output.
- Any experiment without current notebook execution evidence is excluded from the verified-results table.

## 14. Project Evolution: From Initial Approach to Final RL Model

| Stage | Approach | Problem observed | Change | Result/status |
|---|---|---|---|---|
| 1 | Buxton forward simulation with fixed five-delay data | Needed a realistic inverse-problem environment rather than arbitrary labels | Added batched CBF/ATT generation and forward predictions | Retained as the physics foundation. |
| 2 | Direct per-channel noise scale `ref_signal / SNR` | Magnitude noise was too large under the intended SNR convention | Added `RICIAN_NOISE_FACTOR = 2.0` | Retained in current notebook. |
| 3 | Random-start local LM diagnostic | Local optimization is sensitive to initialization and can fail from a poor basin | Tested physics-based initialization as a separate baseline | Retained only as non-RL comparison logic; not the final RL pipeline. |
| 4 | SAC formulation with random-start iterative actions | Needed a learned policy that could make sequential continuous corrections | Used 13-D state and 2-D bounded CBF/ATT increments | This became the final RL formulation. |
| 5 | Earlier SAC-to-LM notebook path | It made it easy to confuse LM performance with policy performance because actions were ignored after handoff | Final executed notebook disables LM by setting handoff to the horizon | Final RL result is SAC-only; older SAC+LM claims are historical/unverified. |
| 6 | Reward based on error improvement, batch CCC, and step cost | Absolute error alone gives weaker directional feedback for each action | Implemented `10 * (previous-current) + 2 * CCC - 0.02` | Used in the verified 1,000-iteration run; batch-level CCC remains a limitation. |
| 7 | Periodic independent evaluation during training | Training reward alone does not establish parameter accuracy | Evaluated fresh 2,000-sample batches every 25 iterations and after training | Produced live evaluation logs and final independent test output. |
| 8 | Final clean RL run | Needed proof that SAC, rather than a baseline, produced the result | Ran 1,000 iterations, saved checkpoint/log, and tested with seed 20260919 | Verified final RL result reported above. |

### Old approach -> issue -> hypothesis -> modification -> decision

- **Noise:** The old per-channel convention overstated magnitude noise. The hypothesis was that the SNR calibration was inconsistent with two-channel magnitude formation. Dividing the per-channel scale by two produced the current corrected convention. Retained.
- **LM from random starts:** The local optimizer had no global exploration guarantee. The hypothesis was that an observable forward-model initializer could find a better basin. This produced a useful non-RL baseline, but it did not make the method RL. Kept separate.
- **SAC plus LM:** A hybrid can be useful when SAC supplies a global prior and LM performs local refinement. However, if LM dominates the final estimate, the result cannot be attributed to SAC. The final notebook therefore disables LM for the final RL result. Historical hybrid results are not used as final SAC evidence.
- **SAC-only random start:** This is the current genuine RL experiment. The actor controls all 12 updates, and the replay buffer and SAC optimizer change the policy. The verified result is good but not claimed to beat the grid baseline.

## 15. What the RL Model Actually Learns

The SAC agent is not learning the Buxton equation. The Buxton model is written explicitly and remains fixed. The environment uses it to turn a candidate CBF/ATT estimate into a predicted signal.

- **Physics model:** maps CBF and ATT to the five predicted signals.
- **Environment:** generates noisy observations, stores the current estimate, applies actions, and computes rewards.
- **Actor:** learns which bounded CBF/ATT increment to apply for a state.
- **Critic:** learns the expected future return of a state-action pair.
- **Replay buffer:** stores and resamples prior transitions.
- **Optimizer:** updates actor, critics, and entropy temperature.
- **LM:** belongs only to earlier/historical hybrid paths or separate baselines; it is disabled in the final RL path.

In plain terms, the learned behavior is an update policy: given the observed signal, the current predicted signal, the current parameter estimate, and the remaining step budget, it learns how to move the estimate so that future parameter-error rewards improve.

## 16. Current Scope vs Intended PLD-Reduction Goal

The current implementation uses five fixed PLD/LD measurements and performs CBF/ATT estimation from them. It does not:

- select the next PLD;
- skip a measurement;
- stop acquisition early;
- optimize acquisition time;
- minimize the number of measurements.

All five signals are present in the state before the first action. Consequently, this is not an adaptive PLD-selection or PLD-reduction result.

A true adaptive acquisition environment would need at least:

1. Partial observations, so unacquired measurements are unavailable.
2. A mask of available or already selected PLDs.
3. A discrete acquisition action identifying the next PLD.
4. A stop action.
5. An acquisition-time or measurement-count cost.
6. A reward that trades parameter accuracy against acquisition burden.

Those changes would define a different RL problem and are not implemented in the current notebook.

## 17. Limitations

1. The data are synthetic Buxton signals with synthetic Rician-like noise, not scanner data.
2. The physiological ranges and model constants are simulation assumptions.
3. The five timing pairs are fixed and fully observed before action selection.
4. The current action changes parameter estimates, not measurement acquisition.
5. CCC is batch-level, so the reward includes population statistics rather than only per-sample information.
6. Results are stochastic despite fixed seeds because GPU execution and evaluation sampling can introduce small variation.
7. One trained seed and one independent test seed do not establish general superiority. Multiple independent training seeds and confidence intervals are still needed.
8. The SAC result does not outperform the current grid-only ablation on the displayed test values. No superiority claim is made.
9. The notebook contains historical LM-related code concepts in its documentation, but the final executed environment raises if an LM handoff is reached. The final configuration avoids that path by setting the handoff to the 12-step horizon.

## 18. Reproducibility and Traceability

To reproduce the current result:

1. Open `rl-asl (6).ipynb` with a PyTorch/Jupyter kernel.
2. Run the cells in order from imports through the environment, replay buffer, SAC definitions, training loop, evaluation functions, and configuration.
3. Run the training cell. It sets `SEED = 20260918` and calls `agent = train(args)`.
4. Wait for iteration `1000/1000` and confirm actor loss, critic loss, alpha, reward, and checkpoint messages.
5. Run the SNR sweep and detailed evaluation cells.
6. Run the final report cell. It sets `TEST_SEED = 20260919`, evaluates the trained `SACAgent`, and prints the separate grid baseline.

The final RL artifacts are:

- [Notebook](rl-asl%20(6).ipynb)
- [Trained checkpoint](experiments/rl_sac_trained/notebook_results/sac_asl_final.pth)
- [Training CSV](experiments/rl_sac_trained/notebook_results/train_log.csv)
- [SAC SNR plot](experiments/rl_sac_trained/notebook_results/sac_nrmse_vs_snr.png)

## 19. Questions to Defend in a Presentation

### Problem understanding

**What is CBF?** Cerebral blood flow, the flow parameter estimated in mL/100 g/min.

**What is ATT?** Arterial transit time, represented internally in milliseconds and reported in seconds in the metrics.

**What is PLD?** Post-labeling delay, the time between labeling and readout.

**Why multiple PLDs?** Different delays expose different arrival and relaxation behavior, providing more information for separating CBF and ATT.

**Why is ASL difficult at low SNR?** The signal is small, the magnitude noise is non-Gaussian, and different CBF/ATT combinations can produce similar observations.

### Physics

**Why Buxton?** It is the explicit kinetic model used by this simulation to connect flow, arrival time, relaxation, and timing to ASL signal.

**What do the main terms do?** Labeling and suppression scale signal; the blood partition coefficient normalizes blood-to-tissue contribution; blood and tissue T1 terms model relaxation; PLD and LD determine arrival and bolus-duration effects.

**Why Rician-like noise?** MRI magnitude is formed from noisy real and imaginary channels, so the magnitude operation creates Rician-like rather than signed additive Gaussian data.

**Why these ranges?** They are the project simulation support for random ground truth and estimates. The notebook does not prove universal optimality.

### RL

**Why RL instead of a one-shot DNN?** This project formulates estimation as sequential continuous correction. RL is used to learn an update policy rather than direct regression.

**What is the state?** Five observed signals, five predicted signals, normalized current CBF and ATT, and normalized time progress.

**What is the action?** A bounded continuous CBF increment and ATT increment. It is not a PLD action.

**What is the reward?** Error improvement multiplied by 10, plus twice batch-level CCC, minus a 0.02 step cost.

**Why SAC?** The action is continuous and bounded, and SAC supplies stochastic exploration, twin critics, replay, target critics, and learned entropy control.

**Why not TD3?** TD3 would also be a plausible continuous-control choice. The current project implemented SAC; it does not contain a controlled TD3 comparison, so no claim that SAC is superior to TD3 is justified.

**What does the actor learn?** Which parameter increment to apply from the current state.

**What does the critic learn?** The expected future discounted reward for a state-action pair.

**Why two critics?** Their minimum reduces Q-value overestimation.

**Why entropy?** To prevent premature collapse to a single action and maintain exploration during training.

**Why replay?** To reuse transitions and reduce temporal correlation in gradient updates.

### Optimization

**Why LM?** LM is useful for local physics-based refinement after a good initialization, but it is not RL. In the current final RL pipeline it is disabled.

**Why not LM alone?** Random-start LM is local and can converge to a poor basin. The grid baseline is a separate way to obtain an observable physics-based initial point, not a learned policy.

**Why does initialization matter?** A local optimizer only sees nearby directions; different starting points can lead to different local solutions.

**Why not simply use grid search?** Grid search can be a strong non-RL baseline, but it does not learn a sequential policy and does not address adaptive acquisition.

### Evaluation

**What is RMSE?** The square root of the mean squared parameter error.

**What is MAE?** The mean absolute parameter error.

**What is CCC?** Lin's concordance correlation coefficient, measuring agreement with attention to correlation, bias, and scale.

**Why evaluate across SNR?** It tests how degradation in measurement quality affects the estimator.

**Why separate training and evaluation?** The final test uses a fresh population and a different seed so the reported metrics are not the same samples used for training.

**Why deterministic actions during evaluation?** The policy is stochastic during training for exploration; deterministic mean actions make evaluation repeatable and measure the learned policy without sampling exploration noise.

### Validity

**Is this actually RL?** Yes, in the final executed path: `agent = train(args)` collects transitions, fills the replay buffer, runs actor/critic/alpha updates, saves a SAC checkpoint, and evaluates the resulting `SACAgent`.

**Was the model actually trained?** Yes. The notebook output contains 1,000 iteration logs, numeric losses after warm-up, changing alpha, changing rewards, and a saved checkpoint.

**What proves learning occurred?** The strongest available evidence is execution of `agent.update` after the replay threshold, non-NaN actor and critic losses, changing alpha, increasing reward from negative warm-up values to approximately 29, and the saved actor/critic state dictionaries. A direct parameter-difference audit was not printed by the notebook.

**How is the baseline different?** The grid baseline uses a deterministic forward-model grid and does not train an actor, critic, or replay buffer. It is reported separately.

**Does the implementation reduce PLDs?** No. All five signals are available before the first action.

**What would make it true adaptive PLD selection?** Partial observations, a PLD-selection action, a stop action, acquisition cost, and a reward balancing accuracy against measurements.

## 20. Final Audit Statement

The defensible conclusion from the current notebook is:

> A genuine SAC agent was trained for iterative CBF/ATT estimation on synthetic five-measurement ASL data. The actor learned continuous parameter-update actions through replay-buffer SAC optimization. The trained policy was evaluated deterministically on a fresh synthetic test population and achieved the metrics reported in Section 11. A separate forward-model grid initializer was retained as a non-RL ablation. The implementation does not perform adaptive PLD reduction, and the SAC policy is not claimed to outperform the non-RL baseline.

Any older result not generated by the current notebook should be described as historical or unverified from the current notebook, not as a current final result.
