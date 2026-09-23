#!/usr/bin/env python
# coding: utf-8

# # SAC Reinforcement Learning for ASL (CBF, ATT) Parameter Estimation
# 
# This is the RL counterpart to the simulation-based supervised DNN in **Ishida et al., JMRI 2023**
# (the base paper). Where the DNN learns a single forward regression `ΔM_obs -> (CBF, ATT)` from
# millions of pre-simulated pairs, this notebook reframes the same inverse problem as a **sequential
# decision process**: an agent starts from a guess, looks at how far its guess's forward-simulated
# signal is from the observed signal, and takes iterative continuous steps in (CBF, ATT) space until
# the residual is small (or a step budget runs out).
# 
# **Design choices** (from the physics in the base paper + the RL literature bridge worked out earlier):
# - **SAC** (Soft Actor-Critic) — entropy-regularized, off-policy, well suited to continuous, noisy,
#   non-convex inverse problems like this one.
# - **12-D state**: 5 observed + 5 currently-predicted multi-delay signals, plus the current
#   normalized (CBF, ATT) estimate.
# - **2-D continuous action**: bounded per-step deltas on (CBF, ATT) — this keeps it a genuine
#   multi-step sequential decision process rather than one-shot regression wearing an RL costume.
# - **Dense reward** = `-NRMSE` of the current signal residual each step — the *same* NRMSE formula
#   the base paper uses for evaluation, so training reward and reported accuracy are literally the
#   same quantity.
# - A small **`clean_fraction`** of training episodes get near-zero noise, to avoid the same
#   distribution-coverage pitfall this project's earlier DNN experiments ran into (a model trained
#   only on a fixed noisy-SNR grid can behave badly at the near-clean end of the real population).
# 
# **How to run on Kaggle:**
# 1. Enable a GPU accelerator (Settings → Accelerator → GPU).
# 2. Run all cells top to bottom.
# 3. Adjust the `Config` cell below (iterations, batch size, etc.) to fit your session's time budget —
#    defaults are tuned to be correct and stable, not necessarily fast; this is meant to be iterated on,
#    the same way the DNN side went through staged experiments (A, B, F, G, H, ...).
# 4. Checkpoints, a training log CSV, and an SNR-sweep comparison plot are written to `/kaggle/working/`.
# 

# ## 1. Imports

# In[1]:


import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())


# In[ ]:





# ## 2. Physics / Environment
# 
# The Buxton single-compartment kinetic model (base paper Eq. 1), the same 5-bolus PLD/LD table,
# constants, and signal scale factor as the DNN pipeline. `ASLEnvironment` wraps this into a
# vectorized, batched, multi-step RL environment.

# In[2]:


# Multi-delay bolus timings from the base paper (Ishida et al. 2023),
# the "combined approach": (PLD [sec], LD [sec]) for boluses 1-5.
PLD_SEC = torch.tensor([0.700, 2.033, 3.366, 0.700])
LD_SEC = torch.tensor([1.333, 1.333, 1.333, 4.000])
N_BOLUS = 4

# MRI / kinetic-model constants (3.0T), matching the base paper.
ALPHA = 0.85              # labeling efficiency
BETA = 0.75               # background suppression efficiency
LAMBDA_BLOOD = 0.9        # blood partition coefficient
T1_TISSUE = 1.2           # sec
T1_BLOOD = 1.66           # sec
SIGNAL_SCALE = 100_000.0  # ΔM/M0 (%) -> scaled signal intensity, as in the paper

# Physiological ground-truth ranges used by the base paper for training.
CBF_MIN, CBF_MAX = 20.0, 90.0        # mL/100 g/min
ATT_MIN, ATT_MAX = 500.0, 3000.0     # msec

# Reference condition used by the paper to anchor rSNR -> noise SD:
# PLD=2000ms, LD=1800ms, CBF=50 mL/100g/min, ATT=1600ms.
REF_PLD_SEC = 2.000
REF_LD_SEC = 1.800
REF_CBF = 50.0
REF_ATT_MSEC = 1600.0
RICIAN_NOISE_FACTOR = 2.0


def buxton_signal(cbf: torch.Tensor, att_msec: torch.Tensor,
                   pld_sec: torch.Tensor, ld_sec: torch.Tensor) -> torch.Tensor:
    """
    Vectorized single-compartment (Buxton) kinetic model, Eq. (1) of the base paper.

    cbf:      [..., 1] or broadcastable, mL/100 g/min
    att_msec: [..., 1] or broadcastable, msec
    pld_sec, ld_sec: [n_bolus] delay-table tensors (broadcast against cbf/att)

    Returns scaled ΔM with shape [..., n_bolus].
    """
    f = cbf / 6000.0                 # mL/100g/min -> mL/g/s
    att = att_msec / 1000.0          # msec -> sec

    w = pld_sec
    tau = ld_sec

    t1_decay = torch.exp(-att / T1_BLOOD)
    term1 = torch.exp(-torch.clamp(w - att, min=0.0) / T1_TISSUE)
    term2 = torch.exp(-torch.clamp(tau + w - att, min=0.0) / T1_TISSUE)
    bracket = term1 - term2

    coeff = 2 * ALPHA * BETA * T1_BLOOD / LAMBDA_BLOOD
    signal = coeff * f * t1_decay * bracket
    return signal * SIGNAL_SCALE


def _reference_signal_scale(device) -> float:
    """Single-delay reference signal used to anchor rSNR -> noise SD."""
    cbf = torch.tensor([REF_CBF], device=device)
    att = torch.tensor([REF_ATT_MSEC], device=device)
    pld = torch.tensor([REF_PLD_SEC], device=device)
    ld = torch.tensor([REF_LD_SEC], device=device)
    sig = buxton_signal(cbf, att, pld, ld)
    return float(sig.abs().item())


# In[5]:


class ASLEnvironment:
    """
    Vectorized multi-step environment implementing a Differentiable
    Physics-Informed Actor-Critic (DPAC) design:

    * Phase 1 -- Macro-exploration (steps 0 .. lm_handoff_step-1): the SAC
      actor applies continuous deltas to the running (CBF, ATT) estimate,
      same as before.
    * Phase 2 -- Micro-exploitation (steps lm_handoff_step .. max_steps-1):
      the SAC action is ignored and a fixed-iteration, native-PyTorch
      Levenberg-Marquardt (LM) solve takes over, using the actor's current
      estimate as its prior and refining it against the observed signal.

    State (13-D): [obs_signal(5), pred_signal(5), norm_cbf(1), norm_att(1),
        time_fraction(1)]. Signals are normalized by the reference-condition
        signal so the network sees O(1) inputs regardless of SIGNAL_SCALE.
        time_fraction = step_count / max_steps also tells the actor when the
        hand-off to the LM solver is about to happen.
    Action (2-D, in [-1, 1]): per-step deltas on CBF/ATT during Phase 1 only,
        scaled by STEP_CBF / STEP_ATT (with mild decay) and clipped to the
        physiological range. Ignored during Phase 2.
    Reward: potential-based shaping on the true normalized parameter
        distance, R_t = (Error_prev - Error_curr) * 10 + CCC_WEIGHT * CCC,
        where CCC is the batch-level Concordance Correlation Coefficient
        between ground truth and current estimate (averaged over CBF/ATT).
        This rewards the actor both for closing the Euclidean gap and for
        handing the LM solver a prior from which it recovers high
        structural (variance/bias) agreement, plus a small per-step cost and
        a terminal accuracy bonus.
    Episode end: `done` (terminal, used for Bellman bootstrapping) is only
        set on true absorption (param error below a tight threshold).
        Hitting `max_steps` is reported separately as `info["timeout"]` and
        must NOT zero out the critic's bootstrap target.

    Note on gradients: this is an off-policy SAC setup -- actions are
    sampled under `torch.no_grad()` (see `SACAgent.act`) and every tensor
    written into the replay buffer is `.detach()`-ed before storage. So the
    LM solver's differentiability is used *within* each `step()` call (to
    get a correct, well-conditioned Jacobian via autograd rather than a
    hand-derived one), not to backpropagate a loss through the whole
    trajectory into the actor's weights -- there is no such path in
    standard off-policy SAC + replay buffer. The actor still learns to hand
    off a good prior purely through the normal RL credit-assignment
    mechanism (the TD-target/Q-function), the same way it learns everything
    else.
    """

    STEP_CBF = 10.0    # max per-step change, mL/100 g/min (Phase 1 only)
    STEP_ATT = 300.0   # max per-step change, msec (Phase 1 only)

    # DPAC / Levenberg-Marquardt hyperparameters
    LM_ITERS = 3            # fixed unrolled LM iterations per Phase-2 step
    LM_LAMBDA = 1e-2        # damping factor
    LM_HANDOFF_FRAC = 0.8   # fraction of the episode spent in Phase 1
    CCC_WEIGHT = 2.0        # weight on the CCC term in the composite reward

    def __init__(self, batch_size: int, max_steps: int = 10,
                 snr_min: float = 2.0, snr_max: float = 20.0,
                 clean_fraction: float = 0.05,
                 init_mode: str = "random",
                 coarse_grid_size: int = 21,
                 lm_handoff_frac: float = None,
                 device: str = "cpu"):
        self.B = batch_size
        self.max_steps = max_steps
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.clean_fraction = clean_fraction
        self.init_mode = init_mode
        self.coarse_grid_size = coarse_grid_size
        self.device = device

        self.pld = PLD_SEC.to(device)
        self.ld = LD_SEC.to(device)
        self.ref_signal = _reference_signal_scale(device)

        # 5 obs + 5 pred + 2 norm_params + 1 time_progress = 13 dimensions
        self.state_dim = 2 * N_BOLUS + 3
        self.action_dim = 2

        # All batch elements step in lockstep (a single env.reset() /
        # env.step() call advances every voxel by one step together), so the
        # hand-off point is a plain step-index threshold.
        handoff_frac = self.LM_HANDOFF_FRAC if lm_handoff_frac is None else lm_handoff_frac
        # Zero is valid for the LM-only ablation; the notebook default remains 0.8.
        self.lm_handoff_step = max(0, int(round(handoff_frac * self.max_steps)))

        self.cbf_true = None
        self.att_true = None
        self.obs_signal = None
        self.cbf_est = None
        self.att_est = None
        self.step_count = None
        self.prev_param_err = None

    def _norm(self, x, lo, hi):
        return (x - lo) / (hi - lo) * 2.0 - 1.0

    def _denorm(self, x, lo, hi):
        return (x + 1.0) / 2.0 * (hi - lo) + lo

    def _predicted_signal(self):
        return buxton_signal(self.cbf_est.unsqueeze(-1),
                              self.att_est.unsqueeze(-1),
                              self.pld, self.ld)

    def _compute_param_error(self):
        """Normalized Euclidean distance between (cbf_est, att_est) and the
        true (cbf_true, att_true), each min-max normalized to [-1, 1]. This
        is the quantity the reward is shaped on -- it is what the paper's
        NRMSE metrics ultimately measure, unlike the signal-domain residual
        the original reward used."""
        norm_cbf_true = self._norm(self.cbf_true, CBF_MIN, CBF_MAX)
        norm_cbf_est = self._norm(self.cbf_est, CBF_MIN, CBF_MAX)
        norm_att_true = self._norm(self.att_true, ATT_MIN, ATT_MAX)
        norm_att_est = self._norm(self.att_est, ATT_MIN, ATT_MAX)
        return torch.sqrt((norm_cbf_true - norm_cbf_est) ** 2 +
                           (norm_att_true - norm_att_est) ** 2)

    @staticmethod
    def _concordance_corr(true, pred, eps: float = 1e-8):
        """Lin's Concordance Correlation Coefficient between two 1-D
        tensors. This is a *batch-level* statistic (needs the variance and
        covariance across voxels), unlike `_compute_param_error`, which is
        per-voxel."""
        true_mean, pred_mean = true.mean(), pred.mean()
        true_var, pred_var = true.var(unbiased=False), pred.var(unbiased=False)
        covariance = ((true - true_mean) * (pred - pred_mean)).mean()
        return (2 * covariance) / (true_var + pred_var + (true_mean - pred_mean) ** 2 + eps)

    def _compute_ccc(self):
        """Average CCC across CBF and ATT, computed in normalized space so
        the two parameters (which differ ~30x in raw scale) contribute on
        comparable footing. Returns a 0-D scalar -- the same bonus is added
        to every voxel's reward this step, since CCC only exists as a
        population statistic."""
        norm_cbf_true = self._norm(self.cbf_true, CBF_MIN, CBF_MAX)
        norm_cbf_est = self._norm(self.cbf_est, CBF_MIN, CBF_MAX)
        norm_att_true = self._norm(self.att_true, ATT_MIN, ATT_MAX)
        norm_att_est = self._norm(self.att_est, ATT_MIN, ATT_MAX)
        ccc_cbf = self._concordance_corr(norm_cbf_true, norm_cbf_est)
        ccc_att = self._concordance_corr(norm_att_true, norm_att_est)
        return (ccc_cbf + ccc_att) / 2.0

    def _lm_signal_and_jacobian(self, norm_cbf, norm_att):
        """Batched forward signal + Jacobian of the Buxton forward model
        w.r.t. the *normalized* (CBF, ATT) parameters.

        Uses `torch.func.vmap(torch.func.jacrev(...))` rather than
        `torch.autograd.functional.jacobian`: the latter treats a whole
        batch as one flat vector-to-vector map, so for a batch of B voxels
        it would materialize a (B*N_BOLUS, B*2) matrix that is almost
        entirely zeros (off-diagonal blocks are zero since voxels are
        independent). `vmap(jacrev(...))` instead computes the true
        per-voxel [B, N_BOLUS, 2] Jacobian directly and efficiently.

        Operating in normalized [-1, 1] space (instead of raw mL/100g/min /
        msec units) is what keeps J^T J well-conditioned for
        `torch.linalg.solve` -- CBF and ATT differ by ~30x in raw scale, so
        literal-units normal equations are poorly conditioned and the LM
        step can blow up.
        """
        def signal_fn(ncbf, natt):
            cbf = self._denorm(ncbf, CBF_MIN, CBF_MAX)
            att = self._denorm(natt, ATT_MIN, ATT_MAX)
            return buxton_signal(cbf.reshape(1, 1), att.reshape(1, 1),
                                  self.pld, self.ld).reshape(-1)

        jac_fn = torch.func.jacrev(signal_fn, argnums=(0, 1))
        d_cbf, d_att = torch.func.vmap(jac_fn)(norm_cbf, norm_att)   # each [B, N_BOLUS]
        J = torch.stack([d_cbf, d_att], dim=-1)                       # [B, N_BOLUS, 2]
        pred = torch.func.vmap(signal_fn)(norm_cbf, norm_att)         # [B, N_BOLUS]
        return pred, J

    def _levenberg_marquardt_step(self, n_iters=None, lam=None):
        """Fixed `n_iters`-iteration LM refinement of the running (CBF, ATT)
        estimate against `obs_signal`, run entirely in normalized parameter
        space. The estimate handed off by the SAC actor is used as the LM
        prior; each iteration solves the damped normal equations

            (J^T J + lambda * diag(J^T J)) * delta = J^T r

        for the update `delta`, applies it, and re-clamps to the
        physiological range before the next iteration.
        """
        n_iters = self.LM_ITERS if n_iters is None else n_iters
        lam = self.LM_LAMBDA if lam is None else lam

        norm_cbf = self._norm(self.cbf_est, CBF_MIN, CBF_MAX)
        norm_att = self._norm(self.att_est, ATT_MIN, ATT_MAX)
        eye2 = torch.eye(2, device=self.device).unsqueeze(0)

        for _ in range(n_iters):
            pred, J = self._lm_signal_and_jacobian(norm_cbf, norm_att)
            r = self.obs_signal - pred                                    # [B, N_BOLUS]

            JT_J = torch.einsum('bni,bnj->bij', J, J)                     # [B, 2, 2]
            JT_r = torch.einsum('bni,bn->bi', J, r)                       # [B, 2]

            diag = torch.diagonal(JT_J, dim1=-2, dim2=-1)                 # [B, 2]
            damping = lam * torch.diag_embed(diag) + 1e-6 * eye2          # stabilize inversion
            A = JT_J + damping

            delta = torch.linalg.solve(A, JT_r.unsqueeze(-1)).squeeze(-1)  # [B, 2]
            norm_cbf = (norm_cbf + delta[:, 0]).clamp(-1.0, 1.0)
            norm_att = (norm_att + delta[:, 1]).clamp(-1.0, 1.0)

        self.cbf_est = self._denorm(norm_cbf, CBF_MIN, CBF_MAX)
        self.att_est = self._denorm(norm_att, ATT_MIN, ATT_MAX)

    def _make_state(self, pred):
        norm_cbf = self._norm(self.cbf_est, CBF_MIN, CBF_MAX)
        norm_att = self._norm(self.att_est, ATT_MIN, ATT_MAX)
        obs_n = self.obs_signal / self.ref_signal
        pred_n = pred / self.ref_signal
        time_fraction = (self.step_count / float(self.max_steps)).unsqueeze(-1)
        return torch.cat([obs_n, pred_n,
                           norm_cbf.unsqueeze(-1), norm_att.unsqueeze(-1),
                           time_fraction], dim=-1)

    def _coarse_grid_init(self, grid_size=21):
        """Observable-only coarse inverse-model initializer.

        It evaluates a fixed CBF/ATT grid against the already observed five
        signals and selects the least-squares candidate independently for
        each sample.  It never accesses cbf_true or att_true.
        """
        cbf_grid = torch.linspace(CBF_MIN, CBF_MAX, grid_size, device=self.device)
        att_grid = torch.linspace(ATT_MIN, ATT_MAX, grid_size, device=self.device)
        cbf_mesh, att_mesh = torch.meshgrid(cbf_grid, att_grid, indexing="ij")
        candidates = buxton_signal(cbf_mesh.reshape(-1, 1), att_mesh.reshape(-1, 1),
                                   self.pld, self.ld)  # [grid_size^2, 5]
        residual = (self.obs_signal[:, None, :] - candidates[None, :, :]).square().mean(dim=-1)
        best = residual.argmin(dim=1)
        self.cbf_est = cbf_mesh.reshape(-1)[best]
        self.att_est = att_mesh.reshape(-1)[best]

    def reset(self):
        B, dev = self.B, self.device
        self.cbf_true = torch.empty(B, device=dev).uniform_(CBF_MIN, CBF_MAX)
        self.att_true = torch.empty(B, device=dev).uniform_(ATT_MIN, ATT_MAX)

        clean_signal = buxton_signal(self.cbf_true.unsqueeze(-1),
                                      self.att_true.unsqueeze(-1),
                                      self.pld, self.ld)

        rsnr = torch.empty(B, device=dev).uniform_(self.snr_min, self.snr_max)
        is_clean = torch.rand(B, device=dev) < self.clean_fraction
        # rSNR is defined against the total complex-noise scale. Since the
        # magnitude observation is generated from two independent channels,
        # use the calibrated per-channel SD expected by the benchmark.
        sigma = (self.ref_signal / (RICIAN_NOISE_FACTOR * rsnr)).unsqueeze(-1).expand(-1, N_BOLUS).clone()
        sigma[is_clean] = 0.0

        # Rician-style magnitude noise: sqrt((clean + n_r)^2 + n_i^2)
        n_r = torch.randn_like(clean_signal) * sigma
        n_i = torch.randn_like(clean_signal) * sigma
        self.obs_signal = torch.sqrt((clean_signal + n_r) ** 2 + n_i ** 2)

        # Initializers may use only observed signal information, never truth.
        if self.init_mode == "random":
            self.cbf_est = torch.empty(B, device=dev).uniform_(CBF_MIN, CBF_MAX)
            self.att_est = torch.empty(B, device=dev).uniform_(ATT_MIN, ATT_MAX)
        elif self.init_mode == "center":
            self.cbf_est = torch.full((B,), (CBF_MIN + CBF_MAX) / 2.0, device=dev)
            self.att_est = torch.full((B,), (ATT_MIN + ATT_MAX) / 2.0, device=dev)
        elif self.init_mode == "coarse_grid":
            self._coarse_grid_init(self.coarse_grid_size)
        else:
            raise ValueError(f"Unknown init_mode: {self.init_mode}")
        self.step_count = torch.zeros(B, device=dev)
        self.prev_param_err = self._compute_param_error()

        pred = self._predicted_signal()
        return self._make_state(pred)

    def step(self, action: torch.Tensor):
        action = action.clamp(-1.0, 1.0)

        # DPAC hand-off. `self.step_count` is identical across the batch
        # (the whole env advances one step at a time), so this is a single
        # scalar check rather than a per-element mask.
        current_step = int(self.step_count[0].item())
        in_lm_phase = current_step >= self.lm_handoff_step

        if not in_lm_phase:
            # Phase 1: SAC macro-exploration.
            decay = 1.0 - 0.5 * (self.step_count / float(self.max_steps))
            d_cbf = action[:, 0] * self.STEP_CBF * decay
            d_att = action[:, 1] * self.STEP_ATT * decay
            self.cbf_est = (self.cbf_est + d_cbf).clamp(CBF_MIN, CBF_MAX)
            self.att_est = (self.att_est + d_att).clamp(ATT_MIN, ATT_MAX)
        else:
            # Phase 2: micro-exploitation. The SAC action is ignored; the
            # differentiable LM solver refines the actor's hand-off prior.
            self._levenberg_marquardt_step()

        self.step_count += 1

        pred = self._predicted_signal()
        cur_param_err = self._compute_param_error()
        cur_ccc = self._compute_ccc()

        # Potential-based reward: reduction in true parameter distance,
        # plus a batch-level CCC structural-agreement bonus.
        step_improvement = (self.prev_param_err - cur_param_err) * 10.0
        self.prev_param_err = cur_param_err

        reward = step_improvement + self.CCC_WEIGHT * cur_ccc - 0.02

        # True termination vs time-limit truncation
        timeout = (self.step_count >= self.max_steps)
        absorbed = (cur_param_err < 0.01)
        done = absorbed  # DO NOT mark done=True on timeouts in Q-target updates

        if timeout.any():
            # Add terminal accuracy bonus
            reward = reward - cur_param_err * 2.0

        next_state = self._make_state(pred)
        info = {
            "param_err": cur_param_err.detach(),
            "ccc": cur_ccc.detach(),
            "timeout": timeout.detach(),
            "in_lm_phase": in_lm_phase,
        }
        return next_state, reward.detach(), done.detach(), info


# ## 3. Replay Buffer

# In[6]:


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.state = torch.zeros((capacity, state_dim), device=device)
        self.action = torch.zeros((capacity, action_dim), device=device)
        self.reward = torch.zeros((capacity, 1), device=device)
        self.next_state = torch.zeros((capacity, state_dim), device=device)
        self.done = torch.zeros((capacity, 1), device=device)

    def add_batch(self, s, a, r, s2, d):
        n = s.shape[0]
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.capacity
        self.state[idx] = s
        self.action[idx] = a
        self.reward[idx] = r.view(-1, 1)
        self.next_state[idx] = s2
        self.done[idx] = d.float().view(-1, 1)  # only True if absorbed, False on timeout
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (self.state[idx], self.action[idx], self.reward[idx],
                self.next_state[idx], self.done[idx])


# ## 4. SAC Networks + Agent
# 
# Gaussian tanh-squashed actor, twin critics, target networks, and **automatic entropy-temperature
# tuning** (removes one whole axis of manual hyperparameter search compared to a fixed entropy
# coefficient).

# In[7]:


LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state, deterministic=False, eps=1e-6):
        mean, log_std = self.forward(state)
        if deterministic:
            return torch.tanh(mean), None
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + eps)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        def head():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        self.q1 = head()
        self.q2 = head()

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


@dataclass
class SACConfig:
    state_dim: int
    action_dim: int
    hidden: int = 256
    gamma: float = 0.98
    tau: float = 0.005
    lr: float = 3e-4
    target_entropy: float = field(default=None)  # defaults to -action_dim
    device: str = "cpu"


class SACAgent:
    def __init__(self, cfg: SACConfig):
        self.cfg = cfg
        dev = cfg.device
        self.actor = GaussianPolicy(cfg.state_dim, cfg.action_dim, cfg.hidden).to(dev)
        self.critic = TwinCritic(cfg.state_dim, cfg.action_dim, cfg.hidden).to(dev)
        self.critic_target = TwinCritic(cfg.state_dim, cfg.action_dim, cfg.hidden).to(dev)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        target_entropy = cfg.target_entropy or -float(cfg.action_dim)
        self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=dev)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

    @property
    def alpha(self):
        # Bound alpha to avoid complete entropy collapse during noisy updates
        return torch.clamp(self.log_alpha.exp(), min=0.05, max=1.0)

    def act(self, state, deterministic=False):
        with torch.no_grad():
            action, _ = self.actor.sample(state, deterministic=deterministic)
        return action

    def update(self, batch):
        s, a, r, s2, d = batch
        gamma, tau = self.cfg.gamma, self.cfg.tau

        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
            q1_t, q2_t = self.critic_target(s2, a2)
            q_t = torch.min(q1_t, q2_t) - self.alpha * logp2
            target = r + (1 - d) * gamma * q_t
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        a_new, logp = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, a_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * logp - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.mul_(1 - tau).add_(tau * p)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
        }

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha,
        }, path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"])


# ## 5. Training Loop

# In[8]:


def train(args):
    device = args.device
    env = ASLEnvironment(batch_size=args.batch_envs, max_steps=args.max_steps,
                          snr_min=args.snr_min, snr_max=args.snr_max,
                          clean_fraction=args.clean_fraction,
                          init_mode=args.init_mode,
                          coarse_grid_size=args.coarse_grid_size,
                          lm_handoff_frac=args.lm_handoff_frac,
                          device=device)

    cfg = SACConfig(state_dim=env.state_dim, action_dim=env.action_dim,
                     hidden=args.hidden, gamma=args.gamma, tau=args.tau,
                     lr=args.lr, device=device)
    agent = SACAgent(cfg)
    buffer = ReplayBuffer(env.state_dim, env.action_dim, args.buffer_capacity, device)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("iteration,mean_reward,mean_final_param_err,critic_loss,actor_loss,alpha\n")

    warmup_iters = max(1, args.warmup_iters)
    t0 = time.time()

    for it in range(1, args.iterations + 1):
        state = env.reset()
        ep_reward = torch.zeros(env.B, device=device)
        final_param_err = None

        for t in range(args.max_steps):
            if it <= warmup_iters:
                action = torch.empty(env.B, env.action_dim, device=device).uniform_(-1, 1)
            else:
                action = agent.act(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            buffer.add_batch(state, action, reward, next_state, done)
            ep_reward += reward
            final_param_err = info["param_err"]
            state = next_state

        stats = {"critic_loss": float("nan"), "actor_loss": float("nan"),
                  "alpha": agent.alpha.item()}
        if it > warmup_iters and buffer.size >= args.min_buffer:
            for _ in range(args.updates_per_iter):
                batch = buffer.sample(args.train_batch)
                stats = agent.update(batch)

        if it % args.log_every == 0 or it == args.iterations:
            mean_r = ep_reward.mean().item()
            mean_param_err = final_param_err.mean().item()
            elapsed = time.time() - t0
            print(f"[it {it:5d}/{args.iterations}] "
                  f"mean_ep_reward={mean_r:8.3f}  final_param_err={mean_param_err:6.4f}  "
                  f"critic_loss={stats['critic_loss']:.4f}  "
                  f"actor_loss={stats['actor_loss']:.4f}  "
                  f"alpha={stats['alpha']:.4f}  ({elapsed:6.1f}s)")
            with open(log_path, "a") as f:
                f.write(f"{it},{mean_r},{mean_param_err},{stats['critic_loss']},"
                        f"{stats['actor_loss']},{stats['alpha']}\n")

        if it % args.eval_every == 0 or it == args.iterations:
            metrics = evaluate(agent, args, device, n_pairs=args.eval_pairs)
            print(f"    eval: CBF NMAE={metrics['cbf_nmae']:.4f} "
                  f"NRMSE={metrics['cbf_nrmse']:.4f} | "
                  f"ATT NMAE={metrics['att_nmae']:.4f} "
                  f"NRMSE={metrics['att_nrmse']:.4f}")

        if it % args.ckpt_every == 0 or it == args.iterations:
            agent.save(os.path.join(args.out_dir, f"sac_asl_it{it}.pth"))

    agent.save(os.path.join(args.out_dir, "sac_asl_final.pth"))
    return agent


# ## 6. Evaluation
# 
# Reports NMAE / NRMSE for CBF and ATT the same way the base paper's Eqs. (3)-(4) / Fig. 3-4 do,
# so results are directly comparable to the DNN numbers already produced for this project.

# In[9]:


def evaluate(agent, args, device, n_pairs=2000):
    env = ASLEnvironment(batch_size=n_pairs, max_steps=args.max_steps,
                          snr_min=args.snr_min, snr_max=args.snr_max,
                          clean_fraction=0.0, init_mode=args.init_mode,
                          coarse_grid_size=args.coarse_grid_size,
                          lm_handoff_frac=args.lm_handoff_frac, device=device)
    state = env.reset()
    with torch.no_grad():
        for _ in range(args.max_steps):
            action = agent.act(state, deterministic=True)
            state, _, _, _ = env.step(action)

    cbf_true, att_true = env.cbf_true, env.att_true
    cbf_pred, att_pred = env.cbf_est, env.att_est

    def nmae(true, pred):
        return (torch.mean(torch.abs(true - pred)) / torch.mean(true)).item()

    def nrmse(true, pred):
        return (torch.sqrt(torch.mean((true - pred) ** 2)) / torch.mean(true)).item()

    return {
        "cbf_nmae": nmae(cbf_true, cbf_pred),
        "cbf_nrmse": nrmse(cbf_true, cbf_pred),
        "att_nmae": nmae(att_true, att_pred),
        "att_nrmse": nrmse(att_true, att_pred),
    }


def evaluate_vs_snr(agent, args, device, snr_grid=(2, 5, 8, 12, 16, 20),
                     n_pairs_per_level=1000):
    """Paper-Figure-3/4-style sweep: metrics as a function of rSNR."""
    rows = []
    for snr in snr_grid:
        env = ASLEnvironment(batch_size=n_pairs_per_level, max_steps=args.max_steps,
                              snr_min=snr, snr_max=snr, clean_fraction=0.0,
                              init_mode=args.init_mode,
                              coarse_grid_size=args.coarse_grid_size,
                              lm_handoff_frac=args.lm_handoff_frac, device=device)
        state = env.reset()
        with torch.no_grad():
            for _ in range(args.max_steps):
                action = agent.act(state, deterministic=True)
                state, _, _, _ = env.step(action)
        cbf_nrmse = (torch.sqrt(torch.mean((env.cbf_true - env.cbf_est) ** 2))
                     / torch.mean(env.cbf_true)).item()
        att_nrmse = (torch.sqrt(torch.mean((env.att_true - env.att_est) ** 2))
                     / torch.mean(env.att_true)).item()
        rows.append((snr, cbf_nrmse, att_nrmse))
    return rows


# ## 7. Config
# 
# Defaults mirror what a first serious run would use. **For a quick sanity check first**
# (recommended before committing a full Kaggle session), drop `iterations` to ~20 and
# `batch_envs` to ~32 and confirm `final_nrmse` is trending down before scaling back up.

# In[10]:


DEFAULT_CONFIG = {
    "iterations": 1000, "batch_envs": 256, "max_steps": 12,
    "snr_min": 2.0, "snr_max": 20.0, "clean_fraction": 0.10,
    "init_mode": "random", "coarse_grid_size": 21, "lm_handoff_frac": 0.8,
    "hidden": 256, "gamma": 0.95, "tau": 0.005, "lr": 3e-4,
    "buffer_capacity": 1_000_000, "train_batch": 256,
    "updates_per_iter": 64, "warmup_iters": 25, "min_buffer": 5000,
    "log_every": 10, "eval_every": 25, "eval_pairs": 2000,
    "ckpt_every": 100, "out_dir": "sac_asl_out", "seed": 1234,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ccc(true, pred, eps=1e-8):
    return (2 * ((true - true.mean()) * (pred - pred.mean())).mean() /
            (true.var(unbiased=False) + pred.var(unbiased=False) +
             (true.mean() - pred.mean()).square() + eps)).item()


def benchmark(agent, args, snrs=(10, 15, 20), n_pairs=2000):
    """Fixed-protocol detailed metrics, serialized by the experiment runner."""
    results = {}
    for snr in snrs:
        env = ASLEnvironment(n_pairs, args.max_steps, snr, snr, 0.0,
                             args.init_mode, args.coarse_grid_size,
                             args.lm_handoff_frac, args.device)
        state = env.reset()
        for _ in range(args.max_steps):
            state, _, _, _ = env.step(agent.act(state, deterministic=True))
        cbf_t, cbf_e = env.cbf_true, env.cbf_est
        att_t, att_e = env.att_true / 1000.0, env.att_est / 1000.0
        results[str(snr)] = {
            "cbf_rmse": torch.sqrt((cbf_t - cbf_e).square().mean()).item(),
            "cbf_mae": (cbf_t - cbf_e).abs().mean().item(),
            "cbf_ccc": ccc(cbf_t, cbf_e),
            "att_rmse": torch.sqrt((att_t - att_e).square().mean()).item(),
            "att_mae": (att_t - att_e).abs().mean().item(),
            "att_ccc": ccc(att_t, att_e),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Reproducible ASL SAC experiment runner")
    parser.add_argument("--config", required=True, help="JSON configuration path")
    parser.add_argument("--checkpoint", help="Optional SAC checkpoint for evaluation-only")
    parser.add_argument("--evaluate-only", action="store_true")
    parsed = parser.parse_args()
    with open(parsed.config) as f:
        supplied = json.load(f)
    config = DEFAULT_CONFIG | supplied
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config["seed"])
    args = SimpleNamespace(**config)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "resolved_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    cfg = SACConfig(11, 2, args.hidden, args.gamma, args.tau, args.lr, device=args.device)
    if parsed.evaluate_only:
        agent = SACAgent(cfg)
        if parsed.checkpoint:
            agent.load(parsed.checkpoint, map_location=args.device)
    else:
        agent = train(args)
    results = benchmark(agent, args, n_pairs=args.eval_pairs)
    with open(os.path.join(args.out_dir, "benchmark.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


# ## Notes / next steps
# 
# - Everything here (step sizes, reward shaping, `max_steps`, network width, `clean_fraction`) is a
#   tunable starting point — same spirit as the staged experiments (A → K) already run on the DNN
#   side of this project.
# - Once trained, overlay this notebook's `nrmse_vs_snr` numbers against the existing DNN and
#   Bayesian-method NRMSE/NMAE tables for a direct three-way comparison.
# - `agent.save(...)` / `agent.load(...)` checkpoint the actor, both critics, and the entropy
#   temperature — everything needed to resume training or run inference-only later.
# 
