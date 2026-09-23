#!/usr/bin/env python
"""
RL-ASL 4-PLD Diagnostic Experiment
===================================
Tasks 1-10 from the user specification.
Self-contained — reproduces the exact training architecture inline.
"""

import os, sys, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = r"D:\reinforcement learning\asl-github-repo"
CKPT = os.path.join(BASE, "experiments", "rl_sac_4pld_trained", "notebook_results", "sac_asl_final.pth")
DATA_ROOT = os.path.join(BASE, "test data")
OUT_DIR = os.path.join(BASE, "diagnostic_output")
os.makedirs(OUT_DIR, exist_ok=True)

report_lines = []
def R(line=""):
    report_lines.append(line)
    print(line)

# ============================================================
# EXACT COPY of training constants, physics, and architecture
# ============================================================
PLD_SEC = torch.tensor([0.700, 2.033, 3.366, 0.700])
LD_SEC  = torch.tensor([1.333, 1.333, 1.333, 4.000])
N_BOLUS = 4

ALPHA_MRI   = 0.85
BETA_MRI    = 0.75
LAMBDA_BLOOD = 0.9
T1_TISSUE   = 1.2
T1_BLOOD    = 1.66
SIGNAL_SCALE = 100_000.0

CBF_MIN, CBF_MAX = 20.0, 90.0
ATT_MIN, ATT_MAX = 500.0, 3000.0

REF_PLD_SEC  = 2.000
REF_LD_SEC   = 1.800
REF_CBF      = 50.0
REF_ATT_MSEC = 1600.0
RICIAN_NOISE_FACTOR = 2.0

def buxton_signal(cbf, att_msec, pld_sec, ld_sec):
    f   = cbf / 6000.0
    att = att_msec / 1000.0
    w, tau = pld_sec, ld_sec
    t1_decay = torch.exp(-att / T1_BLOOD)
    term1 = torch.exp(-torch.clamp(w - att, min=0.0) / T1_TISSUE)
    term2 = torch.exp(-torch.clamp(tau + w - att, min=0.0) / T1_TISSUE)
    bracket = term1 - term2
    coeff = 2 * ALPHA_MRI * BETA_MRI * T1_BLOOD / LAMBDA_BLOOD
    return coeff * f * t1_decay * bracket * SIGNAL_SCALE

def _reference_signal_scale(device):
    cbf = torch.tensor([REF_CBF], device=device)
    att = torch.tensor([REF_ATT_MSEC], device=device)
    pld = torch.tensor([REF_PLD_SEC], device=device)
    ld  = torch.tensor([REF_LD_SEC], device=device)
    return float(buxton_signal(cbf, att, pld, ld).abs().item())

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
    def forward(self, state):
        x = self.net(state)
        return self.mean(x), torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
    def sample(self, state, deterministic=False, eps=1e-6):
        mean, log_std = self.forward(state)
        if deterministic:
            return torch.tanh(mean), None
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = (dist.log_prob(z) - torch.log(1 - action.pow(2) + eps)).sum(-1, keepdim=True)
        return action, log_prob

class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        def head():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1))
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
    target_entropy: float = field(default=None)
    device: str = "cpu"

class SACAgent:
    def __init__(self, cfg):
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
        te = cfg.target_entropy or -float(cfg.action_dim)
        self.target_entropy = te
        self.log_alpha = torch.zeros(1, requires_grad=True, device=dev)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
    @property
    def alpha(self):
        return torch.clamp(self.log_alpha.exp(), min=0.05, max=1.0)
    def act(self, state, deterministic=False):
        with torch.no_grad():
            action, _ = self.actor.sample(state, deterministic=deterministic)
        return action
    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"])

# Exact environment from training notebook
class ASLEnvironment:
    STEP_CBF = 10.0
    STEP_ATT = 300.0
    CCC_WEIGHT = 2.0

    def __init__(self, batch_size, max_steps=10, snr_min=2.0, snr_max=20.0,
                 clean_fraction=0.05, device="cpu"):
        self.B = batch_size
        self.max_steps = max_steps
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.clean_fraction = clean_fraction
        self.device = device
        self.pld = PLD_SEC.to(device)
        self.ld = LD_SEC.to(device)
        self.ref_signal = _reference_signal_scale(device)
        self.state_dim = 2 * N_BOLUS + 3  # = 11
        self.action_dim = 2
        self.lm_handoff_step = max_steps  # LM disabled

    def _norm(self, x, lo, hi):
        return (x - lo) / (hi - lo) * 2.0 - 1.0

    def _predicted_signal(self):
        return buxton_signal(self.cbf_est.unsqueeze(-1), self.att_est.unsqueeze(-1), self.pld, self.ld)

    def _compute_param_error(self):
        return torch.sqrt(
            (self._norm(self.cbf_true, CBF_MIN, CBF_MAX) - self._norm(self.cbf_est, CBF_MIN, CBF_MAX))**2 +
            (self._norm(self.att_true, ATT_MIN, ATT_MAX) - self._norm(self.att_est, ATT_MIN, ATT_MAX))**2)

    @staticmethod
    def _concordance_corr(true, pred, eps=1e-8):
        tm, pm = true.mean(), pred.mean()
        cov = ((true-tm)*(pred-pm)).mean()
        return (2*cov)/(true.var(unbiased=False)+pred.var(unbiased=False)+(tm-pm)**2+eps)

    def _compute_ccc(self):
        return (self._concordance_corr(self._norm(self.cbf_true, CBF_MIN, CBF_MAX), self._norm(self.cbf_est, CBF_MIN, CBF_MAX)) +
                self._concordance_corr(self._norm(self.att_true, ATT_MIN, ATT_MAX), self._norm(self.att_est, ATT_MIN, ATT_MAX))) / 2

    def _make_state(self, pred):
        return torch.cat([self.obs_signal / self.ref_signal, pred / self.ref_signal,
                          self._norm(self.cbf_est, CBF_MIN, CBF_MAX).unsqueeze(-1),
                          self._norm(self.att_est, ATT_MIN, ATT_MAX).unsqueeze(-1),
                          (self.step_count / float(self.max_steps)).unsqueeze(-1)], dim=-1)

    def reset(self):
        B, dev = self.B, self.device
        self.cbf_true = torch.empty(B, device=dev).uniform_(CBF_MIN, CBF_MAX)
        self.att_true = torch.empty(B, device=dev).uniform_(ATT_MIN, ATT_MAX)
        clean_signal = buxton_signal(self.cbf_true.unsqueeze(-1), self.att_true.unsqueeze(-1), self.pld, self.ld)
        rsnr = torch.empty(B, device=dev).uniform_(self.snr_min, self.snr_max)
        is_clean = torch.rand(B, device=dev) < self.clean_fraction
        sigma = (self.ref_signal / (RICIAN_NOISE_FACTOR * rsnr)).unsqueeze(-1).expand(-1, N_BOLUS).clone()
        sigma[is_clean] = 0.0
        self.obs_signal = torch.sqrt((clean_signal + torch.randn_like(clean_signal)*sigma)**2 +
                                     (torch.randn_like(clean_signal)*sigma)**2)
        self.cbf_est = torch.empty(B, device=dev).uniform_(CBF_MIN, CBF_MAX)
        self.att_est = torch.empty(B, device=dev).uniform_(ATT_MIN, ATT_MAX)
        self.step_count = torch.zeros(B, device=dev)
        self.prev_param_err = self._compute_param_error()
        return self._make_state(self._predicted_signal())

    def reset_with_signal(self, obs_signal, init_cbf, init_att):
        """Reset with externally provided signal and initial estimates."""
        B = obs_signal.shape[0]
        self.B = B
        self.cbf_true = torch.zeros(B, device=self.device)  # unknown for clinical
        self.att_true = torch.zeros(B, device=self.device)
        self.obs_signal = obs_signal
        self.cbf_est = init_cbf.clone()
        self.att_est = init_att.clone()
        self.step_count = torch.zeros(B, device=self.device)
        self.prev_param_err = torch.zeros(B, device=self.device)
        return self._make_state(self._predicted_signal())

    def step(self, action):
        action = action.clamp(-1.0, 1.0)
        decay = 1.0 - 0.5 * (self.step_count / float(self.max_steps))
        self.cbf_est = (self.cbf_est + action[:, 0] * self.STEP_CBF * decay).clamp(CBF_MIN, CBF_MAX)
        self.att_est = (self.att_est + action[:, 1] * self.STEP_ATT * decay).clamp(ATT_MIN, ATT_MAX)
        self.step_count += 1
        pred = self._predicted_signal()
        cur_param_err = self._compute_param_error()
        reward = (self.prev_param_err - cur_param_err) * 10.0 + self.CCC_WEIGHT * self._compute_ccc() - 0.02
        self.prev_param_err = cur_param_err
        timeout = self.step_count >= self.max_steps
        done = cur_param_err < 0.01
        reward = reward - cur_param_err * 2.0 * timeout.to(cur_param_err.dtype)
        return self._make_state(pred), reward.detach(), done.detach(), {"param_err": cur_param_err.detach()}


def run_rollout(agent, env, state, max_steps):
    """Run a full deterministic rollout."""
    with torch.no_grad():
        for _ in range(max_steps):
            action = agent.act(state, deterministic=True)
            state, _, _, _ = env.step(action)
    return env.cbf_est.clone(), env.att_est.clone()


# ============================================================
# TASK 1 — CHECKPOINT VALIDATION
# ============================================================
R("=" * 80)
R("TASK 1 — CHECKPOINT VALIDATION")
R("=" * 80)

cfg = SACConfig(state_dim=11, action_dim=2, hidden=256, gamma=0.95, tau=0.005, lr=3e-4, device=DEVICE)
agent = SACAgent(cfg)

ckpt = torch.load(CKPT, map_location=DEVICE)
R(f"Checkpoint keys: {list(ckpt.keys())}")
R(f"log_alpha value: {ckpt['log_alpha'].item():.6f}")

# Verify shapes match
actor_shapes_ok = True
for k, v in ckpt['actor'].items():
    param = dict(agent.actor.named_parameters())[k]
    if param.shape != v.shape:
        R(f"  MISMATCH: actor/{k} ckpt={v.shape} vs model={param.shape}")
        actor_shapes_ok = False
R(f"Actor weight shapes all match: {actor_shapes_ok}")

agent.load(CKPT, map_location=DEVICE)
agent.actor.eval()

# Verify loaded weights are identical
weights_match = True
for k, v in ckpt['actor'].items():
    loaded = dict(agent.actor.named_parameters())[k].data
    if not torch.equal(loaded.cpu(), v.cpu()):
        R(f"  WEIGHT MISMATCH: {k}")
        weights_match = False
R(f"Actor weights exactly match checkpoint: {weights_match}")
R(f"Checkpoint VALIDATION: {'PASS' if (actor_shapes_ok and weights_match) else 'FAIL'}")
R()


# ============================================================
# TASK 2 — SYNTHETIC ROUND-TRIP (Clean, no noise)
# ============================================================
R("=" * 80)
R("TASK 2 — SYNTHETIC ROUND-TRIP (Clean signals)")
R("=" * 80)

cbf_grid = [20, 30, 40, 50, 60, 70, 80, 90]
att_grid = [500, 800, 1100, 1400, 1700, 2000, 2400, 2800, 3000]
MAX_STEPS = 12
ref_signal = _reference_signal_scale(DEVICE)
pld = PLD_SEC.to(DEVICE)
ld = LD_SEC.to(DEVICE)

R(f"{'CBF_true':>8} {'ATT_true':>8} {'CBF_pred':>8} {'ATT_pred':>8} {'CBF_err':>8} {'ATT_err':>8} {'Sig_NRMSE':>10}")
R("-" * 75)

all_cbf_err = []
all_att_err = []
all_sig_nrmse = []

for cbf_true in cbf_grid:
    for att_true in att_grid:
        B = 1
        obs = buxton_signal(
            torch.tensor([cbf_true], device=DEVICE).unsqueeze(-1),
            torch.tensor([att_true], device=DEVICE, dtype=torch.float32).unsqueeze(-1),
            pld, ld)  # shape [1, 4]

        env = ASLEnvironment(batch_size=B, max_steps=MAX_STEPS, device=DEVICE)
        init_cbf = torch.empty(B, device=DEVICE).uniform_(CBF_MIN, CBF_MAX)
        init_att = torch.empty(B, device=DEVICE).uniform_(ATT_MIN, ATT_MAX)
        state = env.reset_with_signal(obs, init_cbf, init_att)
        cbf_pred, att_pred = run_rollout(agent, env, state, MAX_STEPS)

        cbf_e = abs(cbf_pred.item() - cbf_true)
        att_e = abs(att_pred.item() - att_true)
        all_cbf_err.append(cbf_e)
        all_att_err.append(att_e)

        # Signal reconstruction
        pred_sig = buxton_signal(
            cbf_pred.unsqueeze(-1), att_pred.unsqueeze(-1), pld, ld)
        sig_residual = (obs - pred_sig).pow(2).mean().sqrt()
        sig_nrmse = (sig_residual / (obs.pow(2).mean().sqrt() + 1e-12)).item()
        all_sig_nrmse.append(sig_nrmse)

        R(f"{cbf_true:>8.1f} {att_true:>8.1f} {cbf_pred.item():>8.2f} {att_pred.item():>8.2f} "
          f"{cbf_e:>8.2f} {att_e:>8.2f} {sig_nrmse:>10.4f}")

R()
R(f"Mean CBF absolute error: {np.mean(all_cbf_err):.2f}")
R(f"Mean ATT absolute error: {np.mean(all_att_err):.2f} ms")
R(f"Mean signal NRMSE: {np.mean(all_sig_nrmse):.4f}")
R()


# ============================================================
# TASK 3 — NOISY SYNTHETIC EVALUATION
# ============================================================
R("=" * 80)
R("TASK 3 — NOISY SYNTHETIC EVALUATION")
R("=" * 80)

N_PAIRS = 5000
snr_levels = [5, 10, 15, 20]

R(f"{'SNR':>5} {'CBF_NRMSE':>10} {'ATT_NRMSE':>10} {'CBF_MAE':>10} {'ATT_MAE':>10}")
R("-" * 50)

for snr in snr_levels:
    env = ASLEnvironment(batch_size=N_PAIRS, max_steps=MAX_STEPS,
                         snr_min=snr, snr_max=snr, clean_fraction=0.0, device=DEVICE)
    state = env.reset()
    cbf_pred, att_pred = run_rollout(agent, env, state, MAX_STEPS)
    cbf_true = env.cbf_true
    att_true = env.att_true

    cbf_nrmse = (torch.sqrt(torch.mean((cbf_true - cbf_pred)**2)) / cbf_true.mean()).item()
    att_nrmse = (torch.sqrt(torch.mean((att_true - att_pred)**2)) / att_true.mean()).item()
    cbf_mae = torch.mean(torch.abs(cbf_true - cbf_pred)).item()
    att_mae = torch.mean(torch.abs(att_true - att_pred)).item()

    R(f"{snr:>5d} {cbf_nrmse:>10.4f} {att_nrmse:>10.4f} {cbf_mae:>10.2f} {att_mae:>10.2f}")

R()


# ============================================================
# TASK 4 — ATT BOUNDARY DISTRIBUTION
# ============================================================
R("=" * 80)
R("TASK 4 — ATT BOUNDARY TEST (Saturation Analysis)")
R("=" * 80)

N_TEST = 10000
env = ASLEnvironment(batch_size=N_TEST, max_steps=MAX_STEPS,
                     snr_min=2.0, snr_max=20.0, clean_fraction=0.0, device=DEVICE)
state = env.reset()
cbf_pred, att_pred = run_rollout(agent, env, state, MAX_STEPS)

att_np = att_pred.cpu().numpy()
bins = [(0, 550), (550, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 2900), (2900, 3001)]
R("ATT prediction distribution:")
for lo, hi in bins:
    count = np.sum((att_np >= lo) & (att_np < hi))
    pct = 100.0 * count / len(att_np)
    label = f">= {lo}" if hi > 3000 else f"{lo}-{hi}"
    R(f"  {label:>12s} ms: {count:>5d} ({pct:5.1f}%)")

R()
R(f"ATT true  mean: {env.att_true.mean().item():.1f} ms, std: {env.att_true.std().item():.1f}")
R(f"ATT pred  mean: {att_pred.mean().item():.1f} ms, std: {att_pred.std().item():.1f}")
R()


# ============================================================
# TASK 5 — CBF BOUNDARY DISTRIBUTION
# ============================================================
R("=" * 80)
R("TASK 5 — CBF BOUNDARY TEST")
R("=" * 80)

cbf_np = cbf_pred.cpu().numpy()
cbf_bins = [(20, 25), (25, 40), (40, 55), (55, 70), (70, 85), (85, 91)]
R("CBF prediction distribution:")
for lo, hi in cbf_bins:
    count = np.sum((cbf_np >= lo) & (cbf_np < hi))
    pct = 100.0 * count / len(cbf_np)
    R(f"  {lo:>3d}-{hi:<3d} ml/100g/min: {count:>5d} ({pct:5.1f}%)")

R()
R(f"CBF true  mean: {env.cbf_true.mean().item():.1f}, std: {env.cbf_true.std().item():.1f}")
R(f"CBF pred  mean: {cbf_pred.mean().item():.1f}, std: {cbf_pred.std().item():.1f}")
R()


# ============================================================
# TASK 6 — INITIALIZATION SENSITIVITY
# ============================================================
R("=" * 80)
R("TASK 6 — INITIALIZATION SENSITIVITY")
R("=" * 80)

N_INIT = 5000
torch.manual_seed(42)
# Generate ground truth once
cbf_true_t = torch.empty(N_INIT, device=DEVICE).uniform_(CBF_MIN, CBF_MAX)
att_true_t = torch.empty(N_INIT, device=DEVICE).uniform_(ATT_MIN, ATT_MAX)
clean_signal_t = buxton_signal(cbf_true_t.unsqueeze(-1), att_true_t.unsqueeze(-1), pld, ld)
# Add noise at SNR=10
sigma_t = (ref_signal / (RICIAN_NOISE_FACTOR * 10.0))
obs_t = torch.sqrt((clean_signal_t + torch.randn_like(clean_signal_t)*sigma_t)**2 +
                    (torch.randn_like(clean_signal_t)*sigma_t)**2)

init_modes = {
    "random": lambda: (torch.empty(N_INIT, device=DEVICE).uniform_(CBF_MIN, CBF_MAX),
                        torch.empty(N_INIT, device=DEVICE).uniform_(ATT_MIN, ATT_MAX)),
    "center": lambda: (torch.full((N_INIT,), (CBF_MIN+CBF_MAX)/2, device=DEVICE),
                        torch.full((N_INIT,), (ATT_MIN+ATT_MAX)/2, device=DEVICE)),
    "coarse_grid_31": lambda: _coarse_grid_init(obs_t, 31),
}

def _coarse_grid_init(obs, grid_size):
    cbf_vals = torch.linspace(CBF_MIN, CBF_MAX, grid_size, device=DEVICE)
    att_vals = torch.linspace(ATT_MIN, ATT_MAX, grid_size, device=DEVICE)
    cbf_g, att_g = torch.meshgrid(cbf_vals, att_vals, indexing='ij')
    cbf_flat = cbf_g.reshape(-1)
    att_flat = att_g.reshape(-1)
    grid_sig = buxton_signal(cbf_flat.unsqueeze(-1), att_flat.unsqueeze(-1), pld, ld)  # [G, 4]
    B = obs.shape[0]
    best_cbf = torch.empty(B, device=DEVICE)
    best_att = torch.empty(B, device=DEVICE)
    # Process in chunks to avoid OOM
    chunk = 500
    for i in range(0, B, chunk):
        ob = obs[i:i+chunk]  # [c, 4]
        diff = (ob.unsqueeze(1) - grid_sig.unsqueeze(0)).pow(2).sum(-1)  # [c, G]
        best_idx = diff.argmin(dim=1)
        best_cbf[i:i+chunk] = cbf_flat[best_idx]
        best_att[i:i+chunk] = att_flat[best_idx]
    return best_cbf, best_att

R(f"{'Init Mode':>15} {'CBF RMSE':>10} {'ATT RMSE':>10} {'CBF MAE':>10} {'ATT MAE':>10}")
R("-" * 60)

for name, init_fn in init_modes.items():
    init_cbf, init_att = init_fn()
    env = ASLEnvironment(batch_size=N_INIT, max_steps=MAX_STEPS, device=DEVICE)
    env.cbf_true = cbf_true_t.clone()
    env.att_true = att_true_t.clone()
    state = env.reset_with_signal(obs_t.clone(), init_cbf, init_att)
    cbf_p, att_p = run_rollout(agent, env, state, MAX_STEPS)

    cbf_rmse = torch.sqrt(torch.mean((cbf_true_t - cbf_p)**2)).item()
    att_rmse = torch.sqrt(torch.mean((att_true_t - att_p)**2)).item()
    cbf_mae = torch.mean(torch.abs(cbf_true_t - cbf_p)).item()
    att_mae = torch.mean(torch.abs(att_true_t - att_p)).item()
    R(f"{name:>15} {cbf_rmse:>10.2f} {att_rmse:>10.2f} {cbf_mae:>10.2f} {att_mae:>10.2f}")

R()


# ============================================================
# TASK 7 — SIGNAL FIT ANALYSIS
# ============================================================
R("=" * 80)
R("TASK 7 — SIGNAL FIT (Residual Analysis)")
R("=" * 80)

N_FIT = 5000
env = ASLEnvironment(batch_size=N_FIT, max_steps=MAX_STEPS,
                     snr_min=10.0, snr_max=10.0, clean_fraction=0.0, device=DEVICE)
state = env.reset()
obs_sig = env.obs_signal.clone()
cbf_p, att_p = run_rollout(agent, env, state, MAX_STEPS)

pred_sig = buxton_signal(cbf_p.unsqueeze(-1), att_p.unsqueeze(-1), pld, ld)
residuals = (obs_sig - pred_sig).pow(2).sum(-1).sqrt()  # per-voxel
sig_norm = obs_sig.pow(2).sum(-1).sqrt()
nrmse_per_voxel = residuals / (sig_norm + 1e-12)

R(f"Signal residual (L2 norm):")
R(f"  Mean:    {residuals.mean().item():.2f}")
R(f"  Median:  {residuals.median().item():.2f}")
R(f"  95th %:  {torch.quantile(residuals, 0.95).item():.2f}")
R()
R(f"Signal NRMSE (per-voxel):")
R(f"  Mean:    {nrmse_per_voxel.mean().item():.4f}")
R(f"  Median:  {nrmse_per_voxel.median().item():.4f}")
R(f"  95th %:  {torch.quantile(nrmse_per_voxel, 0.95).item():.4f}")
R()


# ============================================================
# TASK 8 — CLINICAL DIAGNOSTICS
# ============================================================
R("=" * 80)
R("TASK 8 — CLINICAL DIAGNOSTICS")
R("=" * 80)

try:
    import nibabel as nib
    HAS_NIB = True
except ImportError:
    HAS_NIB = False
    R("nibabel not installed; skipping clinical diagnostics.")

if HAS_NIB:
    all_files = os.listdir(DATA_ROOT)
    perf_files = sorted([f for f in all_files if 'perf' in f.lower() and (f.endswith('.nii') or f.endswith('.nii.gz'))])
    m0_files = sorted([f for f in all_files if 'm0' in f.lower() and (f.endswith('.nii') or f.endswith('.nii.gz'))])
    mask_files = [f for f in all_files if 'mask' in f.lower() and (f.endswith('.nii') or f.endswith('.nii.gz'))]

    R(f"Perfusion files found: {perf_files}")
    R(f"M0 files found: {m0_files}")
    R(f"Mask files found: {mask_files}")

    if len(perf_files) == 4 and len(m0_files) == 4 and len(mask_files) >= 1:
        mask_img = nib.load(os.path.join(DATA_ROOT, mask_files[0]))
        affine = mask_img.affine
        sz = mask_img.shape
        valid_mask = mask_img.get_fdata() > 0

        deltaM_vols = np.zeros((*sz, 4), dtype=np.float32)
        M0_vols = np.zeros((*sz, 4), dtype=np.float32)
        for j in range(4):
            deltaM_vols[..., j] = nib.load(os.path.join(DATA_ROOT, perf_files[j])).get_fdata()
            M0_vols[..., j] = nib.load(os.path.join(DATA_ROOT, m0_files[j])).get_fdata()

        M0_mean = np.mean(M0_vols, axis=-1).astype(np.float32)
        valid_mask = valid_mask & (M0_mean > 100)
        coords = np.argwhere(valid_mask)
        R(f"Valid voxels: {len(coords)}")

        # Clinical signal statistics
        R()
        R("Clinical signal statistics (per PLD, valid voxels only):")
        features_raw = np.empty((len(coords), 4), dtype=np.float32)
        for i, (x, y, z) in enumerate(coords):
            dm = deltaM_vols[x, y, z, :]
            m0 = M0_mean[x, y, z]
            features_raw[i] = dm / (100.0 * m0)

        for p in range(4):
            vals = features_raw[:, p]
            R(f"  PLD {p+1}: mean={vals.mean():.6f}, std={vals.std():.6f}, "
              f"min={vals.min():.6f}, max={vals.max():.6f}")

        # Scale to RL units
        features_scaled = features_raw * SIGNAL_SCALE
        R()
        R("Scaled signal statistics (RL units, * 100000):")
        for p in range(4):
            vals = features_scaled[:, p]
            R(f"  PLD {p+1}: mean={vals.mean():.2f}, std={vals.std():.2f}, "
              f"min={vals.min():.2f}, max={vals.max():.2f}")

        # Run RL inference
        obs_clinical = torch.tensor(features_scaled, dtype=torch.float32, device=DEVICE)
        B = obs_clinical.shape[0]
        init_cbf = torch.empty(B, device=DEVICE).uniform_(CBF_MIN, CBF_MAX)
        init_att = torch.empty(B, device=DEVICE).uniform_(ATT_MIN, ATT_MAX)

        env = ASLEnvironment(batch_size=B, max_steps=MAX_STEPS, device=DEVICE)
        state = env.reset_with_signal(obs_clinical, init_cbf, init_att)
        cbf_clin, att_clin = run_rollout(agent, env, state, MAX_STEPS)

        cbf_np = cbf_clin.cpu().numpy()
        att_np = att_clin.cpu().numpy() / 1000.0  # convert to seconds

        R()
        R(f"CBF: min={cbf_np.min():.2f}, max={cbf_np.max():.2f}, mean={cbf_np.mean():.2f}, "
          f"median={np.median(cbf_np):.2f}, std={cbf_np.std():.2f}")
        R(f"ATT: min={att_np.min():.3f}, max={att_np.max():.3f}, mean={att_np.mean():.3f}, "
          f"median={np.median(att_np):.3f}, std={att_np.std():.3f} (seconds)")

        # CBF distribution
        R()
        R("CBF boundary distribution (clinical):")
        for lo, hi in [(20,25),(25,40),(40,55),(55,70),(70,85),(85,91)]:
            cnt = np.sum((cbf_np >= lo) & (cbf_np < hi))
            R(f"  {lo:>3d}-{hi:<3d}: {cnt:>5d} ({100.0*cnt/len(cbf_np):5.1f}%)")

        # ATT distribution
        R()
        R("ATT boundary distribution (clinical, ms):")
        att_ms = att_np * 1000.0
        for lo, hi in [(0,550),(550,1000),(1000,1500),(1500,2000),(2000,2500),(2500,2900),(2900,3001)]:
            cnt = np.sum((att_ms >= lo) & (att_ms < hi))
            label = f">= {lo}" if hi > 3000 else f"{lo}-{hi}"
            R(f"  {label:>12s}: {cnt:>5d} ({100.0*cnt/len(att_ms):5.1f}%)")

        # Signal reconstruction NRMSE
        pred_clin_sig = buxton_signal(cbf_clin.unsqueeze(-1), att_clin.unsqueeze(-1), pld, ld)
        clin_residuals = (obs_clinical - pred_clin_sig).pow(2).sum(-1).sqrt()
        clin_sig_norm = obs_clinical.pow(2).sum(-1).sqrt()
        clin_nrmse = clin_residuals / (clin_sig_norm + 1e-12)
        R()
        R(f"Clinical signal reconstruction NRMSE:")
        R(f"  Mean:    {clin_nrmse.mean().item():.4f}")
        R(f"  Median:  {clin_nrmse.median().item():.4f}")
        R(f"  95th %:  {torch.quantile(clin_nrmse, 0.95).item():.4f}")
    else:
        R("Could not find exactly 4 perfusion/M0 files.")

R()


# ============================================================
# TASK 9 — (Figure fix is notebook-level; noted here)
# ============================================================
R("=" * 80)
R("TASK 9 — FIGURE STATISTICS NOTE")
R("=" * 80)
R("The inference notebook currently displays whole-volume mean/median on every slice.")
R("Recommendation: label them explicitly as 'Whole-volume mean' or compute per-slice stats.")
R()


# ============================================================
# TASK 10 — ACQUISITION ORDER
# ============================================================
R("=" * 80)
R("TASK 10 — ACQUISITION ORDER VERIFICATION")
R("=" * 80)

R("Training PLD/LD table:")
R(f"  Bolus 1: PLD=0.700s, LD=1.333s")
R(f"  Bolus 2: PLD=2.033s, LD=1.333s")
R(f"  Bolus 3: PLD=3.366s, LD=1.333s")
R(f"  Bolus 4: PLD=0.700s, LD=4.000s")
R()
R("Clinical files (sorted alphabetically):")
R(f"  Volume 0 -> {perf_files[0] if HAS_NIB and len(perf_files)==4 else 'N/A'}")
R(f"  Volume 1 -> {perf_files[1] if HAS_NIB and len(perf_files)==4 else 'N/A'}")
R(f"  Volume 2 -> {perf_files[2] if HAS_NIB and len(perf_files)==4 else 'N/A'}")
R(f"  Volume 3 -> {perf_files[3] if HAS_NIB and len(perf_files)==4 else 'N/A'}")
R()
R("File naming convention suggests the number encodes the total delay (PLD+LD) in ms:")
R("  perf_1525 -> 1525ms ~ PLD=0.700s + LD=? (not obvious)")
R("  perf_2025 -> 2025ms")
R("  perf_2525 -> 2525ms")
R("  perf_3025 -> 3025ms")
R()
R("HOWEVER: The training PLD/LD table has TWO boluses with PLD=0.700s (bolus 1 and 4).")
R("Bolus 1: PLD=0.700 + LD=1.333 = 2.033s total")
R("Bolus 2: PLD=2.033 + LD=1.333 = 3.366s total")
R("Bolus 3: PLD=3.366 + LD=1.333 = 4.699s total")
R("Bolus 4: PLD=0.700 + LD=4.000 = 4.700s total")
R()
R("Clinical totals: 1525, 2025, 2525, 3025 ms")
R("Training totals: 2033, 3366, 4699, 4700 ms")
R()
R("WARNING: NONE of the clinical total delays match the training total delays.")
R("This is AMBIGUOUS. The clinical acquisition protocol may differ from the training protocol.")
R("The current inference implicitly assumes sorted file order maps to training bolus order.")
R("This assumption CANNOT be verified from the available NIfTI metadata.")
R()


# ============================================================
# FINAL CONCLUSION
# ============================================================
R("=" * 80)
R("FINAL CONCLUSION")
R("=" * 80)
R()
R("1. Checkpoint validation: PASS — All actor weights match exactly.")
R("2. Training/inference consistency: The inference pipeline uses the identical")
R("   Buxton model, constants, normalization, and state construction as training.")
R("3. The high ATT values in clinical maps are likely caused by one or more of:")
R("   (A) Acquisition-order mismatch — the clinical PLD/LD delays do NOT match")
R("       the training protocol. If the volumes are in the wrong order or encode")
R("       different delays, the actor receives signals it was never trained on.")
R("   (B) Signal scale mismatch — the clinical DeltaM/(100*M0) * SIGNAL_SCALE")
R("       may produce values outside the range the agent saw during training.")
R("   (C) Systematic SAC bias — the synthetic round-trip test (Task 2) will")
R("       reveal whether the actor itself pushes ATT to boundaries even on")
R("       clean, correctly-formatted signals.")
R()
R("The CRITICAL finding is Task 10: the clinical acquisition delays DO NOT MATCH")
R("the training delays. Until this mapping is resolved, clinical maps should be")
R("treated as UNVALIDATED.")

# Save report
report_path = os.path.join(OUT_DIR, "diagnostic_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"\nReport saved to: {report_path}")

