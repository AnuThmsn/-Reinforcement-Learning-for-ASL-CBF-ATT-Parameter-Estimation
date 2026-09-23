"""Inference-only validation for the saved 4-PLD SAC actor.

This module intentionally keeps clinical inference separate from ASLEnvironment.
It reuses only the training module's actor and Buxton implementation.
"""

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from rl_asl_4pld import (
    ALPHA,
    ATT_MAX,
    ATT_MIN,
    BETA,
    CBF_MAX,
    CBF_MIN,
    LD_SEC,
    LAMBDA_BLOOD,
    N_BOLUS,
    PLD_SEC,
    RICIAN_NOISE_FACTOR,
    SIGNAL_SCALE,
    T1_BLOOD,
    T1_TISSUE,
    SACAgent,
    SACConfig,
    buxton_signal,
)


CHECKPOINT_KEYS = ("actor", "critic", "critic_target", "log_alpha")


def norm(value, low, high):
    return (value - low) / (high - low) * 2.0 - 1.0


def denorm(value, low, high):
    return (value + 1.0) / 2.0 * (high - low) + low


class InferenceOnly4PLD:
    """Ground-truth-free state and rollout helper for deployment."""

    def __init__(self, observed_signal, max_steps=12, init_mode="random", coarse_grid_size=31):
        self.obs_signal = observed_signal
        self.batch_size = observed_signal.shape[0]
        self.max_steps = max_steps
        self.init_mode = init_mode
        self.coarse_grid_size = coarse_grid_size
        self.device = observed_signal.device
        self.pld = PLD_SEC.to(self.device)
        self.ld = LD_SEC.to(self.device)
        self.ref_signal = float(
            buxton_signal(
                torch.tensor([[50.0]], device=self.device),
                torch.tensor([[1600.0]], device=self.device),
                torch.tensor([2.0], device=self.device),
                torch.tensor([1.8], device=self.device),
            ).abs().item()
        )
        self.cbf_est = None
        self.att_est = None
        self.step_count = None

    @property
    def state_dim(self):
        return 2 * N_BOLUS + 3

    def predicted_signal(self):
        return buxton_signal(self.cbf_est[:, None], self.att_est[:, None], self.pld, self.ld)

    def make_state(self):
        return torch.cat(
            [
                self.obs_signal / self.ref_signal,
                self.predicted_signal() / self.ref_signal,
                norm(self.cbf_est, CBF_MIN, CBF_MAX)[:, None],
                norm(self.att_est, ATT_MIN, ATT_MAX)[:, None],
                (self.step_count / float(self.max_steps))[:, None],
            ],
            dim=-1,
        )

    def coarse_grid_init(self):
        cbf = torch.linspace(CBF_MIN, CBF_MAX, self.coarse_grid_size, device=self.device)
        att = torch.linspace(ATT_MIN, ATT_MAX, self.coarse_grid_size, device=self.device)
        cbf_mesh, att_mesh = torch.meshgrid(cbf, att, indexing="ij")
        candidates = buxton_signal(cbf_mesh.reshape(-1, 1), att_mesh.reshape(-1, 1), self.pld, self.ld)
        best = (self.obs_signal[:, None, :] - candidates[None, :, :]).square().mean(dim=-1).argmin(dim=1)
        self.cbf_est = cbf_mesh.reshape(-1)[best]
        self.att_est = att_mesh.reshape(-1)[best]

    def reset(self):
        if self.init_mode == "random":
            self.cbf_est = torch.empty(self.batch_size, device=self.device).uniform_(CBF_MIN, CBF_MAX)
            self.att_est = torch.empty(self.batch_size, device=self.device).uniform_(ATT_MIN, ATT_MAX)
        elif self.init_mode == "center":
            self.cbf_est = torch.full((self.batch_size,), (CBF_MIN + CBF_MAX) / 2.0, device=self.device)
            self.att_est = torch.full((self.batch_size,), (ATT_MIN + ATT_MAX) / 2.0, device=self.device)
        elif self.init_mode == "coarse_grid":
            self.coarse_grid_init()
        else:
            raise ValueError(f"Unknown init_mode: {self.init_mode}")
        self.step_count = torch.zeros(self.batch_size, device=self.device)
        return self.make_state()

    def step(self, action):
        action = action.clamp(-1.0, 1.0)
        decay = 1.0 - 0.5 * self.step_count / float(self.max_steps)
        self.cbf_est = (self.cbf_est + action[:, 0] * 10.0 * decay).clamp(CBF_MIN, CBF_MAX)
        self.att_est = (self.att_est + action[:, 1] * 300.0 * decay).clamp(ATT_MIN, ATT_MAX)
        self.step_count += 1
        return self.make_state()


def load_agent(checkpoint, device):
    checkpoint = Path(checkpoint)
    raw = torch.load(checkpoint, map_location=device)
    agent = SACAgent(SACConfig(11, 2, 256, 0.95, 0.005, 3e-4, device=device))
    actor_result = agent.actor.load_state_dict(raw["actor"], strict=True)
    critic_result = agent.critic.load_state_dict(raw["critic"], strict=True)
    target_result = agent.critic_target.load_state_dict(raw["critic_target"], strict=True)
    agent.actor.eval()
    return agent, {
        "checkpoint": str(checkpoint),
        "keys": sorted(raw.keys()),
        "expected_keys": list(CHECKPOINT_KEYS),
        "keys_match": sorted(raw.keys()) == sorted(CHECKPOINT_KEYS),
        "actor_missing": actor_result.missing_keys,
        "actor_unexpected": actor_result.unexpected_keys,
        "critic_missing": critic_result.missing_keys,
        "critic_unexpected": critic_result.unexpected_keys,
        "target_missing": target_result.missing_keys,
        "target_unexpected": target_result.unexpected_keys,
        "state_dim": agent.cfg.state_dim,
        "action_dim": agent.cfg.action_dim,
        "hidden": agent.cfg.hidden,
    }


def rollout(agent, observed_signal, init_mode, max_steps=12, coarse_grid_size=31, trajectory=False):
    env = InferenceOnly4PLD(observed_signal, max_steps, init_mode, coarse_grid_size)
    state = env.reset()
    initial_signal = env.predicted_signal().detach().clone()
    records = []
    with torch.no_grad():
        for step in range(max_steps):
            action = agent.act(state, deterministic=True)
            if trajectory:
                records.append(
                    {
                        "step": step,
                        "cbf": env.cbf_est.detach().cpu().tolist(),
                        "att": env.att_est.detach().cpu().tolist(),
                        "action_cbf": action[:, 0].cpu().tolist(),
                        "action_att": action[:, 1].cpu().tolist(),
                    }
                )
            state = env.step(action)
    final_signal = env.predicted_signal().detach()
    return env, initial_signal, final_signal, records


def nrmse(error, scale):
    return (torch.sqrt(error.square().mean()) / scale).item()


def metrics(true_cbf, true_att, pred_cbf, pred_att, observed, initial_signal, final_signal):
    fit_error = final_signal - observed
    initial_error = initial_signal - observed
    return {
        "cbf_mae": (pred_cbf - true_cbf).abs().mean().item(),
        "cbf_rmse": torch.sqrt((pred_cbf - true_cbf).square().mean()).item(),
        "cbf_nrmse": nrmse(pred_cbf - true_cbf, true_cbf.mean()),
        "att_mae": (pred_att - true_att).abs().mean().item(),
        "att_rmse": torch.sqrt((pred_att - true_att).square().mean()).item(),
        "att_nrmse": nrmse(pred_att - true_att, true_att.mean()),
        "signal_nrmse": nrmse(fit_error, observed.abs().mean()),
        "initial_signal_nrmse": nrmse(initial_error, observed.abs().mean()),
        "signal_mae": fit_error.abs().mean().item(),
        "signal_max_abs": fit_error.abs().max().item(),
    }


def synthetic_signals(cbf_values, att_values, snr, device):
    cbf = torch.tensor(cbf_values, dtype=torch.float32, device=device)
    att = torch.tensor(att_values, dtype=torch.float32, device=device)
    clean = buxton_signal(cbf[:, None], att[:, None], PLD_SEC.to(device), LD_SEC.to(device))
    if snr == "clean":
        return cbf, att, clean
    reference = buxton_signal(torch.tensor([[50.0]], device=device), torch.tensor([[1600.0]], device=device),
                              torch.tensor([2.0], device=device), torch.tensor([1.8], device=device)).abs().item()
    sigma = reference / (RICIAN_NOISE_FACTOR * float(snr))
    nr = torch.randn_like(clean) * sigma
    ni = torch.randn_like(clean) * sigma
    return cbf, att, torch.sqrt((clean + nr).square() + ni.square())


def run_synthetic_validation(agent, device):
    cbf_values = [20, 30, 40, 50, 60, 70, 80, 90]
    att_values = [500, 800, 1100, 1400, 1700, 2000, 2400, 2800, 3000]
    pairs = [(cbf, att) for cbf in cbf_values for att in att_values]
    true_cbf = [p[0] for p in pairs]
    true_att = [p[1] for p in pairs]
    rows = []
    init_rows = []
    boundary = {}
    comparison = {}
    trajectories = []
    for snr in ("clean", 5, 10, 15, 20):
        cbf, att, observed = synthetic_signals(true_cbf, true_att, snr, device)
        env, initial, final, _ = rollout(agent, observed, "random")
        row = {"snr": snr, **metrics(cbf, att, env.cbf_est, env.att_est, observed, initial, final)}
        rows.append(row)
        if snr == 10:
            comparison = {
                "true_cbf": cbf.cpu().tolist(),
                "pred_cbf": env.cbf_est.detach().cpu().tolist(),
                "true_att": att.cpu().tolist(),
                "pred_att": env.att_est.detach().cpu().tolist(),
            }
            sample_indices = [0, 20, 40, 60]
            _, _, _, trajectory_rows = rollout(agent, observed[sample_indices], "random", trajectory=True)
            for sample_position, sample_index in enumerate(sample_indices):
                sample_steps = []
                for step_row in trajectory_rows:
                    sample_steps.append({
                        key: (value[sample_position] if isinstance(value, list) else value)
                        for key, value in step_row.items()
                    })
                trajectories.append({"sample_index": sample_index, "steps": sample_steps})
        if snr == 10:
            for mode in ("random", "center", "coarse_grid"):
                e, initial_mode, final_mode, _ = rollout(agent, observed, mode)
                init_rows.append({"initialization": mode, **metrics(cbf, att, e.cbf_est, e.att_est, observed, initial_mode, final_mode)})
        tolerance_cbf = (CBF_MAX - CBF_MIN) * 0.01
        tolerance_att = (ATT_MAX - ATT_MIN) * 0.01
        boundary[str(snr)] = {
            "cbf_lower_near_pct": ((env.cbf_est <= CBF_MIN + tolerance_cbf).float().mean() * 100).item(),
            "cbf_upper_near_pct": ((env.cbf_est >= CBF_MAX - tolerance_cbf).float().mean() * 100).item(),
            "att_lower_near_pct": ((env.att_est <= ATT_MIN + tolerance_att).float().mean() * 100).item(),
            "att_upper_near_pct": ((env.att_est >= ATT_MAX - tolerance_att).float().mean() * 100).item(),
        }
    return rows, init_rows, boundary, comparison, trajectories


def clinical_preprocess(data_root):
    root = Path(data_root)
    perf_files = sorted(p for p in root.iterdir() if "perf" in p.name.lower() and p.suffix in (".nii", ".gz"))
    m0_files = sorted(p for p in root.iterdir() if "m0" in p.name.lower() and p.suffix in (".nii", ".gz"))
    mask_files = sorted(p for p in root.iterdir() if "mask" in p.name.lower() and p.suffix in (".nii", ".gz"))
    result = {"perf_files": [p.name for p in perf_files], "m0_files": [p.name for p in m0_files], "mask_files": [p.name for p in mask_files]}
    if len(perf_files) != 4 or len(m0_files) != 4 or not mask_files:
        result["status"] = "MISSING_INPUTS"
        return result, None
    result["status"] = "AMBIGUOUS_ACQUISITION_ORDER"
    result["message"] = "Filenames do not encode the training PLD/LD table [0.700,2.033,3.366,0.700]/[1.333,1.333,1.333,4.000]; clinical maps were not inferred."
    mask_img = nib.load(mask_files[0])
    valid_mask = mask_img.get_fdata() > 0
    delta = np.stack([nib.load(p).get_fdata() for p in perf_files], axis=-1).astype(np.float32)
    m0 = np.stack([nib.load(p).get_fdata() for p in m0_files], axis=-1).astype(np.float32)
    m0_mean = m0.mean(axis=-1)
    valid_mask &= m0_mean > 100
    observed_fraction = np.divide(
        delta,
        100.0 * m0_mean[..., None],
        out=np.zeros_like(delta, dtype=np.float32),
        where=m0_mean[..., None] > 0,
    )
    observed = observed_fraction * SIGNAL_SCALE
    result["valid_voxels"] = int(valid_mask.sum())
    result["m0_statistics"] = {k: float(v) for k, v in zip(("min", "mean", "max"), (m0_mean[valid_mask].min(), m0_mean[valid_mask].mean(), m0_mean[valid_mask].max()))}
    result["observed_delta_statistics"] = {k: float(v) for k, v in zip(("min", "mean", "max"), (delta[valid_mask].min(), delta[valid_mask].mean(), delta[valid_mask].max()))}
    result["scaled_signal_statistics"] = [{"pld_position": i + 1, "min": float(observed[..., i][valid_mask].min()), "mean": float(observed[..., i][valid_mask].mean()), "max": float(observed[..., i][valid_mask].max())} for i in range(4)]
    return result, {"mask_img": mask_img, "valid_mask": valid_mask, "observed": observed, "shape": mask_img.shape}


def run_validation(checkpoint, data_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, checkpoint_report = load_agent(checkpoint, device)
    rows, init_rows, boundary, comparison, trajectories = run_synthetic_validation(agent, device)
    clinical_report, clinical_data = clinical_preprocess(data_root)
    report = {
        "checkpoint": checkpoint_report,
        "training_contract": {
            "state_dim": 11, "action_dim": 2, "hidden": 256, "pld_sec": PLD_SEC.tolist(), "ld_sec": LD_SEC.tolist(),
            "signal_scale": SIGNAL_SCALE, "cbf_bounds": [CBF_MIN, CBF_MAX], "att_bounds": [ATT_MIN, ATT_MAX],
            "initialization": "random", "max_steps": 12, "lm_handoff_step": 12, "lm_active": False,
            "actor_mode": "eval", "deterministic_actions": True, "optimizer_called": False,
            "constants": {"ALPHA": ALPHA, "BETA": BETA, "LAMBDA_BLOOD": LAMBDA_BLOOD, "T1_TISSUE": T1_TISSUE, "T1_BLOOD": T1_BLOOD},
        },
        "synthetic": rows,
        "initialization": init_rows,
        "boundary": boundary,
        "comparison_snr_10": comparison,
        "trajectories": trajectories,
        "clinical": clinical_report,
        "clinical_inference_status": clinical_report["status"],
    }
    path = Path(output_root) / "inference_summary.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (Path(output_root) / "synthetic_trajectories.json").write_text(json.dumps(trajectories, indent=2), encoding="utf-8")
    return report, str(path)


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    report, path = run_validation(base / "experiments" / "rl_sac_4pld_trained" / "notebook_results" / "sac_asl_final.pth", base / "test data", base / "rl_inference_output")
    print(json.dumps(report, indent=2))
    print("Saved:", path)