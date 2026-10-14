import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rl_asl_4pld import (
    buxton_signal,
    PLD_SEC,
    LD_SEC,
    ALPHA,
    BETA,
    LAMBDA_BLOOD,
    T1_TISSUE,
    T1_BLOOD,
    SIGNAL_SCALE,
    CBF_MIN,
    CBF_MAX,
    ATT_MIN,
    ATT_MAX,
    RICIAN_NOISE_FACTOR,
    SACAgent,
    SACConfig,
)
from rl_4pld_validation import load_agent

def norm(x, lo, hi):
    return 2.0 * (x - lo) / (hi - lo) - 1.0

def denorm(x, lo, hi):
    return 0.5 * (x + 1.0) * (hi - lo) + lo

# Output directory
BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "controlled_4pld_benchmark")
os.makedirs(BENCHMARK_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Benchmark] Using device: {DEVICE}")

# -------------------------------------------------------------
# Metric Calculations
# -------------------------------------------------------------
def calc_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def calc_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))

def calc_lins_ccc(y_true, y_pred):
    # Population convention (ddof=0)
    mu_t = np.mean(y_true)
    mu_p = np.mean(y_pred)
    var_t = np.var(y_true)
    var_p = np.var(y_pred)
    cov = np.mean((y_true - mu_t) * (y_pred - mu_p))
    denom = var_t + var_p + (mu_t - mu_p) ** 2
    if denom < 1e-12:
        return 1.0
    return float((2.0 * cov) / denom)

def compute_all_metrics(y_true_cbf, y_pred_cbf, y_true_att, y_pred_att):
    return {
        "CBF_RMSE": calc_rmse(y_true_cbf, y_pred_cbf),
        "CBF_MAE": calc_mae(y_true_cbf, y_pred_cbf),
        "CBF_CCC": calc_lins_ccc(y_true_cbf, y_pred_cbf),
        "ATT_RMSE": calc_rmse(y_true_att, y_pred_att),
        "ATT_MAE": calc_mae(y_true_att, y_pred_att),
        "ATT_CCC": calc_lins_ccc(y_true_att, y_pred_att),
    }

# -------------------------------------------------------------
# Reference Signal & Noise Calibration
# -------------------------------------------------------------
def get_ref_signal():
    ref = buxton_signal(
        torch.tensor([[50.0]], device=DEVICE),
        torch.tensor([[1.6]], device=DEVICE),
        torch.tensor([2.0], device=DEVICE),
        torch.tensor([1.8], device=DEVICE),
    ).abs().item()
    return float(ref)

# -------------------------------------------------------------
# Test Cohort Generation
# -------------------------------------------------------------
def generate_fixed_test_cohort(n_samples=2000, seed=4242):
    print(f"[DataGen] Generating fixed test cohort: N={n_samples}, seed={seed}")
    rng = np.random.default_rng(seed)
    cbf_true = rng.uniform(CBF_MIN, CBF_MAX, n_samples).astype(np.float32)
    att_true = rng.uniform(ATT_MIN, ATT_MAX, n_samples).astype(np.float32)

    cbf_t = torch.tensor(cbf_true, device=DEVICE).unsqueeze(-1)
    att_t = torch.tensor(att_true, device=DEVICE).unsqueeze(-1)
    pld_t = PLD_SEC.to(DEVICE)
    ld_t = LD_SEC.to(DEVICE)

    with torch.no_grad():
        clean_signal = buxton_signal(cbf_t, att_t, pld_t, ld_t)

    ref_signal = get_ref_signal()
    test_cohort = {
        "cbf_true": cbf_true,
        "att_true": att_true,
        "clean_signal": clean_signal.cpu().numpy(),
        "ref_signal": ref_signal,
        "noisy_signals": {},
    }

    snr_levels = [10.0, 15.0, 20.0]
    for snr in snr_levels:
        # Fixed noise seed per SNR
        noise_rng = np.random.default_rng(seed + int(snr * 100))
        sigma = ref_signal / (RICIAN_NOISE_FACTOR * snr)
        clean_np = test_cohort["clean_signal"]
        nr = noise_rng.normal(0.0, sigma, size=clean_np.shape).astype(np.float32)
        ni = noise_rng.normal(0.0, sigma, size=clean_np.shape).astype(np.float32)
        obs_signal = np.sqrt((clean_np + nr) ** 2 + ni ** 2)
        test_cohort["noisy_signals"][snr] = obs_signal

    npz_path = os.path.join(BENCHMARK_DIR, "test_cohort_data.npz")
    np.savez_compressed(
        npz_path,
        cbf_true=cbf_true,
        att_true=att_true,
        clean_signal=test_cohort["clean_signal"],
        obs_snr10=test_cohort["noisy_signals"][10.0],
        obs_snr15=test_cohort["noisy_signals"][15.0],
        obs_snr20=test_cohort["noisy_signals"][20.0],
    )
    print(f"[DataGen] Saved test cohort to {npz_path}")
    return test_cohort

# -------------------------------------------------------------
# Model 1 & 2: Classical Solvers (Coarse Grid & Grid + LM)
# -------------------------------------------------------------
class ClassicalEstimators:
    def __init__(self, coarse_grid_size=31, device=DEVICE):
        self.device = device
        self.coarse_grid_size = coarse_grid_size
        self.pld = PLD_SEC.to(device)
        self.ld = LD_SEC.to(device)

        cbf = torch.linspace(CBF_MIN, CBF_MAX, coarse_grid_size, device=device)
        att = torch.linspace(ATT_MIN, ATT_MAX, coarse_grid_size, device=device)
        cbf_mesh, att_mesh = torch.meshgrid(cbf, att, indexing="ij")
        self.grid_cbf = cbf_mesh.reshape(-1)
        self.grid_att = att_mesh.reshape(-1)

        with torch.no_grad():
            self.candidates = buxton_signal(
                self.grid_cbf.unsqueeze(-1),
                self.grid_att.unsqueeze(-1),
                self.pld,
                self.ld,
            )

    def predict_grid(self, obs_signal_tensor):
        # obs_signal_tensor: (B, 4)
        t0 = time.perf_counter()
        with torch.no_grad():
            diff = obs_signal_tensor[:, None, :] - self.candidates[None, :, :]
            mse = diff.square().mean(dim=-1)
            best_idx = mse.argmin(dim=1)
            cbf_est = self.grid_cbf[best_idx].cpu().numpy()
            att_est = self.grid_att[best_idx].cpu().numpy()
        t1 = time.perf_counter()
        return cbf_est, att_est, (t1 - t0)

    def predict_grid_lm(self, obs_signal_tensor, n_iters=5, lam=0.01):
        t0 = time.perf_counter()
        with torch.no_grad():
            diff = obs_signal_tensor[:, None, :] - self.candidates[None, :, :]
            best_idx = diff.square().mean(dim=-1).argmin(dim=1)
            cbf_est = self.grid_cbf[best_idx].clone()
            att_est = self.grid_att[best_idx].clone()

            B = obs_signal_tensor.shape[0]
            for _ in range(n_iters):
                pred = buxton_signal(cbf_est.unsqueeze(-1), att_est.unsqueeze(-1), self.pld, self.ld)
                res = obs_signal_tensor - pred # (B, 4)

                # Numerical Jacobian wrt normalized parameters
                eps_cbf = 1e-3
                eps_att = 1e-4
                pred_cbf_p = buxton_signal((cbf_est + eps_cbf).unsqueeze(-1), att_est.unsqueeze(-1), self.pld, self.ld)
                pred_att_p = buxton_signal(cbf_est.unsqueeze(-1), (att_est + eps_att).unsqueeze(-1), self.pld, self.ld)

                J_cbf = (pred_cbf_p - pred) / eps_cbf # (B, 4)
                J_att = (pred_att_p - pred) / eps_att # (B, 4)

                J = torch.stack([J_cbf, J_att], dim=-1) # (B, 4, 2)
                Jt = J.transpose(1, 2) # (B, 2, 4)
                JtJ = torch.bmm(Jt, J) # (B, 2, 2)
                diag = torch.eye(2, device=self.device).unsqueeze(0).expand(B, -1, -1)
                H = JtJ + lam * diag
                g = torch.bmm(Jt, res.unsqueeze(-1)) # (B, 2, 1)

                try:
                    delta = torch.linalg.solve(H, g).squeeze(-1) # (B, 2)
                    cbf_est = (cbf_est + delta[:, 0]).clamp(CBF_MIN, CBF_MAX)
                    att_est = (att_est + delta[:, 1]).clamp(ATT_MIN, ATT_MAX)
                except Exception:
                    break

            cbf_final = cbf_est.cpu().numpy()
            att_final = att_est.cpu().numpy()
        t1 = time.perf_counter()
        return cbf_final, att_final, (t1 - t0)

# -------------------------------------------------------------
# Model 3: Supervised Deep Neural Network (BoundedNet)
# -------------------------------------------------------------
class BoundedNet(nn.Module):
    def __init__(self, input_dim=4, n_hidden_layers=4, n_neurons=64, y_min=0.0, y_max=100.0):
        super().__init__()
        self.y_min = y_min
        self.y_max = y_max
        layers = [nn.Linear(input_dim, n_neurons), nn.ELU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.ELU()]
        layers.append(nn.Linear(n_neurons, 1))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        raw = self.backbone(x)
        return self.y_min + (self.y_max - self.y_min) * torch.sigmoid(raw)

def train_supervised_dnn(n_samples=100000, epochs=25, batch_size=256, seed=20261014):
    print(f"\n[DNN Training] Generating training data: N={n_samples}, seed={seed}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    cbf_tr = rng.uniform(CBF_MIN, CBF_MAX, n_samples).astype(np.float32)
    att_tr = rng.uniform(ATT_MIN, ATT_MAX, n_samples).astype(np.float32)

    cbf_t = torch.tensor(cbf_tr, device=DEVICE).unsqueeze(-1)
    att_t = torch.tensor(att_tr, device=DEVICE).unsqueeze(-1)
    pld_t = PLD_SEC.to(DEVICE)
    ld_t = LD_SEC.to(DEVICE)

    with torch.no_grad():
        clean_tr = buxton_signal(cbf_t, att_t, pld_t, ld_t)

    ref_signal = get_ref_signal()
    # Continuous noise sampling: SNR uniform in [2, 20] + 10% clean
    snr_vals = rng.uniform(2.0, 20.0, (n_samples, 1)).astype(np.float32)
    is_clean = rng.random((n_samples, 1)) < 0.10

    sigma = ref_signal / (RICIAN_NOISE_FACTOR * snr_vals)
    sigma[is_clean] = 0.0

    sigma_t = torch.tensor(sigma, device=DEVICE)
    nr = torch.randn_like(clean_tr) * sigma_t
    ni = torch.randn_like(clean_tr) * sigma_t
    obs_tr = torch.sqrt((clean_tr + nr) ** 2 + ni ** 2)

    X_tr_n = obs_tr / ref_signal
    Y_cbf = torch.tensor(cbf_tr, device=DEVICE).unsqueeze(-1)
    Y_att = torch.tensor(att_tr, device=DEVICE).unsqueeze(-1)

    # Initialize CBF and ATT models
    cbf_net = BoundedNet(input_dim=4, n_hidden_layers=4, n_neurons=64, y_min=CBF_MIN, y_max=CBF_MAX).to(DEVICE)
    att_net = BoundedNet(input_dim=4, n_hidden_layers=4, n_neurons=64, y_min=ATT_MIN, y_max=ATT_MAX).to(DEVICE)

    opt_cbf = torch.optim.Adam(cbf_net.parameters(), lr=1e-3, weight_decay=1e-5)
    opt_att = torch.optim.Adam(att_net.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_tr_n, Y_cbf, Y_att)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"[DNN Training] Starting training: {epochs} epochs on {DEVICE}...")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        cbf_net.train()
        att_net.train()
        running_cbf_loss = 0.0
        running_att_loss = 0.0
        for bx, by_cbf, by_att in loader:
            pred_cbf = cbf_net(bx)
            loss_cbf = criterion(pred_cbf, by_cbf)
            opt_cbf.zero_grad()
            loss_cbf.backward()
            opt_cbf.step()
            running_cbf_loss += loss_cbf.item() * bx.size(0)

            pred_att = att_net(bx)
            loss_att = criterion(pred_att, by_att)
            opt_att.zero_grad()
            loss_att.backward()
            opt_att.step()
            running_att_loss += loss_att.item() * bx.size(0)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | CBF Loss: {running_cbf_loss/n_samples:.4f} | ATT Loss: {running_att_loss/n_samples:.4f}")
    t1 = time.time()
    training_wall_clock = t1 - t0
    print(f"[DNN Training] Completed in {training_wall_clock:.2f} s")

    # Save models
    cbf_path = os.path.join(BENCHMARK_DIR, "dnn_cbf_best.pth")
    att_path = os.path.join(BENCHMARK_DIR, "dnn_att_best.pth")
    torch.save(cbf_net.state_dict(), cbf_path)
    torch.save(att_net.state_dict(), att_path)

    return cbf_net, att_net, training_wall_clock

# -------------------------------------------------------------
# Model 4 & 5: SAC Evaluators (Pure SAC & Hybrid SAC + LM)
# -------------------------------------------------------------
def evaluate_sac_inference(agent, obs_signals_tensor, mode="pure_sac", max_steps=12, seed=4242):
    # obs_signals_tensor: (B, 4)
    t0 = time.perf_counter()
    B = obs_signals_tensor.shape[0]
    dev = obs_signals_tensor.device
    pld = PLD_SEC.to(dev)
    ld = LD_SEC.to(dev)
    ref_signal = get_ref_signal()

    # Reproducible random initialization
    torch.manual_seed(seed)
    cbf_est = torch.empty(B, device=dev).uniform_(CBF_MIN, CBF_MAX)
    att_est = torch.empty(B, device=dev).uniform_(ATT_MIN, ATT_MAX)

    lm_start_step = 9 if mode == "sac_lm" else max_steps

    with torch.no_grad():
        for step in range(max_steps):
            if step < lm_start_step:
                # SAC macro action
                pred_sig = buxton_signal(cbf_est.unsqueeze(-1), att_est.unsqueeze(-1), pld, ld)
                state = torch.cat([
                    obs_signals_tensor / ref_signal,
                    pred_sig / ref_signal,
                    norm(cbf_est, CBF_MIN, CBF_MAX).unsqueeze(-1),
                    norm(att_est, ATT_MIN, ATT_MAX).unsqueeze(-1),
                    torch.full((B, 1), step / float(max_steps), device=dev),
                ], dim=-1)

                action = agent.act(state, deterministic=True)
                decay = 1.0 - 0.5 * (step / float(max_steps))
                cbf_est = (cbf_est + action[:, 0] * 10.0 * decay).clamp(CBF_MIN, CBF_MAX)
                att_est = (att_est + action[:, 1] * 0.3 * decay).clamp(ATT_MIN, ATT_MAX)
            else:
                # Levenberg-Marquardt step
                pred = buxton_signal(cbf_est.unsqueeze(-1), att_est.unsqueeze(-1), pld, ld)
                res = obs_signals_tensor - pred

                eps_cbf = 1e-3
                eps_att = 1e-4
                pred_cbf_p = buxton_signal((cbf_est + eps_cbf).unsqueeze(-1), att_est.unsqueeze(-1), pld, ld)
                pred_att_p = buxton_signal(cbf_est.unsqueeze(-1), (att_est + eps_att).unsqueeze(-1), pld, ld)

                J_cbf = (pred_cbf_p - pred) / eps_cbf
                J_att = (pred_att_p - pred) / eps_att

                J = torch.stack([J_cbf, J_att], dim=-1) # (B, 4, 2)
                Jt = J.transpose(1, 2) # (B, 2, 4)
                JtJ = torch.bmm(Jt, J)
                diag = torch.eye(2, device=dev).unsqueeze(0).expand(B, -1, -1)
                H = JtJ + 0.01 * diag
                g = torch.bmm(Jt, res.unsqueeze(-1))

                try:
                    delta = torch.linalg.solve(H, g).squeeze(-1)
                    cbf_est = (cbf_est + delta[:, 0]).clamp(CBF_MIN, CBF_MAX)
                    att_est = (att_est + delta[:, 1]).clamp(ATT_MIN, ATT_MAX)
                except Exception:
                    pass

        cbf_final = cbf_est.cpu().numpy()
        att_final = att_est.cpu().numpy()
    t1 = time.perf_counter()
    return cbf_final, att_final, (t1 - t0)

# -------------------------------------------------------------
# Main Benchmark Pipeline
# -------------------------------------------------------------
def run_benchmark():
    print("======================================================================")
    print("   CONTROLLED ASL CBF/ATT PARAMETER ESTIMATION 4-PLD BENCHMARK")
    print("======================================================================")

    # 1. Generate or load fixed held-out test cohort
    test_cohort = generate_fixed_test_cohort(n_samples=2000, seed=4242)
    cbf_true = test_cohort["cbf_true"]
    att_true = test_cohort["att_true"]
    ref_signal = test_cohort["ref_signal"]

    # 2. Instantiate Classical Estimators
    classical = ClassicalEstimators(coarse_grid_size=31, device=DEVICE)

    # 3. Train Supervised DNN (BoundedNet)
    dnn_cbf, dnn_att, dnn_train_time = train_supervised_dnn(n_samples=100000, epochs=25, batch_size=256, seed=20261014)

    # 4. Load Trained SAC Agent
    sac_ckpt_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "experiments",
        "rl_sac_4pld_trained",
        "notebook_results",
        "sac_asl_final.pth",
    )
    print(f"\n[SAC] Loading verified checkpoint: {sac_ckpt_path}")
    sac_agent, _ = load_agent(sac_ckpt_path, DEVICE)
    sac_train_time = 1860.0 # From verified training log (1,000 iterations = 31 min)

    # 5. Execute Evaluation Across SNR 10, 15, 20
    models = ["GRID31", "GRID31_LM", "DNN_BOUNDED", "SAC_PURE", "SAC_LM"]
    snrs = [10.0, 15.0, 20.0]

    metrics_rows = []
    runtime_rows = []
    per_sample_rows = []

    for snr in snrs:
        print(f"\n--- Evaluating at SNR = {snr:.1f} ---")
        obs_np = test_cohort["noisy_signals"][snr]
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)
        X_norm_tensor = obs_tensor / ref_signal

        for model_name in models:
            if model_name == "GRID31":
                pred_cbf, pred_att, infer_time = classical.predict_grid(obs_tensor)
                train_time = 0.0
            elif model_name == "GRID31_LM":
                pred_cbf, pred_att, infer_time = classical.predict_grid_lm(obs_tensor, n_iters=5, lam=0.01)
                train_time = 0.0
            elif model_name == "DNN_BOUNDED":
                t0 = time.perf_counter()
                with torch.no_grad():
                    pred_cbf = dnn_cbf(X_norm_tensor).squeeze(-1).cpu().numpy()
                    pred_att = dnn_att(X_norm_tensor).squeeze(-1).cpu().numpy()
                t1 = time.perf_counter()
                infer_time = t1 - t0
                train_time = dnn_train_time
            elif model_name == "SAC_PURE":
                pred_cbf, pred_att, infer_time = evaluate_sac_inference(
                    sac_agent, obs_tensor, mode="pure_sac", max_steps=12, seed=4242
                )
                train_time = sac_train_time
            elif model_name == "SAC_LM":
                pred_cbf, pred_att, infer_time = evaluate_sac_inference(
                    sac_agent, obs_tensor, mode="sac_lm", max_steps=12, seed=4242
                )
                train_time = sac_train_time

            # Compute metrics
            m = compute_all_metrics(cbf_true, pred_cbf, att_true, pred_att)
            metrics_rows.append({
                "Model": model_name,
                "SNR": snr,
                "CBF_RMSE": m["CBF_RMSE"],
                "CBF_MAE": m["CBF_MAE"],
                "CBF_CCC": m["CBF_CCC"],
                "ATT_RMSE": m["ATT_RMSE"],
                "ATT_MAE": m["ATT_MAE"],
                "ATT_CCC": m["ATT_CCC"],
                "Training_Time_s": train_time,
                "Inference_Time_s": infer_time,
                "Per_Sample_ms": (infer_time / len(cbf_true)) * 1000.0,
            })

            runtime_rows.append({
                "Model": model_name,
                "SNR": snr,
                "Sample_Count": len(cbf_true),
                "Training_Time_s": train_time,
                "Inference_Time_s": infer_time,
                "Per_Sample_ms": (infer_time / len(cbf_true)) * 1000.0,
                "Device": DEVICE,
            })

            print(f"[{model_name:<11} | SNR {snr:4.1f}] CBF MAE={m['CBF_MAE']:6.3f}, CCC={m['CBF_CCC']:6.4f} | ATT MAE={m['ATT_MAE']:6.3f}s, CCC={m['ATT_CCC']:6.4f} | Infer={infer_time*1000:6.1f}ms")

            # Store per-sample predictions
            for idx in range(len(cbf_true)):
                per_sample_rows.append({
                    "Model": model_name,
                    "SNR": snr,
                    "Sample_ID": idx,
                    "CBF_True": float(cbf_true[idx]),
                    "CBF_Pred": float(pred_cbf[idx]),
                    "ATT_True": float(att_true[idx]),
                    "ATT_Pred": float(pred_att[idx]),
                })

    # Save CSV deliverables
    metrics_df = pd.DataFrame(metrics_rows)
    runtime_df = pd.DataFrame(runtime_rows)
    per_sample_df = pd.DataFrame(per_sample_rows)

    metrics_csv = os.path.join(BENCHMARK_DIR, "results_metrics.csv")
    runtime_csv = os.path.join(BENCHMARK_DIR, "results_runtime.csv")
    per_sample_csv = os.path.join(BENCHMARK_DIR, "per_sample_predictions.csv")

    metrics_df.to_csv(metrics_csv, index=False)
    runtime_df.to_csv(runtime_csv, index=False)
    per_sample_df.to_csv(per_sample_csv, index=False)

    print(f"\n[Output] Saved results_metrics.csv to {metrics_csv}")
    print(f"[Output] Saved results_runtime.csv to {runtime_csv}")
    print(f"[Output] Saved per_sample_predictions.csv ({len(per_sample_df)} rows) to {per_sample_csv}")

    # Generate Publication Plots
    generate_benchmark_plots(metrics_df, per_sample_df)

    return metrics_df, runtime_df

# -------------------------------------------------------------
# Plotting Suite
# -------------------------------------------------------------
def generate_benchmark_plots(metrics_df, per_sample_df):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {
        "GRID31": "#95a5a6",
        "GRID31_LM": "#e67e22",
        "DNN_BOUNDED": "#2ecc71",
        "SAC_PURE": "#3498db",
        "SAC_LM": "#9b59b6",
    }
    markers = {
        "GRID31": "s",
        "GRID31_LM": "^",
        "DNN_BOUNDED": "D",
        "SAC_PURE": "o",
        "SAC_LM": "v",
    }

    # Plot 1: Accuracy vs SNR (4 Subplots: CBF RMSE, ATT RMSE, CBF CCC, ATT CCC)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    models = ["GRID31", "GRID31_LM", "DNN_BOUNDED", "SAC_PURE", "SAC_LM"]

    for m in models:
        sub = metrics_df[metrics_df["Model"] == m].sort_values("SNR")
        snrs = sub["SNR"].values
        axes[0, 0].plot(snrs, sub["CBF_RMSE"], marker=markers[m], color=colors[m], lw=2.2, label=m)
        axes[0, 1].plot(snrs, sub["ATT_RMSE"], marker=markers[m], color=colors[m], lw=2.2, label=m)
        axes[1, 0].plot(snrs, sub["CBF_CCC"], marker=markers[m], color=colors[m], lw=2.2, label=m)
        axes[1, 1].plot(snrs, sub["ATT_CCC"], marker=markers[m], color=colors[m], lw=2.2, label=m)

    axes[0, 0].set_title("CBF RMSE vs SNR (Lower is Better)", fontsize=13, fontweight="bold")
    axes[0, 0].set_ylabel("CBF RMSE (mL/100g/min)", fontsize=11)
    axes[0, 0].set_xlabel("SNR", fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].set_title("ATT RMSE vs SNR (Lower is Better)", fontsize=13, fontweight="bold")
    axes[0, 1].set_ylabel("ATT RMSE (seconds)", fontsize=11)
    axes[0, 1].set_xlabel("SNR", fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=9)

    axes[1, 0].set_title("CBF Concordance (CCC) vs SNR (Higher is Better)", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Lin's CCC", fontsize=11)
    axes[1, 0].set_xlabel("SNR", fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=9)

    axes[1, 1].set_title("ATT Concordance (CCC) vs SNR (Higher is Better)", fontsize=13, fontweight="bold")
    axes[1, 1].set_ylabel("Lin's CCC", fontsize=11)
    axes[1, 1].set_xlabel("SNR", fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=9)

    plt.suptitle("4-PLD ASL Parameter Estimation Accuracy Comparison (N=2,000 per SNR)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    p1 = os.path.join(BENCHMARK_DIR, "plot_accuracy_vs_snr.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"[Plot] Saved accuracy vs SNR plot to {p1}")

    # Plot 2: Accuracy vs Training Time (SNR=15)
    sub15 = metrics_df[metrics_df["SNR"] == 15.0].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for _, row in sub15.iterrows():
        m = row["Model"]
        t_tr = max(row["Training_Time_s"], 0.1) # Avoid log(0)
        axes[0].scatter(t_tr, row["CBF_RMSE"], color=colors[m], s=160, marker=markers[m], label=m, zorder=5)
        axes[0].annotate(m, (t_tr * 1.15, row["CBF_RMSE"]), fontsize=10, fontweight="semibold")

        axes[1].scatter(t_tr, row["ATT_RMSE"], color=colors[m], s=160, marker=markers[m], label=m, zorder=5)
        axes[1].annotate(m, (t_tr * 1.15, row["ATT_RMSE"]), fontsize=10, fontweight="semibold")

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Training Time (seconds, log scale)", fontsize=11)
    axes[0].set_ylabel("CBF RMSE (mL/100g/min)", fontsize=11)
    axes[0].set_title("CBF Estimation Accuracy vs Training Cost (SNR=15)", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Training Time (seconds, log scale)", fontsize=11)
    axes[1].set_ylabel("ATT RMSE (seconds)", fontsize=11)
    axes[1].set_title("ATT Estimation Accuracy vs Training Cost (SNR=15)", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Accuracy vs. Training Time Trade-off (SNR=15)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p2 = os.path.join(BENCHMARK_DIR, "plot_accuracy_vs_training_time.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"[Plot] Saved accuracy vs training time plot to {p2}")

    # Plot 3: Accuracy vs Inference Speed
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in sub15.iterrows():
        m = row["Model"]
        ax.scatter(row["Per_Sample_ms"], row["CBF_RMSE"], color=colors[m], s=160, marker=markers[m], label=m, zorder=5)
        ax.annotate(f"{m} ({row['Per_Sample_ms']:.2f}ms)", (row["Per_Sample_ms"] * 1.1, row["CBF_RMSE"]), fontsize=10)

    ax.set_xscale("log")
    ax.set_xlabel("Inference Time per Sample (ms, log scale)", fontsize=11)
    ax.set_ylabel("CBF RMSE (mL/100g/min)", fontsize=11)
    ax.set_title("Inference Efficiency vs. Estimation Accuracy (SNR=15)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    p3 = os.path.join(BENCHMARK_DIR, "plot_accuracy_vs_inference_time.png")
    plt.tight_layout()
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"[Plot] Saved accuracy vs inference time plot to {p3}")

    # Plot 4: Scatter Plots of Predictions vs Ground Truth (SNR=15)
    fig, axes = plt.subplots(len(models), 2, figsize=(11, 4 * len(models)))
    sub_samples = per_sample_df[per_sample_df["SNR"] == 15.0]

    for idx, m in enumerate(models):
        m_data = sub_samples[sub_samples["Model"] == m]
        # CBF Scatter
        axes[idx, 0].scatter(m_data["CBF_True"], m_data["CBF_Pred"], alpha=0.25, s=8, color=colors[m])
        axes[idx, 0].plot([0, 100], [0, 100], "k--", lw=1.2)
        axes[idx, 0].set_title(f"{m} — CBF Predictions (SNR=15)", fontsize=11, fontweight="bold")
        axes[idx, 0].set_xlabel("True CBF (mL/100g/min)", fontsize=9)
        axes[idx, 0].set_ylabel("Pred CBF (mL/100g/min)", fontsize=9)
        axes[idx, 0].set_xlim(-2, 102)
        axes[idx, 0].set_ylim(-2, 102)
        axes[idx, 0].grid(True, alpha=0.3)

        # ATT Scatter
        axes[idx, 1].scatter(m_data["ATT_True"], m_data["ATT_Pred"], alpha=0.25, s=8, color=colors[m])
        axes[idx, 1].plot([0.5, 3.0], [0.5, 3.0], "k--", lw=1.2)
        axes[idx, 1].set_title(f"{m} — ATT Predictions (SNR=15)", fontsize=11, fontweight="bold")
        axes[idx, 1].set_xlabel("True ATT (seconds)", fontsize=9)
        axes[idx, 1].set_ylabel("Pred ATT (seconds)", fontsize=9)
        axes[idx, 1].set_xlim(0.4, 3.1)
        axes[idx, 1].set_ylim(0.4, 3.1)
        axes[idx, 1].grid(True, alpha=0.3)

    plt.suptitle("Predicted vs. Ground Truth Parameters (SNR=15, N=2,000)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p4 = os.path.join(BENCHMARK_DIR, "plot_scatter_predictions.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"[Plot] Saved scatter predictions plot to {p4}")

if __name__ == "__main__":
    run_benchmark()
