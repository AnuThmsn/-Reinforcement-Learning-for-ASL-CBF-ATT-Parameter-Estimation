import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# Import definitions from the notebook script
import sys
sys.path.append(os.getcwd())
# Fix any arg-parsing issues when importing as a module
import importlib.util
spec = importlib.util.spec_from_file_location("script", "rl-asl-4pld.py")
script = importlib.util.module_from_spec(spec)
sys.modules["script"] = script
spec.loader.exec_module(script)

from types import SimpleNamespace
args = SimpleNamespace(
    hidden=256,
    gamma=0.95,
    tau=0.005,
    lr=3e-4,
    max_steps=12,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
device = args.device

# Create diagnostic output directory
os.makedirs("diagnostics", exist_ok=True)

# 1. Load the existing trained SAC model
cfg = script.SACConfig(
    state_dim=11, # 2*N_BOLUS + 3
    action_dim=2,
    hidden=args.hidden,
    gamma=args.gamma,
    tau=args.tau,
    lr=args.lr,
    device=device
)
agent = script.SACAgent(cfg)
ckpt_path = os.path.join("experiments", "rl_sac_4pld_trained", "notebook_results", "sac_asl_final.pth")
if not os.path.exists(ckpt_path):
    print(f"ERROR: {ckpt_path} not found.")
    sys.exit(1)
agent.load(ckpt_path, map_location=device)

# 2. Run diagnostic evaluation (SNR 10, 15, 20) with fixed seeds
def ccc(true, pred, eps=1e-8):
    true_mean, pred_mean = true.mean(), pred.mean()
    true_var = true.var(unbiased=False)
    pred_var = pred.var(unbiased=False)
    covariance = ((true - true_mean) * (pred - pred_mean)).mean()
    return (2.0 * covariance / (true_var + pred_var + (true_mean - pred_mean).square() + eps)).item()

snr_grid = [10, 15, 20]
n_pairs = 2000

for snr in snr_grid:
    print(f"\n--- SNR {snr} Diagnostic ---")
    seed = 42 + snr
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env = script.ASLEnvironment(batch_size=n_pairs, max_steps=args.max_steps,
                         snr_min=snr, snr_max=snr, clean_fraction=0.0, device=device)
    # Ensure LM is disabled for pure SAC evaluation (like the fix I just made)
    env.lm_handoff_step = args.max_steps
    
    state = env.reset()
    with torch.no_grad():
        for _ in range(args.max_steps):
            action = agent.act(state, deterministic=True)
            state, _, _, _ = env.step(action)
            
    cbf_t = env.cbf_true.cpu().numpy()
    cbf_e = env.cbf_est.cpu().numpy()
    att_t = env.att_true.cpu().numpy()
    att_e = env.att_est.cpu().numpy()
    
    # ----------------------------------------------------
    # UNIT BUG FIX: NO division by 1000.0 for att_t/att_e!
    # They are already in [0.5, 3.0] seconds.
    # ----------------------------------------------------
    
    # Calculate Metrics
    cbf_rmse = np.sqrt(np.mean((cbf_t - cbf_e) ** 2))
    cbf_mae = np.mean(np.abs(cbf_t - cbf_e))
    cbf_ccc = ccc(torch.tensor(cbf_t), torch.tensor(cbf_e))
    cbf_bias = np.mean(cbf_e - cbf_t)
    cbf_med = np.median(np.abs(cbf_t - cbf_e))
    
    att_rmse = np.sqrt(np.mean((att_t - att_e) ** 2))
    att_mae = np.mean(np.abs(att_t - att_e))
    att_ccc = ccc(torch.tensor(att_t), torch.tensor(att_e))
    att_bias = np.mean(att_e - att_t)
    att_med = np.median(np.abs(att_t - att_e))
    
    print("CBF:")
    print(f"  RMSE: {cbf_rmse:.4f} | MAE: {cbf_mae:.4f} | CCC: {cbf_ccc:.4f}")
    print(f"  Bias: {cbf_bias:.4f} | Median AE: {cbf_med:.4f}")
    print(f"  True Range: {cbf_t.mean():.2f} ± {cbf_t.std():.2f} [{cbf_t.min():.2f}, {cbf_t.max():.2f}]")
    print(f"  Pred Range: {cbf_e.mean():.2f} ± {cbf_e.std():.2f} [{cbf_e.min():.2f}, {cbf_e.max():.2f}]")
    
    print("\nATT (seconds):")
    print(f"  RMSE: {att_rmse:.4f} | MAE: {att_mae:.4f} | CCC: {att_ccc:.4f}")
    print(f"  Bias: {att_bias:.4f} | Median AE: {att_med:.4f}")
    print(f"  True Range: {att_t.mean():.2f} ± {att_t.std():.2f} [{att_t.min():.2f}, {att_t.max():.2f}]")
    print(f"  Pred Range: {att_e.mean():.2f} ± {att_e.std():.2f} [{att_e.min():.2f}, {att_e.max():.2f}]")
    
    # Diagnostic Plots (Only save for SNR 20 to avoid clutter)
    if snr == 20:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.scatter(cbf_t, cbf_e, alpha=0.3, s=2)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.xlabel("True CBF")
        plt.ylabel("Pred CBF")
        plt.title("CBF: True vs Pred (SNR 20)")
        
        plt.subplot(1, 2, 2)
        plt.scatter(att_t, att_e, alpha=0.3, s=2)
        plt.plot([0.5, 3.0], [0.5, 3.0], 'r--')
        plt.xlabel("True ATT (s)")
        plt.ylabel("Pred ATT (s)")
        plt.title("ATT: True vs Pred (SNR 20)")
        plt.tight_layout()
        plt.savefig("diagnostics/scatter_true_vs_pred.png")
        plt.close()
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.scatter(cbf_t, cbf_e - cbf_t, alpha=0.3, s=2)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel("True CBF")
        plt.ylabel("CBF Error (Pred - True)")
        plt.title("CBF Error vs True (SNR 20)")
        
        plt.subplot(1, 2, 2)
        plt.scatter(att_t, att_e - att_t, alpha=0.3, s=2)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel("True ATT (s)")
        plt.ylabel("ATT Error (Pred - True)")
        plt.title("ATT Error vs True (SNR 20)")
        plt.tight_layout()
        plt.savefig("diagnostics/scatter_error_vs_true.png")
        plt.close()
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(cbf_e - cbf_t, bins=50)
        plt.xlabel("CBF Error")
        plt.title("CBF Error Histogram")
        
        plt.subplot(1, 2, 2)
        plt.hist(att_e - att_t, bins=50)
        plt.xlabel("ATT Error (s)")
        plt.title("ATT Error Histogram")
        plt.tight_layout()
        plt.savefig("diagnostics/histogram_error.png")
        plt.close()
        
        # 4. Check whether ATT is systematically biased (Binning)
        print("\n--- ATT Systematic Bias Analysis (SNR 20) ---")
        bins = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]
        print(f"{'Bin (s)':<12} | {'RMSE':<8} | {'MAE':<8} | {'Bias':<8} | {'N'}")
        for (low, high) in bins:
            mask = (att_t >= low) & (att_t < high)
            if high == 3.0: mask = (att_t >= low) & (att_t <= high)
            b_true = att_t[mask]
            b_pred = att_e[mask]
            if len(b_true) > 0:
                b_rmse = np.sqrt(np.mean((b_true - b_pred) ** 2))
                b_mae = np.mean(np.abs(b_true - b_pred))
                b_bias = np.mean(b_pred - b_true)
                print(f"{low:.1f}-{high:.1f}      | {b_rmse:<8.4f} | {b_mae:<8.4f} | {b_bias:<8.4f} | {len(b_true)}")
            else:
                print(f"{low:.1f}-{high:.1f}      | N/A")

# 7. Check identifiability of ATT with these 4 PLDs
print("\n--- Physics Identifiability Analysis ---")
cbf_levels = [20.0, 40.0, 60.0, 80.0]
att_grid = np.linspace(0.5, 3.0, 26) # 0.1 step

plt.figure(figsize=(8, 6))
for c in cbf_levels:
    signals_for_cbf = []
    for a in att_grid:
        c_t = torch.tensor([c], device=device)
        a_t = torch.tensor([a], device=device)
        sig = script.buxton_signal(c_t.unsqueeze(-1), a_t.unsqueeze(-1), script.PLD_SEC.to(device), script.LD_SEC.to(device))
        signals_for_cbf.append(sig.cpu().numpy()[0])
    signals_for_cbf = np.array(signals_for_cbf)
    
    # Plot mean signal vs ATT just to visualize
    mean_sig = signals_for_cbf.mean(axis=1)
    plt.plot(att_grid, mean_sig, label=f'CBF={c}')

plt.xlabel('ATT (s)')
plt.ylabel('Mean Signal across 4 PLDs')
plt.title('4-PLD Signal vs ATT')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("diagnostics/identifiability_signal_vs_att.png")
plt.close()

# Sensitivity analysis (Jacobian)
c_val = 50.0
a_val = 1.5
c_t = torch.tensor([[c_val]], device=device, requires_grad=True)
a_t = torch.tensor([[a_val]], device=device, requires_grad=True)
sig = script.buxton_signal(c_t, a_t, script.PLD_SEC.to(device), script.LD_SEC.to(device))

J = []
for i in range(4): # 4 PLDs
    c_t.grad, a_t.grad = None, None
    sig[0, i].backward(retain_graph=True)
    J.append([c_t.grad.item(), a_t.grad.item()])

J = np.array(J)
print(f"Jacobian at CBF=50, ATT=1.5:\n[dSig/dCBF, dSig/dATT]\n{J}")
U, S, V = np.linalg.svd(J, full_matrices=False)
print(f"Singular values of Jacobian: {S}")
print(f"Condition number: {S[0]/S[1]:.2f}")
print("Column norms (Sensitivity):")
print(f"|dSig/dCBF|: {np.linalg.norm(J[:, 0]):.4f}")
print(f"|dSig/dATT|: {np.linalg.norm(J[:, 1]):.4f}")
print("Note: If the condition number is high, or ATT sensitivity is near zero, the problem is ill-posed.")

print("\nDIAGNOSTIC COMPLETE.")
