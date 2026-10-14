import os
import sys
import json
import hashlib
import torch
import numpy as np
import pandas as pd
from types import SimpleNamespace
from rl_4pld_experiment import SACAgent, SACConfig, ASLEnvironment, set_seed, buxton_signal, ccc

def sha256_file(path):
    if not os.path.exists(path): return "MISSING"
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def verify_files():
    print("--- 1 & 3: File Hashes and Copy Check ---")
    script = "rl_4pld_experiment.py"
    config = "experiments/rl_sac_4pld/config.json"
    ckpt500 = "experiments/rl_sac_4pld/sac_asl_it500.pth"
    ckpt_final = "experiments/rl_sac_4pld/sac_asl_4pld_final.pth"
    
    hashes = {
        "script": sha256_file(script),
        "config": sha256_file(config),
        "sac_asl_it500.pth": sha256_file(ckpt500),
        "sac_asl_4pld_final.pth": sha256_file(ckpt_final)
    }
    
    with open("verification/code_hashes.txt", "w") as f:
        for k, v in hashes.items():
            f.write(f"{k}: {v}\n")
            
    is_copy = (hashes["sac_asl_it500.pth"] == hashes["sac_asl_4pld_final.pth"]) and (hashes["sac_asl_it500.pth"] != "MISSING")
    if is_copy:
        print("final checkpoint is merely a copy of iteration-500 checkpoint")
        with open("verification/code_hashes.txt", "a") as f:
            f.write("\nNOTE: final checkpoint is merely a copy of iteration-500 checkpoint\n")
    return hashes

def inspect_checkpoint():
    print("--- 2: Checkpoint Inspection ---")
    ckpt_path = "experiments/rl_sac_4pld/sac_asl_it500.pth"
    if not os.path.exists(ckpt_path): return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    inspection = {
        "keys": list(ckpt.keys()),
        "actor_params": sum(p.numel() for p in ckpt["actor"].values()),
        "critic_params": sum(p.numel() for p in ckpt["critic"].values()),
        "log_alpha_value": ckpt["log_alpha"].item() if "log_alpha" in ckpt else None,
        "actor_layers": {k: {"shape": list(v.shape), "min": v.min().item(), "max": v.max().item(), "mean": v.mean().item()} for k, v in list(ckpt["actor"].items())[:5]}
    }
    with open("verification/checkpoint_inspection.json", "w") as f:
        json.dump(inspection, f, indent=2)

def verify_training_log():
    print("--- 4: Training Log Verification ---")
    log_path = "experiments/rl_sac_4pld/train_log.csv"
    if not os.path.exists(log_path): return
    df = pd.read_csv(log_path)
    
    with open("verification/training_log_verification.txt", "w") as f:
        f.write(f"Num rows: {len(df)}\n")
        f.write(f"Min iteration: {df['iteration'].min()}\n")
        f.write(f"Max iteration: {df['iteration'].max()}\n")
        f.write(f"Has iter 500: {500 in df['iteration'].values}\n")
        f.write(f"Has iter 1000: {1000 in df['iteration'].values}\n")
        
        expected_iters = list(range(10, df['iteration'].max() + 10, 10))
        actual_iters = df['iteration'].tolist()
        f.write(f"Continuous iterations: {expected_iters == actual_iters}\n\n")
        
        f.write("First 10 rows:\n")
        f.write(df.head(10).to_string() + "\n\n")
        f.write("Last 10 rows:\n")
        f.write(df.tail(10).to_string() + "\n")

def get_args():
    with open("experiments/rl_sac_4pld/config.json", "r") as f:
        c = json.load(f)
    c["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return SimpleNamespace(**c)

def generate_test_populations(args):
    print("--- 7: Fixed Test Sets ---")
    snrs = [2, 5, 10, 15, 20]
    n_pairs = 2000
    for snr in snrs:
        set_seed(42 + snr)
        env = ASLEnvironment(n_pairs, args.max_steps, snr, snr, 0.0, "coarse_grid", args.coarse_grid_size, args.lm_handoff_frac, args.device)
        env.reset()
        np.savez(f"verification/test_population_SNR_{snr}.npz",
                 true_cbf=env.cbf_true.cpu().numpy(),
                 true_att=env.att_true.cpu().numpy(),
                 obs_signal=env.obs_signal.cpu().numpy())

def run_inference(agent, args, snr, run_id):
    data = np.load(f"verification/test_population_SNR_{snr}.npz")
    set_seed(42 + snr) # FORCE EXACT SAME SEED AS GENERATION
    n_pairs = len(data["true_cbf"])
    env = ASLEnvironment(n_pairs, args.max_steps, snr, snr, 0.0, "coarse_grid", args.coarse_grid_size, args.lm_handoff_frac, args.device)
    state = env.reset() # NOW reset() generates the exact same cbf_true, att_true, obs_signal, AND cbf_est, att_est
    
    for _ in range(args.max_steps):
        action = agent.act(state, deterministic=True)
        state, _, _, _ = env.step(action)
        
    pred_cbf = env.cbf_est.cpu().numpy()
    pred_att = env.att_est.cpu().numpy()
    
    np.savez(f"verification/predictions_SNR_{snr}_{run_id}.npz",
             true_cbf=data["true_cbf"], pred_cbf=pred_cbf,
             true_att=data["true_att"], pred_att=pred_att)
             
    c_t, c_e = data["true_cbf"], pred_cbf
    a_t, a_e = data["true_att"]/1000.0, pred_att/1000.0
    
    metrics = {
        "cbf_mae": np.abs(c_t - c_e).mean(),
        "cbf_rmse": np.sqrt(np.mean((c_t - c_e)**2)),
        "cbf_ccc": ccc(torch.tensor(c_t), torch.tensor(c_e)),
        "att_mae": np.abs(a_t - a_e).mean(),
        "att_rmse": np.sqrt(np.mean((a_t - a_e)**2)),
        "att_ccc": ccc(torch.tensor(a_t), torch.tensor(a_e)),
    }
    return metrics, pred_cbf, pred_att

def verify_inference():
    print("--- 8-12: Determinism & Metrics ---")
    args = get_args()
    cfg = SACConfig(11, 2, args.hidden, args.gamma, args.tau, args.lr, device=args.device)
    
    agentA = SACAgent(cfg)
    agentA.load("experiments/rl_sac_4pld/sac_asl_it500.pth", map_location=args.device)
    
    results_A = {}
    for snr in [2, 5, 10, 15, 20]:
        metrics, pC_A, pA_A = run_inference(agentA, args, snr, "runA")
        results_A[snr] = metrics
        
        # Run B (same agent instance)
        _, pC_B, pA_B = run_inference(agentA, args, snr, "runB")
        
        # Run C (fresh agent instance, loaded checkpoint)
        agentC = SACAgent(cfg)
        agentC.load("experiments/rl_sac_4pld/sac_asl_it500.pth", map_location=args.device)
        _, pC_C, pA_C = run_inference(agentC, args, snr, "runC")
        
        diff_B = max(np.abs(pC_A - pC_B).max(), np.abs(pA_A - pA_B).max())
        diff_C = max(np.abs(pC_A - pC_C).max(), np.abs(pA_A - pA_C).max())
        print(f"SNR {snr} | Max diff A vs B: {diff_B:.2e} | Max diff A vs C: {diff_C:.2e}")
        
    # Compare against old csv
    old_df = pd.read_csv("experiments/rl_sac_4pld/snr_results.csv")
    comparison = []
    for snr in [2, 5, 10, 15, 20]:
        old_row = old_df[old_df['snr'] == snr].iloc[0]
        new_m = results_A[snr]
        comparison.append({
            "snr": snr,
            "old_cbf_mae": old_row['cbf_mae'], "new_cbf_mae": new_m['cbf_mae'], "diff_cbf_mae": abs(old_row['cbf_mae'] - new_m['cbf_mae']),
            "old_att_mae": old_row['att_mae'], "new_att_mae": new_m['att_mae'], "diff_att_mae": abs(old_row['att_mae'] - new_m['att_mae'])
        })
    pd.DataFrame(comparison).to_csv("verification/reported_vs_recomputed.csv", index=False)

def verify_progression():
    print("--- 14 & 15: Progression Check ---")
    args = get_args()
    cfg = SACConfig(11, 2, args.hidden, args.gamma, args.tau, args.lr, device=args.device)
    prog = []
    
    prev_checksum = None
    for it in [100, 200, 300, 400, 500]:
        path = f"experiments/rl_sac_4pld/sac_asl_it{it}.pth"
        exists = os.path.exists(path)
        if not exists: continue
        
        agent = SACAgent(cfg)
        agent.load(path, map_location=args.device)
        checksum = sum(p.sum().item() for p in agent.actor.parameters())
        
        metrics, _, _ = run_inference(agent, args, 20, f"it{it}")
        prog.append({
            "iteration": it, "exists": exists, "checksum": checksum,
            "changed_from_prev": checksum != prev_checksum if prev_checksum is not None else True,
            "cbf_rmse_snr20": metrics["cbf_rmse"]
        })
        prev_checksum = checksum
        
    pd.DataFrame(prog).to_csv("verification/checkpoint_progression.csv", index=False)

def recompute_grid():
    print("--- 13: Parameter Grid Test ---")
    args = get_args()
    cfg = SACConfig(11, 2, args.hidden, args.gamma, args.tau, args.lr, device=args.device)
    agent = SACAgent(cfg)
    agent.load("experiments/rl_sac_4pld/sac_asl_it500.pth", map_location=args.device)
    
    cbf_grid = np.linspace(0, 100, 11).tolist()
    att_grid = np.linspace(0.5, 3.0, 11).tolist()
    
    env = ASLEnvironment(len(cbf_grid) * len(att_grid), args.max_steps, 20.0, 20.0, 0.0, "coarse_grid", args.coarse_grid_size, args.lm_handoff_frac, args.device)
    cbf_mesh, att_mesh = torch.meshgrid(torch.tensor(cbf_grid, device=args.device), torch.tensor(att_grid, device=args.device), indexing="ij")
    state = env.reset()
    # OVERWRITE AFTER RESET:
    env.cbf_true = cbf_mesh.flatten()
    env.att_true = att_mesh.flatten() * 1000.0
    env.clean_signal = buxton_signal(env.cbf_true.unsqueeze(-1), env.att_true.unsqueeze(-1), env.pld, env.ld)
    env.obs_signal = env.clean_signal 
    # RE-INITIALIZE OBS-DEPENDENT ESTIMATES:
    env._coarse_grid_init(args.coarse_grid_size)
    env.step_count = torch.zeros(len(env.cbf_true), device=args.device)
    state = env._make_state(env._predicted_signal())
    
    for _ in range(args.max_steps):
        state, _, _, _ = env.step(agent.act(state, deterministic=True))
        
    results = []
    c_t = env.cbf_true.cpu().numpy()
    a_t = (env.att_true / 1000.0).cpu().numpy()
    c_e = env.cbf_est.cpu().numpy()
    a_e = (env.att_est / 1000.0).cpu().numpy()
    
    for i in range(len(c_t)):
        results.append(f"{c_t[i]},{a_t[i]},{c_e[i]},{a_e[i]},{abs(c_t[i]-c_e[i])},{abs(a_t[i]-a_e[i])}")
        
    with open("verification/recomputed_parameter_grid.csv", "w") as f:
        f.write("true_cbf,true_att,est_cbf,est_att,cbf_err,att_err\n")
        f.write("\n".join(results) + "\n")

if __name__ == "__main__":
    verify_files()
    inspect_checkpoint()
    verify_training_log()
    args = get_args()
    generate_test_populations(args)
    verify_inference()
    verify_progression()
    recompute_grid()
    print("VERIFICATION SCRIPT COMPLETE.")
