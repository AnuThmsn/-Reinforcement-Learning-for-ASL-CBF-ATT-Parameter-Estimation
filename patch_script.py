import sys, json, os, re

with open(r'D:\reinforcement learning\asl-github-repo\rl_4pld_experiment.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace everything from def main(): downwards
idx = content.find("def main():")
if idx == -1:
    print("Could not find def main():")
    sys.exit(1)

main_block = """
def parameter_grid_test(agent, args, cbf_grid, att_grid):
    results = []
    env = ASLEnvironment(len(cbf_grid) * len(att_grid), args.max_steps, 20.0, 20.0, 0.0,
                         "coarse_grid", args.coarse_grid_size, args.lm_handoff_frac, args.device)
    cbf_mesh, att_mesh = torch.meshgrid(torch.tensor(cbf_grid, device=args.device), 
                                        torch.tensor(att_grid, device=args.device), indexing="ij")
    env.cbf_true = cbf_mesh.flatten()
    env.att_true = att_mesh.flatten() * 1000.0  # att_grid is in seconds, convert to msec
    
    env.clean_signal = buxton_signal(env.cbf_true.unsqueeze(-1),
                                     env.att_true.unsqueeze(-1),
                                     env.pld, env.ld)
    env.obs_signal = env.clean_signal # Clean grid test
    state = env.reset()
    for _ in range(args.max_steps):
        state, _, _, _ = env.step(agent.act(state, deterministic=True))
    
    cbf_t = env.cbf_true.cpu().numpy()
    cbf_e = env.cbf_est.cpu().numpy()
    att_t = (env.att_true / 1000.0).cpu().numpy()
    att_e = (env.att_est / 1000.0).cpu().numpy()
    
    for i in range(len(cbf_t)):
        results.append(f"{cbf_t[i]},{att_t[i]},{cbf_e[i]},{att_e[i]}")
    
    with open(os.path.join(args.out_dir, "parameter_grid_results.csv"), "w") as f:
        f.write("true_cbf,true_att,est_cbf,est_att\\n")
        f.write("\\n".join(results) + "\\n")

def checkpoint_validation(args, first_results):
    print("\\n--- Phase 11: Checkpoint Validation ---")
    cfg = SACConfig(11, 2, args.hidden, args.gamma, args.tau, args.lr, device=args.device)
    new_agent = SACAgent(cfg)
    new_agent.load(os.path.join(args.out_dir, "sac_asl_4pld_final.pth"), map_location=args.device)
    
    new_results = benchmark(new_agent, args, snrs=(20,), n_pairs=2000)
    
    match = True
    if abs(new_results["20"]["cbf_rmse"] - first_results["20"]["cbf_rmse"]) > 1e-4:
        match = False
            
    print(f"Checkpoint round-trip match: {match}")
    return match

def main():
    parser = argparse.ArgumentParser(description="Reproducible ASL SAC experiment runner")
    parser.add_argument("--config", required=False, help="JSON configuration path")
    parser.add_argument("--evaluate-only", action="store_true")
    parsed = parser.parse_args()

    supplied = {}
    if parsed.config:
        with open(parsed.config) as f:
            supplied = json.load(f)
            
    DEFAULT_CONFIG.update({
        "out_dir": "experiments/rl_sac_4pld",
        "iterations": 1000,
        "init_mode": "coarse_grid"
    })
    
    config = DEFAULT_CONFIG | supplied
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config["seed"])
    args = SimpleNamespace(**config)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    cfg = SACConfig(11, 2, args.hidden, args.gamma, args.tau, args.lr, device=args.device)
    agent = SACAgent(cfg)
    
    if not parsed.evaluate_only:
        agent = train(args)
        agent.save(os.path.join(args.out_dir, "sac_asl_4pld_final.pth"))
    else:
        agent.load(os.path.join(args.out_dir, "sac_asl_4pld_final.pth"), map_location=args.device)
        
    print("\\n--- Phase 13: SNR Robustness ---")
    results = benchmark(agent, args, snrs=(2, 5, 10, 15, 20), n_pairs=args.eval_pairs)
    
    with open(os.path.join(args.out_dir, "snr_results.csv"), "w") as f:
        f.write("snr,cbf_mae,cbf_rmse,cbf_ccc,att_mae,att_rmse,att_ccc\\n")
        for snr, m in results.items():
            f.write(f"{snr},{m['cbf_mae']},{m['cbf_rmse']},{m['cbf_ccc']},"
                    f"{m['att_mae']},{m['att_rmse']},{m['att_ccc']}\\n")
    print(json.dumps(results, indent=2))
    
    with open(os.path.join(args.out_dir, "evaluation_results.csv"), "w") as f:
        f.write("snr,cbf_mae,cbf_rmse,cbf_ccc,att_mae,att_rmse,att_ccc\\n")
        m = results['20'] # Use highest SNR for base eval
        f.write(f"20,{m['cbf_mae']},{m['cbf_rmse']},{m['cbf_ccc']},"
                f"{m['att_mae']},{m['att_rmse']},{m['att_ccc']}\\n")
                
    print("\\n--- Phase 12: Parameter Grid Test ---")
    cbf_grid = np.linspace(0, 100, 11).tolist()
    att_grid = np.linspace(0.5, 3.0, 11).tolist()
    parameter_grid_test(agent, args, cbf_grid, att_grid)
    
    checkpoint_validation(args, results)

if __name__ == "__main__":
    main()
"""

new_content = content[:idx] + main_block
with open(r'D:\reinforcement learning\asl-github-repo\rl_4pld_experiment.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

