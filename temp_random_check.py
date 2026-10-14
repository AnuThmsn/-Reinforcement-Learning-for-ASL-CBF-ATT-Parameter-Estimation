import torch
import numpy as np
from rl_4pld_experiment import SACAgent, SACConfig
import run_verification
args = run_verification.get_args()
cfg = SACConfig(11, 2, args.hidden, args.gamma, args.tau, args.lr, device=args.device)
rand_agent = SACAgent(cfg)
metrics, _, _ = run_verification.run_inference(rand_agent, args, 20, "random")
print(f"Random Init SNR 20 CBF MAE: {metrics['cbf_mae']:.3f}")
print(f"Random Init SNR 20 ATT MAE: {metrics['att_mae']:.3f}")

