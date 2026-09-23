import json

notebook = {
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_code(source):
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    })

def add_md(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    })

add_md("# RL-ASL 4-PLD Clinical Inference\nThis notebook loads the trained 4-PLD SAC model and runs inference on real clinical NIfTI datasets, displaying the results directly in the notebook.")

add_code("""%matplotlib inline
import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from rl_asl_4pld import ASLEnvironment, SACConfig, SACAgent, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

args = type('Args', (), {
    'max_steps': 12,
    'hidden': 256,
    'gamma': 0.95,
    'tau': 0.005,
    'lr': 3e-4,
    'init_mode': 'coarse_grid',
    'coarse_grid_size': 31,
    'lm_handoff_frac': 0.8,
    'device': device
})()

cfg = SACConfig(11, 2, args.hidden, args.gamma, args.tau, args.lr, device=device)
agent = SACAgent(cfg)
agent_path = os.path.join("experiments", "rl_sac_4pld_trained", "notebook_results", "sac_asl_final.pth")
if os.path.exists(agent_path):
    agent.load(agent_path, map_location=device)
    print("Loaded trained SAC agent from:", agent_path)
else:
    print("WARNING: Trained agent weights not found. Running with untrained weights.")
""")

add_code("""DATA_ROOT = "test data"
OUTPUT_ROOT = "rl_inference_output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 1. Robust File Discovery (Directly in test data folder)
all_files = os.listdir(DATA_ROOT)
perf_files = sorted([f for f in all_files if 'perfusion' in f.lower() and f.endswith('.nii.gz')])
m0_files = sorted([f for f in all_files if 'm0' in f.lower() and f.endswith('.nii.gz')])
mask_files = [f for f in all_files if 'mask' in f.lower() and f.endswith('.nii.gz')]

if len(perf_files) != 4 or len(m0_files) != 4 or len(mask_files) == 0:
    print(f"Error: Missing files in {DATA_ROOT}. Found {len(perf_files)} Perfusion and {len(m0_files)} M0 files.")
else:
    print(f"Found exactly 4 Perfusion and 4 M0 files. Starting processing...")
    
    # 2. Load Data
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

    if len(coords) > 0:
        # 3. Prepare features
        features = np.empty((len(coords), 4), dtype=np.float32)
        for i, (x, y, z) in enumerate(coords):
            dm = deltaM_vols[x, y, z, :]
            m0 = M0_mean[x, y, z]
            # Match physical fraction * RL scale
            features[i] = (dm / (100.0 * m0)) * 100_000.0

        batch_size = 5000
        cbf_preds = np.zeros(len(coords), dtype=np.float32)
        att_preds = np.zeros(len(coords), dtype=np.float32)

        for b in range(0, len(coords), batch_size):
            b_features = torch.tensor(features[b:b+batch_size], dtype=torch.float32, device=device)
            B = b_features.shape[0]

            env = ASLEnvironment(batch_size=B, max_steps=args.max_steps, snr_min=10, snr_max=10, 
                                 clean_fraction=0.0, init_mode=args.init_mode, 
                                 coarse_grid_size=args.coarse_grid_size, lm_handoff_frac=args.lm_handoff_frac, 
                                 device=device)
            
            env.reset()
            env.obs_signal = b_features
            if args.init_mode == 'coarse_grid':
                env._coarse_grid_init(args.coarse_grid_size)
                
            state = env._make_state(env._predicted_signal())

            with torch.no_grad():
                for _ in range(args.max_steps):
                    action = agent.act(state, deterministic=True)
                    state, _, _, _ = env.step(action)

            cbf_preds[b:b+batch_size] = env.cbf_est.cpu().numpy()
            att_preds[b:b+batch_size] = (env.att_est.cpu().numpy() / 1000.0)

        # 4. Map back to 3D volume
        CBF_map = np.full(sz, np.nan, dtype=np.float32)
        ATT_map = np.full(sz, np.nan, dtype=np.float32)
        for i, (x, y, z) in enumerate(coords):
            CBF_map[x, y, z] = cbf_preds[i]
            ATT_map[x, y, z] = att_preds[i]

        CBF_save = np.nan_to_num(CBF_map, nan=0.0)
        ATT_save = np.nan_to_num(ATT_map, nan=0.0)
        nib.save(nib.Nifti1Image(CBF_save, affine), os.path.join(OUTPUT_ROOT, f"rl_cbf_patient.nii.gz"))
        nib.save(nib.Nifti1Image(ATT_save, affine), os.path.join(OUTPUT_ROOT, f"rl_att_patient.nii.gz"))

        # 5. Visualization
        fig, axes = plt.subplots(3, 2, figsize=(10, 14))
        fig.patch.set_facecolor('white')
        fig.suptitle('RL ASL Results - Patient Inference\\nCBF (ml/100g/min) and ATT (s)', fontsize=16, fontweight='bold', y=0.98)

        global_cbf_mean = np.nanmean(CBF_map[valid_mask])
        global_cbf_median = np.nanmedian(CBF_map[valid_mask])
        global_att_mean = np.nanmean(ATT_map[valid_mask])
        global_att_median = np.nanmedian(ATT_map[valid_mask])

        slices = [15, 20, 25]
        for i, z in enumerate(slices):
            if z >= sz[2]: continue

            cbf_slice = np.rot90(CBF_map[:, :, z])
            cbf_slice_masked = np.ma.masked_where(np.isnan(cbf_slice) | (cbf_slice == 0), cbf_slice)
            cbf_min = cbf_slice_masked.min() if cbf_slice_masked.count() > 0 else 0.0
            cbf_max = cbf_slice_masked.max() if cbf_slice_masked.count() > 0 else 0.0

            im_cbf = axes[i, 0].imshow(cbf_slice_masked, cmap='jet', vmin=CBF_MIN, vmax=CBF_MAX)
            axes[i, 0].set_title(f'CBF Slice {z} ({cbf_min:.1f}-{cbf_max:.1f})\\nmean={global_cbf_mean:.1f} median={global_cbf_median:.1f}', color='black', fontsize=11)
            axes[i, 0].axis('off')
            cb_cbf = fig.colorbar(im_cbf, ax=axes[i, 0], fraction=0.046, pad=0.04)

            att_slice = np.rot90(ATT_map[:, :, z])
            att_slice_masked = np.ma.masked_where(np.isnan(att_slice) | (att_slice == 0), att_slice)
            att_min = att_slice_masked.min() if att_slice_masked.count() > 0 else 0.0
            att_max = att_slice_masked.max() if att_slice_masked.count() > 0 else 0.0

            im_att = axes[i, 1].imshow(att_slice_masked, cmap='jet', vmin=ATT_MIN/1000.0, vmax=ATT_MAX/1000.0)
            axes[i, 1].set_title(f'ATT Slice {z} ({att_min:.2f}-{att_max:.2f} s)\\nmean={global_att_mean:.2f} median={global_att_median:.2f}', color='black', fontsize=11)
            axes[i, 1].axis('off')
            cb_att = fig.colorbar(im_att, ax=axes[i, 1], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(OUTPUT_ROOT, "rl_slices_patient.png"), facecolor='white', bbox_inches='tight', dpi=150)
        
        # Display the image inside the Jupyter Notebook cell!
        plt.show()
        
        print(f"\\n--- PROCESSING COMPLETE ---")
    else:
        print("No valid voxels found.")
""")

with open("D:/reinforcement learning/asl-github-repo/rl_4pld_inference.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
