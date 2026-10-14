# 4-PLD SAC Experiment Report

## 1. Objective
The objective was to conduct a controlled research experiment to train a new SAC model using 4 explicitly defined PLDs `[1.525, 2.025, 2.525, 3.025]` while isolating the effect of the acquisition configuration without changing the underlying architecture, reward, state representation, or physics model.

## 2. Why 4 PLDs
This experiment aims to simulate a 4-PLD acquisition protocol corresponding to the specific delays `[1.525, 2.025, 2.525, 3.025]` seconds, providing a synthetic baseline for evaluating parameter recovery under these timings without overwriting the 5-PLD baseline.

## 3. 4-PLD Protocol & Simulation Requirements
- **PLD:** `[1.525, 2.025, 2.525, 3.025]` seconds (As specified for the controlled experiment).
- **LD (Labeling Duration):** **COULD NOT BE ESTABLISHED.**
- **Noise Model:** Rician noise across multiple SNR levels.
- **Buxton Simulation:** Single-compartment kinetic model using CBF `20-90` ml/100g/min and ATT `500-3000` ms.

## 4. State & Action Architecture Verification (Pre-training checks)
- **State Representation:** 11 Dimensions 
  (`4 x observed signal`, `4 x predicted signal`, `1 x norm CBF`, `1 x norm ATT`, `1 x time fraction`)
- **Action Representation:** 2 Dimensions (`CBF increment`, `ATT increment`)
- **Actor Architecture:** Input = 11, Output = 2, Hidden = 256. Twin Critics remain consistent.

---

## 5. EXPERIMENT HALTED: Missing LD Constraint
Following the explicit instructions: *"If the LD cannot be established from the available source material, STOP and report that exact issue rather than silently inventing an LD."*

The execution of this experiment has been **HALTED** prior to training because an appropriate fixed Labeling Duration (LD) for this specific `[1.525, 2.025, 2.525, 3.025]` PLD sequence is not defined anywhere in the existing project, source material, or test data metadata. 

Specifically:
1. The Ishida et al. 2023 paper explicitly uses `LD = [1.333, 1.333, 1.333, 4.000]` for a completely different set of PLDs (`0.700` and `3.366`). It provides no fixed LD mapping for `1.525-3.025`.
2. The project's reference condition uses `REF_LD_SEC = 1.800`, but this is a calibration anchor for mathematical noise generation, not an acquisition protocol defined for this experiment.
3. No standard fixed LD (e.g., 1500 ms) is documented in the repository for these files.

I have refused to silently invent an LD (such as blindly assuming `1.5s` or `1.8s`) to bypass this check. 

---

### Impact on Required Tasks (Tasks 5-18)
Because the protocol definition failed the strict verification constraint in Task 2, the training, validation, SNR robustness sweeps, parameter-recovery test, checkpoint round-trip, and clinical inference could not be executed. No model was trained, no checkpoint was overwritten, and the existing 5-PLD baseline remains safely unmodified.

### Path Forward
I have successfully created the requested isolated 4-PLD experiment pipeline: `rl_4pld_experiment.py`. It explicitly enforces the 4-PLD sequence and the state dimensions (`dim=11`), but requires the LD to be explicitly supplied.

To unblock this controlled experiment and generate the required numerical outputs and checkpoint (`sac_asl_4pld_final.pth`), you can execute the script with the exact Labeling Duration (in seconds) once it is known:

```powershell
python rl_4pld_experiment.py --ld <EXACT_LABEL_DURATION>
```

