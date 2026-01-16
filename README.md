
---

## 4. Methodology

### Algorithm
- Proximal Policy Optimisation (PPO)
- Stable-Baselines3 implementation

### Observations
- RGB images resized to **84 × 84**
- Optional geometric vector:
  - Distance to goal  
  - Initial distance  
  - Normalised progress  

### Architectures
- **Baseline**: CNN encoder + PPO policy/value heads  
- **Geometry-aware**: CNN encoder + pose embedding + feature-level modulation  

### Training Regime
1. Single-scene learning (FP1)  
2. Multi-scene curriculum (FP1–FP5)  
3. Out-of-distribution evaluation (FP6–FP8)  

### Metrics
- Success Rate  
- Episodic Return  
- Distance-based SPL proxy  

---

## 5. Experimental Environment

| Component | Specification |
|--------|---------------|
| Simulator | AI2-THOR |
| OS | Ubuntu 22.04 (WSL2) |
| GPU | NVIDIA RTX 5070 (8GB) |
| CPU | Intel i7-14650HX |
| Python | 3.12 |
| PyTorch | 2.3 |
| RL Library | Stable-Baselines3 |

---

## 6. Reproducibility

Reproducibility is a core design objective:

- Fixed random seeds  
- Identical hyperparameters across agents  
- Deterministic evaluation  
- Explicit separation of training and evaluation  
- Version-controlled software stack  

See `07_reproduce/README_REPRODUCE.txt` for step-by-step instructions.

---

## 7. Results Summary

Key findings (see dissertation for full analysis):

- **FP1 Success Rate**:  
  Geometry-aware PPO (82.5%) vs Baseline PPO (62.5%)

- **Training Efficiency**:  
  Baseline required significantly more steps to reach comparable performance

- **Generalisation**:  
  Geometry-aware agents showed improved stability under curriculum learning

- **OOD Performance**:  
  Modest but consistent gains; severe distribution shift remains challenging

---

## 8. Limitations

- No explicit memory or mapping module  
- Approximate SPL metric  
- No domain randomisation  
- Single navigation task (point-goal)  

These limitations are analysed in detail in Chapter 6 of the dissertation.

---

## 9. Reproducibility Checklist (NeurIPS / ICLR Style)

- [x] Clear problem definition  
- [x] Explicit algorithm description  
- [x] Fixed random seeds  
- [x] Identical hyperparameters across models  
- [x] Deterministic evaluation  
- [x] Multiple evaluation environments  
- [x] Failure cases discussed  
- [ ] Trained checkpoints released (available on request)

---

## 10. Citation

If you use or build upon this work, please cite:

> Nkume, F. U. (2026). *Geometry-Aware Reinforcement Learning for Visual Navigation Agents*.  
> MSc Dissertation, University of Greater Manchester.

---

## 11. License and Usage

This repository is intended for **academic and research use**.  
Please contact the author before commercial or derivative use.
