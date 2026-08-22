# Generated Ensemble-Uncertainty Evidence

Generated with seed 41, bootstrap seed 42, and five members.

| File | Meaning |
|---|---|
| `loss_curve.png` | member-mean train/validation Gaussian NLL |
| `epistemic_map.png` | ensemble mean-disagreement std across state space |
| `uncertainty_decomposition.png` | true/learned aleatoric, epistemic, and total std |
| `trajectory_sampling.png` | TS∞ and TS1 sampled trajectory particles |
| `training_history.csv` | per-epoch ensemble/member losses |
| `training_summary.json` | reproducibility and training metadata |
| `evaluation_metrics.json` | calibration, ID/OOD disagreement, rollout metadata |
| `checkpoint.pt` | local generated ensemble checkpoint; gitignored |

Regenerate with:

```bash
.venv/bin/python 04_uncertainty/02_ensemble/train.py
.venv/bin/python 04_uncertainty/02_ensemble/evaluate.py
```
