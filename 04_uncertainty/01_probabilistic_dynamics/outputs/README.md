# Generated Aleatoric-Uncertainty Evidence

Generated with seed 37 and `heteroscedastic-point-v1`.

| File | Meaning |
|---|---|
| `loss_curve.png` | train/validation Gaussian NLL |
| `aleatoric_std.png` | known versus predicted horizontal transition std over state x |
| `sampled_rollouts.png` | 64 stochastic model trajectories, true trajectory, and mean trajectory |
| `training_history.csv` | epoch-level objective values |
| `training_summary.json` | training/reproducibility metadata |
| `evaluation_metrics.json` | RMSE, NLL, coverage, std correlation, rollout metadata |
| `checkpoint.pt` | local generated format-1 checkpoint; gitignored |

Regenerate with:

```bash
.venv/bin/python 04_uncertainty/01_probabilistic_dynamics/train.py
.venv/bin/python 04_uncertainty/01_probabilistic_dynamics/evaluate.py
```
