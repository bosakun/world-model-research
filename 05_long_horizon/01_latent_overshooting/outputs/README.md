# Generated Long-Horizon Evidence

Generated with seed 47 and `controlled-oscillator-v1`.

| File | Meaning |
|---|---|
| `loss_curve.png` | train/validation overshooting objective and one-step term |
| `long_rollout.png` | true versus 30-step predicted position/velocity |
| `compounding_error.png` | autoregressive MSE by horizon with one-step reference |
| `training_history.csv` | epoch-level losses |
| `training_summary.json` | configuration and final training results |
| `evaluation_metrics.json` | distance/horizon errors and reproducibility metadata |
| `checkpoint.pt` | generated format-1 checkpoint; gitignored |

Regenerate with:

```bash
.venv/bin/python 05_long_horizon/01_latent_overshooting/train.py
.venv/bin/python 05_long_horizon/01_latent_overshooting/evaluate.py
```
