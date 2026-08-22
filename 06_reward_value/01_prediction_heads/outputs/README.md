# Generated Reward/Value/Continuation Evidence

Generated with seed 59 and `goal-navigation-v1`.

| File | Meaning |
|---|---|
| `loss_curve.png` | joint and per-head train/validation losses |
| `prediction_sequence.png` | target/predicted reward, value, continuation over one valid episode |
| `training_history.csv` | epoch loss values |
| `training_summary.json` | configuration/training metadata |
| `evaluation_metrics.json` | RMSE, continuation calibration, counts, entry point |
| `checkpoint.pt` | generated format-1 checkpoint; gitignored |

Regenerate with:

```bash
.venv/bin/python 06_reward_value/01_prediction_heads/train.py
.venv/bin/python 06_reward_value/01_prediction_heads/evaluate.py
```
