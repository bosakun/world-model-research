# Generated Temporal-Abstraction Evidence

Generated with seed 53 and fixed five-action chunks.

| File | Meaning |
|---|---|
| `loss_curve.png` | train/validation macro transition MSE |
| `macro_rollout.png` | true versus predicted boundary position/velocity |
| `macro_error.png` | boundary MSE at primitive-equivalent horizons |
| `training_history.csv` | epoch losses |
| `training_summary.json` | reproducibility metadata |
| `evaluation_metrics.json` | macro and primitive-equivalent horizon metrics |
| `checkpoint.pt` | generated format-1 checkpoint; gitignored |

Regenerate with:

```bash
.venv/bin/python 05_long_horizon/02_temporal_abstraction/train.py
.venv/bin/python 05_long_horizon/02_temporal_abstraction/evaluate.py
```
