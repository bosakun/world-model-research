# Generated Transformer Memory Evidence

Generated on 2026-08-22 with seed 29 and `config.py` defaults.

| File | Meaning |
|---|---|
| `loss_curve.png` | train/validation objective and future prediction terms |
| `one_step_prediction.png` | true next images versus teacher-forced causal predictions |
| `sequence_rollout.png` | true sequence versus autoregressive predicted-latent rollout |
| `attention_map.png` | final-layer, head-averaged causal attention for one sequence |
| `rollout_error.png` | plain pixel MSE by autoregressive horizon |
| `training_history.csv` | epoch-level loss components |
| `training_summary.json` | reproducibility and final training metadata |
| `evaluation_metrics.json` | one-step/rollout/state-head/shape metrics |
| `checkpoint.pt` | local format-version-1 checkpoint; generated and gitignored |

Regenerate from repository root:

```bash
.venv/bin/python 03_memory/04_transformer_memory/train.py
.venv/bin/python 03_memory/04_transformer_memory/evaluate.py
```
