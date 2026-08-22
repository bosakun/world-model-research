# Generated RSSM Evidence

Generated on 2026-08-22 with seed 23 and the default configuration in `config.py`.

| File | Meaning |
|---|---|
| `loss_curve.png` | train/validation objective, validation weighted reconstruction, and raw KL |
| `reconstruction.png` | true partial observations versus posterior-mean reconstructions |
| `latent_rollout.png` | one posterior seed followed by six prior-only imagined frames |
| `rollout_error.png` | plain pixel MSE for each prior rollout horizon |
| `training_history.csv` | epoch-level objective components |
| `training_summary.json` | reproducibility and final training metadata |
| `evaluation_metrics.json` | reconstruction, state-head, rollout, shape, and parameter metrics |
| `checkpoint.pt` | format-version-1 local checkpoint; gitignored because it is generated binary data |

Regenerate from repository root:

```bash
uv run python 03_memory/03_rssm/train.py
uv run python 03_memory/03_rssm/evaluate.py
```
